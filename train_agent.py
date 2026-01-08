"""Training harness for PortfolioEnv using Stable-Baselines3.

Usage examples:
  python train_agent.py --price-csv data/prices.csv --n-assets 3 --total-timesteps 10000 --out model.zip

The script supports selecting the RL algorithm via `--algo` (default PPO).
"""
import argparse
import os
import pandas as pd
import numpy as np
import random
import torch
from typing import Callable
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from portfolio_env import PortfolioEnv


def make_env_factory(
    price_csv,
    n_assets,
    transaction_cost,
    reward_alpha,
    rank: int = 0,
    seed: int | None = None,
    vol_warmup_steps: int = 20,
    turnover_penalty: float = 0.0,
    window: int = 20,
    walk_forward_pct: float = 0.0,
    action_smoothing: float = 0.0,
    reward_scale: float = 1.0,
):
    """Factory function that creates PortfolioEnv instances.
    
    Patch 3: Does NOT call env.reset() - lets SB3 handle initial reset.
    Patch 4: Each env worker gets a distinct seed offset (seed + rank) ensuring
    different random start positions across vectorized environments.
    Patch 7: Walk-forward sampling moved to env reset() for regime diversity.
    
    Args:
        price_csv: Path to price data CSV
        n_assets: Number of assets in portfolio
        transaction_cost: Transaction cost per unit turnover
        reward_alpha: Risk aversion coefficient for variance penalty
        rank: Worker rank for seed offset (0, 1, 2, ...)
        seed: Base seed for reproducibility (None = random)
        vol_warmup_steps: Number of steps for volatility warm-up
        turnover_penalty: Penalty coefficient for turnover
        window: Lookback window for feature calculation
        walk_forward_pct: Fraction of data to use for walk-forward (0=disabled)
        action_smoothing: EMA coefficient for action smoothing (Patch 4)
        reward_scale: Soft reward scaling factor (Patch 5)
    """
    df = None
    if price_csv:
        df = pd.read_csv(price_csv, index_col=0, parse_dates=True)

    def _init():
        # Patch 4: Each env gets its own seed offset for determinism when requested
        # This ensures vectorized envs start at different random positions
        env_seed = None
        if seed is not None:
            env_seed = int(seed) + int(rank)
            # set global seeds for the env creation context
            np.random.seed(env_seed)
            random.seed(env_seed)
            try:
                torch.manual_seed(env_seed)
            except Exception:
                pass

        # Patch 9: Walk-forward window sampling
        # Slice data to different windows per worker for better generalization
        env_df = df
        if walk_forward_pct > 0 and df is not None:
            n_rows = len(df)
            window_size = int(n_rows * walk_forward_pct)
            if window_size > window + 10:  # Ensure enough data for an episode
                # Each worker gets a different starting offset
                max_start = n_rows - window_size
                if max_start > 0:
                    rng = np.random.default_rng(env_seed if env_seed else rank)
                    start_idx = rng.integers(0, max_start + 1)
                    env_df = df.iloc[start_idx:start_idx + window_size].copy()

        # Patch 3: Do NOT call env.reset() here - let SB3 handle the initial reset
        # Double reset breaks reproducibility and advances RNG incorrectly
        # SB3 will call reset() exactly once when it needs to
        env = PortfolioEnv(
            n_assets=n_assets,
            price_df=env_df,
            transaction_cost=transaction_cost,
            reward_alpha=reward_alpha,
            vol_warmup_steps=vol_warmup_steps,
            turnover_penalty=turnover_penalty,
            window=window,
            action_smoothing=action_smoothing,
            reward_scale=reward_scale,
        )
        
        # Patch 8: Store env_seed for SB3 to use via env.reset(seed=...)
        # SB3's VecEnv will handle seeding properly
        env._init_seed = env_seed
        
        return env

    return _init


# =============================================================================
# Patch 7: Learning Rate Warm-up Schedule
# =============================================================================

def linear_warmup_schedule(
    initial_lr: float,
    warmup_steps: int,
    total_timesteps: int,
) -> Callable[[float], float]:
    """Create a learning rate schedule with linear warm-up.
    
    During warm-up phase, LR increases linearly from 0 to initial_lr.
    After warm-up, LR decays linearly to 0 over remaining training.
    
    Args:
        initial_lr: Target learning rate after warm-up
        warmup_steps: Number of steps for warm-up phase
        total_timesteps: Total training timesteps
    
    Returns:
        Schedule function that takes progress (1.0 -> 0.0) and returns LR
    """
    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        current_step = (1.0 - progress_remaining) * total_timesteps
        
        if current_step < warmup_steps:
            # Warm-up phase: linear increase from 0 to initial_lr
            warmup_ratio = current_step / warmup_steps
            return initial_lr * warmup_ratio
        else:
            # Post warm-up: linear decay from initial_lr to 0
            decay_progress = (current_step - warmup_steps) / (total_timesteps - warmup_steps)
            return initial_lr * (1.0 - decay_progress)
    
    return schedule


class ValueNetworkWarmupCallback(BaseCallback):
    """Callback to reduce value function learning rate during early training.
    
    Patch 6: Apply value-network warm-up to reduce early-episode value learning impact
    during noisy volatility estimation. This prevents early value loss spikes from
    destabilizing policy improvement.
    """
    
    def __init__(self, warmup_steps: int = 5000, initial_vf_coef: float = 0.1, 
                 target_vf_coef: float = 0.5, verbose: int = 0):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self.initial_vf_coef = initial_vf_coef
        self.target_vf_coef = target_vf_coef
    
    def _on_step(self) -> bool:
        if self.n_calls < self.warmup_steps:
            # Linear interpolation from initial to target vf_coef
            ratio = self.n_calls / self.warmup_steps
            current_vf_coef = self.initial_vf_coef + ratio * (self.target_vf_coef - self.initial_vf_coef)
            # Update vf_coef in the model (PPO stores this)
            if hasattr(self.model, 'vf_coef'):
                self.model.vf_coef = current_vf_coef
        return True


class GradientClipCallback(BaseCallback):
    """Callback to monitor and optionally clip gradients during training.
    
    Provides gradient norm logging for debugging training stability.
    """
    
    def __init__(self, max_grad_norm: float = 0.5, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.max_grad_norm = max_grad_norm
        self.log_freq = log_freq
        self.grad_norms = []
    
    def _on_step(self) -> bool:
        # Log gradient norm periodically (SB3 handles clipping internally)
        if self.n_calls % self.log_freq == 0:
            # Access policy parameters
            try:
                total_norm = 0.0
                for param in self.model.policy.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.grad_norms.append(total_norm)
                
                if self.verbose > 0 and len(self.grad_norms) % 10 == 0:
                    print(f"[GradClip] Step {self.n_calls}: grad_norm={total_norm:.4f}")
            except Exception:
                pass  # Gradient may not be available at all steps
        return True


def train(args):
    # set global random seed for reproducibility
    if args.seed is not None:
        set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        try:
            torch.manual_seed(args.seed)
        except Exception:
            pass

    # build a list of environment factories for vectorized envs
    # Patch 4: Each env gets a distinct seed offset (seed + rank) for different start positions
    # Patch 9: Walk-forward sampling via walk_forward_pct
    n_envs = max(1, int(args.n_envs))
    env_fns = [
        make_env_factory(
            args.price_csv,
            args.n_assets,
            args.transaction_cost,
            args.reward_alpha,
            rank=i,
            seed=args.seed,
            vol_warmup_steps=args.vol_warmup_steps,
            turnover_penalty=args.turnover_penalty,
            window=args.window,
            walk_forward_pct=args.walk_forward_pct,
            action_smoothing=args.action_smoothing,
            reward_scale=args.reward_scale,
        )
        for i in range(n_envs)
    ]
    
    # Use SubprocVecEnv for parallel execution if n_envs > 1 and subproc flag set
    if args.use_subproc and n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    tensorboard_log = args.tensorboard_log if args.tensorboard_log else None
    
    # Patch 7: Learning rate warm-up schedule
    if args.lr_warmup_steps > 0:
        lr_schedule = linear_warmup_schedule(
            initial_lr=args.learning_rate,
            warmup_steps=args.lr_warmup_steps,
            total_timesteps=args.total_timesteps,
        )
    else:
        lr_schedule = args.learning_rate
    
    if args.algo.upper() == "PPO":
        # Patch 8: PPO hyperparameter defaults optimized for finance
        # - Smaller batch size for noisy financial data
        # - More conservative clipping for stable updates
        # - Entropy coefficient for exploration
        # - GAE lambda for better advantage estimation
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=tensorboard_log,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    # checkpoint callback
    callbacks = []
    if args.checkpoint_freq and args.checkpoint_freq > 0:
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=ckpt_dir, name_prefix="rl_model")
        callbacks.append(ckpt_cb)
    
    # Patch 6: Value network warm-up callback
    if args.vf_warmup_steps > 0:
        vf_warmup_cb = ValueNetworkWarmupCallback(
            warmup_steps=args.vf_warmup_steps,
            initial_vf_coef=0.1,
            target_vf_coef=args.vf_coef,
            verbose=1 if args.log_gradients else 0,
        )
        callbacks.append(vf_warmup_cb)
    
    # Add gradient monitoring callback
    if args.log_gradients:
        grad_cb = GradientClipCallback(max_grad_norm=args.max_grad_norm, log_freq=1000, verbose=1)
        callbacks.append(grad_cb)

    cb = callbacks[0] if len(callbacks) == 1 else callbacks if callbacks else None
    model.learn(total_timesteps=args.total_timesteps, callback=cb)

    model.save(args.out)
    print(f"Model saved to {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Train RL agent for portfolio optimization")
    
    # Data and environment
    p.add_argument("--price-csv", help="CSV of price matrix (index dates, columns tickers)")
    p.add_argument("--n-assets", type=int, required=True, help="Number of assets")
    p.add_argument("--window", type=int, default=20, help="Lookback window for features")
    
    # Algorithm selection
    p.add_argument("--algo", default="PPO", help="RL algorithm to use (PPO)")
    p.add_argument("--total-timesteps", type=int, default=10000)
    p.add_argument("--out", default="models/ppo_model", help="Output path for model (without .zip)")
    
    # Patch 8: PPO hyperparameter defaults optimized for finance
    # These defaults are tuned for noisy financial time series
    p.add_argument("--learning-rate", type=float, default=1e-4, 
                   help="Learning rate (default: 1e-4, conservative for finance)")
    p.add_argument("--n-steps", type=int, default=256,
                   help="Steps per update (default: 256, smaller for financial data)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Minibatch size (default: 64)")
    p.add_argument("--n-epochs", type=int, default=5,
                   help="Number of epochs per update (default: 5)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
    p.add_argument("--gae-lambda", type=float, default=0.95,
                   help="GAE lambda for advantage estimation (default: 0.95)")
    p.add_argument("--clip-range", type=float, default=0.1,
                   help="PPO clip range (default: 0.1, conservative for stability)")
    p.add_argument("--ent-coef", type=float, default=0.02,
                   help="Entropy coefficient for exploration (default: 0.02)")
    p.add_argument("--vf-coef", type=float, default=0.5,
                   help="Value function coefficient (default: 0.5)")
    p.add_argument("--max-grad-norm", type=float, default=0.5,
                   help="Max gradient norm for clipping (default: 0.5)")
    
    # Learning rate warm-up
    p.add_argument("--lr-warmup-steps", type=int, default=0,
                   help="LR warm-up steps (0=disabled)")
    p.add_argument("--vf-warmup-steps", type=int, default=0,
                   help="Value function warm-up steps (Patch 6, 0=disabled)")
    p.add_argument("--log-gradients", action="store_true",
                   help="Log gradient norms during training")
    
    # Reward shaping (Patches 4, 5, 6)
    p.add_argument("--transaction-cost", type=float, default=0.0)
    p.add_argument("--reward-alpha", type=float, default=0.0, help="Risk aversion coefficient")
    p.add_argument("--vol-warmup-steps", type=int, default=20, 
                   help="Steps for volatility warm-up")
    p.add_argument("--turnover-penalty", type=float, default=0.0, 
                   help="Turnover penalty coefficient")
    p.add_argument("--action-smoothing", type=float, default=0.0,
                   help="Action EMA smoothing coefficient (0=disabled, 0.3-0.5 recommended)")
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="Soft reward scaling factor (1.0=disabled)")
    
    # Patch 9: Walk-forward sampling
    p.add_argument("--walk-forward-pct", type=float, default=0.0,
                   help="Fraction of data for walk-forward windows (0=disabled, Patch 9)")
    
    # Training infrastructure
    p.add_argument("--checkpoint-freq", type=int, default=0, 
                   help="Save checkpoint every N steps (0=disabled)")
    p.add_argument("--tensorboard-log", type=str, default="", 
                   help="TensorBoard log directory")
    p.add_argument("--seed", type=int, default=None, 
                   help="Global seed for reproducibility")
    p.add_argument("--n-envs", type=int, default=1, 
                   help="Number of parallel envs (vectorized)")
    p.add_argument("--use-subproc", action="store_true",
                   help="Use SubprocVecEnv for true parallelism")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
