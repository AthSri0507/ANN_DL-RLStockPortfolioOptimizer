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
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from portfolio_env import PortfolioEnv


def make_env_factory(price_csv, n_assets, transaction_cost, reward_alpha, rank: int = 0, seed: int | None = None):
    df = None
    if price_csv:
        df = pd.read_csv(price_csv, index_col=0, parse_dates=True)

    def _init():
        # Each env gets its own seed offset for determinism when requested
        env = PortfolioEnv(n_assets=n_assets, price_df=df, transaction_cost=transaction_cost, reward_alpha=reward_alpha)
        if seed is not None:
            s = int(seed) + int(rank)
            # set global seeds for the env creation context
            np.random.seed(s)
            random.seed(s)
            try:
                torch.manual_seed(s)
            except Exception:
                pass
            # gym envs typically accept seeding through reset; keep consistent
        return env

    return _init


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
    n_envs = max(1, int(args.n_envs))
    env_fns = [make_env_factory(args.price_csv, args.n_assets, args.transaction_cost, args.reward_alpha, rank=i, seed=args.seed) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    tensorboard_log = args.tensorboard_log if args.tensorboard_log else None
    if args.algo.upper() == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=args.learning_rate, tensorboard_log=tensorboard_log)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    # checkpoint callback
    callbacks = []
    if args.checkpoint_freq and args.checkpoint_freq > 0:
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=ckpt_dir, name_prefix="rl_model")
        callbacks.append(ckpt_cb)

    cb = callbacks[0] if len(callbacks) == 1 else callbacks if callbacks else None
    model.learn(total_timesteps=args.total_timesteps, callback=cb)

    model.save(args.out)
    print(f"Model saved to {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--price-csv", help="CSV of price matrix (index dates, columns tickers)")
    p.add_argument("--n-assets", type=int, required=True, help="Number of assets")
    p.add_argument("--algo", default="PPO", help="RL algorithm to use (PPO)")
    p.add_argument("--total-timesteps", type=int, default=10000)
    p.add_argument("--out", default="models/ppo_model", help="Output path for model (without .zip)")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--transaction-cost", type=float, default=0.0)
    p.add_argument("--reward-alpha", type=float, default=0.0, help="Penalty multiplier on returns")
    p.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoint every N steps (0 = disabled)")
    p.add_argument("--tensorboard-log", type=str, default="", help="TensorBoard log directory")
    p.add_argument("--seed", type=int, default=None, help="Global seed for reproducibility")
    p.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs (vectorized) to create")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
