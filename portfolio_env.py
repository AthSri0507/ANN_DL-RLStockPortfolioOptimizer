import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Optional


class RunningNormalizer:
    """Online mean/std normalizer using Welford's algorithm.
    
    Patch 1: Fixed variance computation to use proper Welford formula.
    The variance is computed as M2 / (count - 1) for unbiased estimation,
    with safeguards to prevent division by zero and variance shrinkage.
    """
    def __init__(self, size, eps=1e-8):
        self.size = size
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset normalizer statistics to initial state."""
        self.mean = np.zeros(self.size, dtype=np.float64)  # Use float64 for numerical stability
        self.M2 = np.zeros(self.size, dtype=np.float64)    # Sum of squared differences from mean
        self.count = 0  # Integer count, not eps-initialized

    def update(self, x):
        """Update running statistics using Welford's online algorithm."""
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2  # Welford's M2 accumulator

    def normalize(self, x):
        """Normalize observation using running mean and std."""
        x = np.asarray(x, dtype=np.float64)
        if self.count < 2:
            # Not enough samples for variance estimation, return centered only
            return ((x - self.mean) / 1.0).astype(np.float32)
        
        # Unbiased variance: M2 / (n - 1), with floor to prevent tiny std
        var = self.M2 / (self.count - 1)
        std = np.sqrt(np.maximum(var, self.eps))
        return ((x - self.mean) / (std + self.eps)).astype(np.float32)


class PortfolioEnv(gym.Env):
    """
    RL Portfolio Environment with:
    - Risk-adjusted reward
    - Feature normalization
    - Long-only, fully-invested portfolio
    """

    def __init__(
        self,
        n_assets: int,
        price_df: Optional[pd.DataFrame],
        transaction_cost: float = 0.0005,
        reward_alpha: float = 0.0,
        window: int = 20,
        randomize_start: bool = True,
        start_step: Optional[int] = None,
        vol_warmup_steps: int = 20,
        turnover_penalty: float = 0.02,
        action_smoothing: float = 0.3,
        reward_scale: float = 1.0,
        cvar_beta: float = 0.0,
    ):
    
        super().__init__()
        self.n_assets = n_assets
        self.price_df = price_df.copy()
        self.transaction_cost = transaction_cost
        self.reward_alpha = reward_alpha
        self.window = window
        self.randomize_start = randomize_start
        self.start_step = start_step  # Fixed start step (overrides randomize_start if set)
        
        # Patch 5: Soft volatility warm-up to stabilize early-step rewards
        # During first vol_warmup_steps, use fallback vol to avoid exploding/vanishing rewards
        self.vol_warmup_steps = vol_warmup_steps
        self.fallback_vol = 0.02  # ~2% daily vol as baseline
        
        # Patch 6: Explicit turnover penalty (separate from transaction cost)
        # Penalizes large weight changes to reduce noisy reallocations
        self.turnover_penalty = turnover_penalty
        
        # Patch 4: Action smoothing (EMA) to reduce high-frequency weight noise
        # alpha=0 means no smoothing, alpha=1 means fully use previous weights
        # Recommended: 0.3-0.5 for reducing noise while keeping exploration
        self.action_smoothing = action_smoothing
        self._smoothed_weights = None  # Initialized on reset
        
        # Patch 5: Reward scaling to keep rewards in consistent magnitude
        # Soft scaling (not clipping) to preserve learning signal
        self.reward_scale = reward_scale

        # CVaR-aware downside risk penalty
        # Penalizes left-tail risk to align training with evaluation metrics (VaR/CVaR)
        # cvar_beta=0.0 means no CVaR penalty (default, preserves existing behavior)
        self.cvar_beta = cvar_beta

        # Per-asset features:
        # [1d ret, 5d ret, 20d ret, 20d vol, 20d momentum]
        self.n_features_per_asset = 5
        self.n_global_features = 3
        self.obs_dim = n_assets * self.n_features_per_asset + self.n_global_features

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32
        )

        self.current_step = 0
        self.current_weights = np.ones(n_assets) / n_assets
        self.state = np.zeros(self.obs_dim, dtype=np.float32)

        self.returns_history = []
        self.normalizer = RunningNormalizer(self.obs_dim)
        
        # Store random state for reproducibility
        self._np_random = None

        if self.price_df.shape[1] != self.n_assets:
            raise ValueError("price_df columns must match n_assets")

    def reset(self, seed=None, options=None):
        # Handle seeding for reproducibility
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        # Patch 1: Reset normalizer on reset() to prevent historic stats leakage
        self.normalizer.reset()

        # Patch 3: Randomize episode start index
        # Valid range: [window, len(price_df) - 2] to ensure we can step at least once
        min_start = self.window
        max_start = len(self.price_df) - 2
        
        if self.start_step is not None:
            # Use fixed start step if provided (for deterministic testing/evaluation)
            self.current_step = max(min_start, min(self.start_step, max_start))
        elif self.randomize_start and max_start > min_start:
            # Randomize start within valid range
            self.current_step = self._np_random.integers(min_start, max_start + 1)
        else:
            # Default to start at window
            self.current_step = min_start

        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self._smoothed_weights = self.current_weights.copy()  # Patch 4: Init smoothed weights
        self.returns_history = []
        self._build_state()

        return self.state, {}

    def step(self, action):
        raw = np.clip(action, 0.0, None)
        target_weights = self._normalize_action(raw)
        
        # Patch 4: Action smoothing (EMA) to reduce high-frequency weight noise
        # This reduces turnover from noisy policy outputs while preserving PPO exploration
        if self.action_smoothing > 0 and self._smoothed_weights is not None:
            # EMA: smoothed = alpha * previous + (1 - alpha) * target
            weights = self.action_smoothing * self._smoothed_weights + (1 - self.action_smoothing) * target_weights
            weights = self._normalize_action(weights)  # Re-normalize to ensure sum=1
        else:
            weights = target_weights
        
        self._smoothed_weights = weights.copy()  # Update smoothed weights for next step

        turnover = np.sum(np.abs(weights - self.current_weights))
        trans_cost = self.transaction_cost * turnover
        
        # Patch 6: Explicit turnover penalty (quadratic) to reduce noisy reallocations
        # This is separate from transaction cost and penalizes large weight changes
        turnover_pen = self.turnover_penalty * (turnover ** 2)

        p_t = self.price_df.iloc[self.current_step].values
        p_tp1 = self.price_df.iloc[self.current_step + 1].values

        asset_returns = (p_tp1 / p_t) - 1.0
        asset_returns = np.nan_to_num(asset_returns, nan=0.0)

        portfolio_return = np.dot(weights, asset_returns)
        self.returns_history.append(portfolio_return)

        # Rolling volatility
        if len(self.returns_history) >= self.window:
            vol = np.std(self.returns_history[-self.window:])
        else:
            vol = np.std(self.returns_history) if len(self.returns_history) > 1 else 1e-4

        # Patch 5: Soft volatility warm-up to stabilize early-step rewards
        # Blend between fallback_vol and actual vol during warm-up period
        episode_step = len(self.returns_history)
        if episode_step < self.vol_warmup_steps:
            # Linear interpolation: starts at 100% fallback, ends at 100% actual
            warmup_ratio = episode_step / self.vol_warmup_steps
            effective_vol = (1 - warmup_ratio) * self.fallback_vol + warmup_ratio * vol
        else:
            effective_vol = vol

        # CVaR downside risk penalty (quantile-based, standard finance formulation)
        # CVaR = E[return | return <= VaR_alpha], aligns with evaluation metrics
        cvar_penalty = 0.0
        if self.cvar_beta > 0 and len(self.returns_history) >= self.window:
            recent_returns = np.array(self.returns_history[-self.window:])
            alpha = 0.05  # 95% CVaR (worst 5% tail)
            q = np.quantile(recent_returns, alpha)
            tail = recent_returns[recent_returns <= q]
            if len(tail) > 0:
                cvar_95 = np.mean(tail)
                # cvar_95 is typically negative; clip to avoid rewarding in low-vol regimes
                cvar_penalty = self.cvar_beta * max(0.0, -cvar_95)

        # Risk-adjusted reward (mean-variance per-step)
        # reward_alpha acts as risk aversion (lambda)
        raw_reward = portfolio_return - self.reward_alpha * (effective_vol ** 2) - trans_cost - turnover_pen - cvar_penalty
        
        # Patch 5: Soft reward scaling to keep magnitude consistent
        # Uses tanh-based scaling to bound extreme rewards without hard clipping
        # This preserves the learning signal while preventing value function instability
        if self.reward_scale != 1.0:
            # Soft scaling: scale * tanh(raw / scale) preserves sign and relative ordering
            reward = self.reward_scale * np.tanh(raw_reward / self.reward_scale)
        else:
            reward = raw_reward

        self.current_weights = weights
        self.current_step += 1

        terminated = self.current_step >= len(self.price_df) - 2
        truncated = False

        if not terminated:
            self._build_state()

        info = {
            "portfolio_return": portfolio_return,
            "volatility": vol,
            "effective_volatility": effective_vol,
            "turnover": turnover,
            "turnover_penalty": turnover_pen,
            "cvar_penalty": cvar_penalty,
            "raw_reward": raw_reward,
            "scaled_reward": reward,
            "target_weights": target_weights,
            "smoothed_weights": weights,
        }

        return self.state, float(reward), terminated, truncated, info

    def _build_state(self):
        t = self.current_step
        prices = self.price_df.values
        n = prices.shape[0]

        # helper to avoid negative indices (use earliest available price)
        def at_lag(k):
            idx = max(0, t - k)
            return prices[idx]

        p_t = prices[t]
        p_1 = at_lag(1)
        p_5 = at_lag(5)
        p_20 = at_lag(20)

        ret1 = (p_t / p_1) - 1.0
        ret5 = (p_t / p_5) - 1.0
        ret20 = (p_t / p_20) - 1.0

        start = max(0, t - self.window)
        window_prices = prices[start:t + 1]
        if window_prices.shape[0] >= 2:
            rets = (window_prices[1:] / window_prices[:-1]) - 1.0
        else:
            rets = np.zeros((1, self.n_assets))

        vol20 = np.nan_to_num(np.std(rets, axis=0))
        mom20 = np.nan_to_num(np.mean(rets, axis=0))

        per_asset = np.stack([ret1, ret5, ret20, vol20, mom20], axis=1).ravel()

        global_feats = np.array([
            np.mean(ret1),
            np.mean(vol20),
            float(t) / float(n)
        ])

        raw_state = np.concatenate([per_asset, global_feats])
        
        # Patch 2: Normalize observations THEN update running stats
        # This ensures agent never sees future-informed normalization
        self.state = self.normalizer.normalize(raw_state).astype(np.float32)
        self.normalizer.update(raw_state)

    def _normalize_action(self, raw):
        s = raw.sum()
        if s <= 0:
            return np.ones(self.n_assets) / self.n_assets
        return raw / s
