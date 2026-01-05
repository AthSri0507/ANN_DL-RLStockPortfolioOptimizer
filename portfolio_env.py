import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Optional


class RunningNormalizer:
    """Online mean/std normalizer (Welford)."""
    def __init__(self, size, eps=1e-8):
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = eps

    def update(self, x):
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2

    def normalize(self, x):
        std = np.sqrt(self.var / self.count)
        return (x - self.mean) / (std + 1e-8)


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
        transaction_cost: float = 0.001,
        reward_alpha: float = 0.0,
        window: int = 20
    ):
        super().__init__()
        self.n_assets = n_assets
        self.price_df = price_df.copy()
        self.transaction_cost = transaction_cost
        self.reward_alpha = reward_alpha
        self.window = window

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

        if self.price_df.shape[1] != self.n_assets:
            raise ValueError("price_df columns must match n_assets")

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.window
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.returns_history = []
        self._build_state()

        return self.state, {}

    def step(self, action):
        raw = np.clip(action, 0.0, None)
        weights = self._normalize_action(raw)

        turnover = np.sum(np.abs(weights - self.current_weights))
        trans_cost = self.transaction_cost * turnover

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

        # Risk-adjusted reward (mean-variance per-step)
        # reward_alpha acts as risk aversion (lambda)
        reward = portfolio_return - self.reward_alpha * (vol ** 2) - trans_cost

        self.current_weights = weights
        self.current_step += 1

        terminated = self.current_step >= len(self.price_df) - 2
        truncated = False

        if not terminated:
            self._build_state()

        info = {
            "portfolio_return": portfolio_return,
            "volatility": vol,
            "weights": weights
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
        self.normalizer.update(raw_state)
        self.state = self.normalizer.normalize(raw_state).astype(np.float32)

    def _normalize_action(self, raw):
        s = raw.sum()
        if s <= 0:
            return np.ones(self.n_assets) / self.n_assets
        return raw / s
