import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Optional


class PortfolioEnv(gym.Env):
    """Minimal Gym-compatible portfolio environment skeleton.

    Observation: flattened per-asset features + global features (shape = N * F + G)
    Action: continuous non-negative vector of length N mapped to allocation weights summing to 1.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_assets: int, n_features_per_asset: int = 5, n_global_features: int = 3,
                 transaction_cost: float = 0.0, price_df: Optional[pd.DataFrame] = None, reward_alpha: float = 0.0):
        super().__init__()
        assert n_assets > 0
        self.n_assets = n_assets
        self.n_features_per_asset = n_features_per_asset
        self.n_global_features = n_global_features
        self.transaction_cost = float(transaction_cost)

        obs_dim = n_assets * n_features_per_asset + n_global_features
        # observation can be any finite real number
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # raw action space: non-negative numbers which will be normalized to sum to 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32)

        # internal state placeholders
        self.current_step = 0
        self.state = np.zeros(obs_dim, dtype=np.float32)
        self.current_weights = np.zeros(n_assets, dtype=np.float32)
        # price series (dates x tickers) used to compute returns; columns order must match assets
        self.price_df = None if price_df is None else price_df.copy()
        self.reward_alpha = float(reward_alpha)
        if self.price_df is not None:
            # ensure columns length matches n_assets
            if len(self.price_df.columns) != self.n_assets:
                raise ValueError("price_df columns must match n_assets length")

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        # initial equal weights
        w = np.ones(self.n_assets, dtype=np.float32)
        w /= w.sum()
        self.current_weights = w
        self.state = np.zeros_like(self.state)
        # reset step pointer for price series
        self.current_step = 0
        if return_info:
            return self.state, {"weights": self.current_weights.copy()}
        return self.state

    def step(self, action):
        # clip and normalize action to yield allocation weights
        raw = np.array(action, dtype=np.float32)
        raw = np.clip(raw, 0.0, None)
        weights = self._normalize_action(raw)

        # compute turnover and transaction cost (simple linear cost on weight change)
        turnover = float(np.abs(weights - self.current_weights).sum())
        trans_cost = float(self.transaction_cost) * turnover

        portfolio_return = 0.0
        done = False
        # if price series available, compute next-step portfolio return using new weights
        if self.price_df is not None:
            if self.current_step + 1 >= len(self.price_df):
                # no future price to compute return
                portfolio_return = 0.0
                done = True
            else:
                p_t = self.price_df.iloc[self.current_step].astype(float).values
                p_tp1 = self.price_df.iloc[self.current_step + 1].astype(float).values
                # avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    asset_rets = (p_tp1 / p_t) - 1.0
                    asset_rets = np.nan_to_num(asset_rets, nan=0.0, posinf=0.0, neginf=0.0)
                portfolio_return = float(np.dot(weights, asset_rets))
                self.current_step += 1

        # reward: realized portfolio_return minus transaction costs and optional penalty
        reward = portfolio_return - trans_cost - self.reward_alpha * abs(portfolio_return)

        self.current_weights = weights
        info = {"weights": weights, "turnover": turnover, "transaction_cost": trans_cost, "portfolio_return": portfolio_return}
        return self.state, float(reward), bool(done), info

    def _normalize_action(self, raw: np.ndarray) -> np.ndarray:
        s = float(raw.sum())
        if s <= 0:
            # fallback to uniform allocation
            w = np.ones(self.n_assets, dtype=np.float32) / float(self.n_assets)
        else:
            w = raw / s
        return w.astype(np.float32)

    def render(self, mode="human"):
        print(f"Step {self.current_step}: weights={self.current_weights}")

    def close(self):
        return None
