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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Gymnasium-compatible reset signature: returns (obs, info)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        # initial equal weights
        w = np.ones(self.n_assets, dtype=np.float32)
        w /= w.sum()
        self.current_weights = w
        # initialize state based on price history (if available)
        self.state = np.zeros_like(self.state)
        self._build_state()
        # reset step pointer for price series
        self.current_step = 0
        return self.state, {"weights": self.current_weights.copy()}

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
        # advance observation/state to reflect new time index
        # Gymnasium step signature: (obs, reward, terminated, truncated, info)
        if not done:
            # current_step already incremented when computing returns
            self._build_state()
        terminated = bool(done)
        truncated = False
        return self.state, float(reward), terminated, truncated, info

    def _build_state(self):
        """Populate `self.state` with per-asset and global features based on `self.current_step`.

        Per-asset features (N x F): default F=5
          - 1d return, 5d return, 20d return, 20d vol (std of daily rets), 20d momentum (mean)
        Global features (G): mean 1d return, mean 20d vol, normalized step index
        """
        if self.price_df is None or len(self.price_df) == 0:
            self.state[:] = 0.0
            return

        t = int(self.current_step)
        prices = self.price_df.values.astype(float)
        n_dates, n_assets = prices.shape

        # helper to get price at index safely
        def price_at(idx):
            if idx < 0:
                return np.full(n_assets, np.nan, dtype=float)
            if idx >= n_dates:
                return np.full(n_assets, np.nan, dtype=float)
            return prices[idx]

        p_t = price_at(t)
        p_t_1 = price_at(t - 1)
        p_t_5 = price_at(t - 5)
        p_t_20 = price_at(t - 20)

        # daily returns series up to t (for vol/momentum)
        # compute returns for window up to last 20 days
        start_idx = max(1, t - 20 + 1)
        rets_window = []
        for i in range(start_idx, t + 1):
            prev = price_at(i - 1)
            cur = price_at(i)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = (cur / prev) - 1.0
                r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            rets_window.append(r)
        if len(rets_window) == 0:
            rets_window = [np.zeros(n_assets, dtype=float)]
        rets_window = np.vstack(rets_window)

        # per-asset feature vector
        feat_list = []
        # 1d
        with np.errstate(divide='ignore', invalid='ignore'):
            one_d = (p_t / p_t_1) - 1.0
        one_d = np.nan_to_num(one_d, nan=0.0, posinf=0.0, neginf=0.0)
        feat_list.append(one_d)
        # 5d
        with np.errstate(divide='ignore', invalid='ignore'):
            ret5 = (p_t / p_t_5) - 1.0
        ret5 = np.nan_to_num(ret5, nan=0.0, posinf=0.0, neginf=0.0)
        feat_list.append(ret5)
        # 20d
        with np.errstate(divide='ignore', invalid='ignore'):
            ret20 = (p_t / p_t_20) - 1.0
        ret20 = np.nan_to_num(ret20, nan=0.0, posinf=0.0, neginf=0.0)
        feat_list.append(ret20)
        # vol 20d (std over rets_window)
        vol20 = np.std(rets_window, axis=0)
        feat_list.append(vol20)
        # momentum 20d (mean over window)
        mom20 = np.mean(rets_window, axis=0)
        feat_list.append(mom20)

        per_asset_feats = np.stack(feat_list, axis=1)  # shape (n_assets, F)

        # global features
        mean_1d = np.nan_to_num(np.mean(one_d), nan=0.0)
        mean_vol20 = np.nan_to_num(np.mean(vol20), nan=0.0)
        norm_step = float(t) / max(1, n_dates - 1)
        global_feats = np.array([mean_1d, mean_vol20, norm_step], dtype=float)

        # flatten into state vector: per-asset flattened first, then global
        flat = np.concatenate([per_asset_feats.ravel(), global_feats])
        # ensure expected size
        if flat.size != self.state.size:
            # if sizes mismatch due to config, resize or pad/truncate
            s = self.state.size
            if flat.size < s:
                tmp = np.zeros(s, dtype=float)
                tmp[:flat.size] = flat
                flat = tmp
            else:
                flat = flat[:s]

        self.state = flat.astype(np.float32)

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
