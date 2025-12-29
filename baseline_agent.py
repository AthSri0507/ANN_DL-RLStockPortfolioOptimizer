import numpy as np
from typing import Optional


class BaselineAgent:
    """Simple baseline agent providing a few heuristic policies.

    Policies:
    - `equal_weight`: uniform allocation across assets.
    - `random_uniform`: random non-negative vector normalized to sum to 1.
    - `momentum`: weights proportional to recent returns (simple heuristic).
    """

    def __init__(self, n_assets: int, seed: Optional[int] = None):
        self.n_assets = n_assets
        if seed is not None:
            np.random.seed(seed)

    def equal_weight(self) -> np.ndarray:
        w = np.ones(self.n_assets, dtype=np.float32) / float(self.n_assets)
        return w

    def random_uniform(self) -> np.ndarray:
        x = np.random.rand(self.n_assets).astype(np.float32)
        s = float(x.sum())
        if s <= 0:
            return self.equal_weight()
        return (x / s).astype(np.float32)

    def momentum(self, price_series_last: np.ndarray, price_series_prev: np.ndarray) -> np.ndarray:
        """Simple momentum: weight ~ max(0, last/prev - 1), normalized.

        - `price_series_last`, `price_series_prev` are 1D arrays of same length giving prices at t and t-1.
        """
        rets = (price_series_last / price_series_prev) - 1.0
        pos = np.clip(rets, 0.0, None)
        s = float(pos.sum())
        if s <= 0:
            return self.equal_weight()
        return (pos / s).astype(np.float32)


def example_usage():
    agent = BaselineAgent(3, seed=0)
    print("equal:", agent.equal_weight())
    print("random:", agent.random_uniform())


if __name__ == "__main__":
    example_usage()
