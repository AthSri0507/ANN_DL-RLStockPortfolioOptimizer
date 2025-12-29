from typing import Dict

import pandas as pd


def portfolio_time_series(price_df: pd.DataFrame, quantities: Dict[str, float]) -> pd.Series:
    """Compute portfolio value time-series given price matrix and per-ticker quantities.

    - `price_df`: DataFrame with columns for tickers and datetime index
    - `quantities`: mapping ticker -> number of shares
    """
    if price_df.empty:
        return pd.Series(dtype=float)
    missing = set(quantities.keys()) - set(price_df.columns)
    if missing:
        raise KeyError(f"Quantities provided for unknown tickers: {missing}")
    # align quantities to columns order
    qty = pd.Series(quantities)
    qty = qty.reindex(price_df.columns).fillna(0.0)
    # multiply and sum
    values = price_df.multiply(qty, axis=1).sum(axis=1)
    return values


def current_weights_from_quantities(price_df: pd.DataFrame, quantities: Dict[str, float], cash: float = 0.0) -> pd.Series:
    """Return current portfolio weights (including cash fraction) from latest prices and quantities.

    Returns a Series indexed by tickers with an extra 'CASH' entry if cash>0.
    """
    if price_df.empty:
        return pd.Series(dtype=float)
    last = price_df.iloc[-1]
    qty = {k: quantities.get(k, 0.0) for k in price_df.columns}
    mv = pd.Series({t: float(qty.get(t, 0.0)) * float(last[t]) for t in price_df.columns})
    total = mv.sum() + float(cash)
    if total == 0:
        return mv * 0.0
    weights = mv / total
    if cash and cash > 0:
        weights = weights.append(pd.Series({"CASH": float(cash) / total}))
    return weights


def market_index_signals(index_series: pd.Series) -> Dict[str, float]:
    """Compute simple market index signals: MA50, MA200, ma_ratio, and recent z-score.

    Returns a dict with numeric signals.
    """
    s = index_series.dropna().astype(float)
    if s.empty:
        return {"ma50": float("nan"), "ma200": float("nan"), "ma_ratio": float("nan"), "zscore": float("nan")}
    ma50 = s.rolling(window=50, min_periods=1).mean().iloc[-1]
    ma200 = s.rolling(window=200, min_periods=1).mean().iloc[-1]
    ma_ratio = ma50 / ma200 if ma200 != 0 else float("nan")
    # z-score over past 252 trading days (or available)
    window = min(len(s), 252)
    recent = s.iloc[-window:]
    z = (recent.iloc[-1] - recent.mean()) / recent.std() if recent.std() != 0 else 0.0
    return {"ma50": float(ma50), "ma200": float(ma200), "ma_ratio": float(ma_ratio), "zscore": float(z)}


if __name__ == "__main__":
    print("global_features module ready")
