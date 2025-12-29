import math
from typing import Dict

import pandas as pd


def build_price_matrix(price_data: Dict[str, pd.DataFrame], price_col: str = "Adj Close", fill_method: str = "ffill") -> pd.DataFrame:
    """Align per-ticker DataFrames into a single price matrix (dates x tickers).

    - price_data: mapping ticker -> DataFrame (as returned by `yfinance`)
    - price_col: preferred price column name; falls back to 'Close'
    - fill_method: how to fill missing values: 'ffill', 'bfill', or 'none'
    """
    series = {}
    for ticker, df in price_data.items():
        if df is None or df.empty:
            continue
        # Prefer adjusted close when available (handles splits/dividends)
        if price_col in df.columns:
            s = df[price_col].copy()
        elif "Close" in df.columns:
            # If only Close is available, attempt simple split-adjustment
            s = _adjust_close_for_splits(df)
        else:
            # try first numeric column
            s = df.select_dtypes("number").iloc[:, 0].copy()
        s.name = ticker
        series[ticker] = s
    if not series:
        return pd.DataFrame()
    price_df = pd.concat(series.values(), axis=1, keys=series.keys()).sort_index()
    if fill_method == "ffill":
        price_df = price_df.ffill()
    elif fill_method == "bfill":
        price_df = price_df.bfill()
    return price_df


def compute_base_features(price_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute per-asset time-series features.

    Returns a dict with keys: `returns`, `ret_5d`, `ret_20d`, `vol_20d`, `momentum_20d`, `drawdown`.
    """
    out = {}
    if price_df.empty:
        return {"returns": pd.DataFrame()}

    # simple returns
    returns = price_df.pct_change()
    out["returns"] = returns

    # multi-day returns as cumulative product of (1+ret)-1
    out["ret_5d"] = (1 + returns).rolling(window=5, min_periods=1).apply(lambda x: x.add(1).prod() - 1)
    out["ret_20d"] = (1 + returns).rolling(window=20, min_periods=1).apply(lambda x: x.add(1).prod() - 1)

    # rolling volatility (sample std) over 20 days
    out["vol_20d"] = returns.rolling(window=20, min_periods=5).std()

    # momentum: simple 20-day momentum (price / price.shift(20) - 1)
    out["momentum_20d"] = price_df / price_df.shift(20) - 1

    # drawdown: (price / rolling_max) - 1
    rolling_max = price_df.cummax()
    out["drawdown"] = price_df / rolling_max - 1

    return out


def _adjust_close_for_splits(df: pd.DataFrame) -> pd.Series:
    """Attempt to adjust a `Close` series for discrete splits using observed large jumps.

    This is a heuristic fallback used when `Adj Close` is not available. It detects
    integer split ratios by looking for large negative returns and rescales earlier prices.
    """
    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index)
    # compute raw returns
    ret = close.pct_change()
    # identify potential split days where price drops by more than 40% (e.g., 2:1 split -> ~-50%)
    split_days = ret[ret < -0.4]
    if split_days.empty:
        return close

    # build cumulative adjustment factor (walk backwards applying split factors)
    adj = pd.Series(1.0, index=close.index)
    # iterate over split days and estimate integer split factor
    for day in split_days.index:
        # ratio of prior/after price approx split factor
        i = close.index.get_loc(day)
        if i == 0:
            continue
        prior = close.iloc[i - 1]
        after = close.iloc[i]
        if after == 0 or pd.isna(after):
            continue
        raw = prior / after
        # round to nearest integer (2,3,4..), but only accept reasonable values
        factor = float(round(raw))
        if factor >= 2 and abs(raw - factor) / factor < 0.2:
            # apply factor to all earlier dates (divide earlier prices by factor)
            # use integer location to affect strictly earlier positions
            adj.iloc[:i] = adj.iloc[:i] / factor

    adjusted = close * adj
    return adjusted


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional percentile ranks per row across columns.

    Returns a DataFrame with same shape with values in (0,1].
    """
    if df.empty:
        return df
    return df.rank(axis=1, pct=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="store_true")
    args = parser.parse_args()
    if args.example:
        print("features module ready")
