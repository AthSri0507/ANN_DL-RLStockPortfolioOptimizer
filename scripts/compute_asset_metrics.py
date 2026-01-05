"""Compute asset-level metrics for a price matrix.

Outputs a CSV with per-asset metrics:
 - mean_daily_return, annual_return
 - daily_vol, annual_vol
 - contribution_return (weight * annual_return)
 - contribution_vol (marginal vol contribution)
 - corr_with_portfolio
 - momentum_{window} (mean simple returns over `window`)

Usage:
  python scripts/compute_asset_metrics.py --price-csv data/price_matrix.csv --out-dir simulations/metrics --weights-csv my_weights.csv

If `--weights-csv` is omitted, equal weights are assumed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compute_asset_metrics(price_df: pd.DataFrame, weights: pd.Series | None = None, periods_per_year: int = 252, window: int = 20):
    prices = price_df.astype(float).copy()
    # simple returns
    rets = prices.pct_change().dropna()

    if weights is None:
        weights = pd.Series(1.0 / prices.shape[1], index=prices.columns)
    else:
        # ensure Series and normalized
        weights = pd.Series(weights).reindex(prices.columns).fillna(0.0)
        s = weights.sum()
        if s == 0:
            weights = pd.Series(1.0 / prices.shape[1], index=prices.columns)
        else:
            weights = weights / s

    mean_daily = rets.mean()
    daily_vol = rets.std(ddof=0)

    annual_return = mean_daily * periods_per_year
    annual_vol = daily_vol * np.sqrt(periods_per_year)

    # contribution to return (simple): weight * annual_return
    contribution_return = weights * annual_return

    # annualize covariance and compute variance/vol contributions
    cov_daily = rets.cov(ddof=0)
    cov_annual = cov_daily * periods_per_year
    w = weights.values.reshape(-1, 1)
    port_var_annual = float((w.T @ cov_annual.values @ w).squeeze())
    port_vol_annual = np.sqrt(port_var_annual) if port_var_annual > 0 else 0.0

    # marginal contribution to portfolio variance (annual)
    marginal_annual = (cov_annual.values @ w).squeeze()
    # contribution to variance (additive)
    contribution_variance = weights.values * marginal_annual
    # percent contribution to portfolio variance
    pct_contribution_variance = contribution_variance / port_var_annual if port_var_annual > 0 else np.zeros_like(contribution_variance)
    # contribution to portfolio vol (approx): w * marginal / port_vol
    if port_vol_annual > 0:
        contribution_vol = (weights.values * marginal_annual) / port_vol_annual
    else:
        contribution_vol = np.zeros_like(marginal_annual)

    # correlation with portfolio returns (using daily returns)
    port_rets = (rets * weights).sum(axis=1)
    corr_with_portfolio = rets.corrwith(port_rets)

    # recent momentum: use log-return over `window` (robust to single-day spikes)
    prices = price_df.astype(float)
    if prices.shape[0] > window:
        price_now = prices.iloc[-1]
        price_prev = prices.shift(window).iloc[-1]
        # log momentum
        momentum = np.log(price_now / price_prev)
    else:
        momentum = pd.Series(0.0, index=prices.columns)

    # max drawdown per asset
    cum = (1.0 + rets).cumprod()
    running_max = cum.cummax()
    drawdowns = (running_max - cum) / running_max
    max_drawdown = drawdowns.max()

    df = pd.DataFrame(index=prices.columns)
    df["mean_daily_return"] = mean_daily
    df["annual_return"] = annual_return
    df["daily_vol"] = daily_vol
    df["annual_vol"] = annual_vol
    df["weight"] = weights
    df["contribution_return"] = contribution_return
    df["contribution_vol"] = contribution_vol
    df["contribution_variance"] = contribution_variance
    df["pct_contribution_variance"] = pct_contribution_variance
    df["max_drawdown"] = max_drawdown
    df["corr_with_portfolio"] = corr_with_portfolio
    df[f"momentum_{window}"] = momentum

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--price-csv", required=True, help="Wide CSV with index=Date and columns=tickers")
    p.add_argument("--out-dir", default="simulations/metrics", help="Directory to save metrics CSV")
    p.add_argument("--weights-csv", default=None, help="Optional CSV with columns ['ticker','weight'] or headerless two-column file")
    p.add_argument("--periods-per-year", type=int, default=252)
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.price_csv, index_col=0, parse_dates=True)

    weights = None
    if args.weights_csv is not None:
        wdf = pd.read_csv(args.weights_csv)
        if list(wdf.columns[:2]) == ["ticker", "weight"] or list(wdf.columns[:2]) == ["Ticker", "Weight"]:
            weights = pd.Series(wdf.iloc[:, 1].values, index=wdf.iloc[:, 0].values)
        else:
            # try headerless two-column
            weights = pd.Series(wdf.iloc[:, 1].values, index=wdf.iloc[:, 0].values)

    metrics = compute_asset_metrics(df, weights=weights, periods_per_year=args.periods_per_year, window=args.window)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "asset_metrics.csv"
    metrics.to_csv(out_path)

    print(f"Wrote asset metrics to {out_path}")


if __name__ == "__main__":
    main()
