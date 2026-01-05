"""Simulate correlated GBM paths using Cholesky decomposition.

This script:
 - loads a wide price CSV (index dates, columns tickers)
 - computes log-returns, per-asset mean/std and empirical correlation
 - performs Cholesky on the correlation (with jitter fallback)
 - simulates `n_paths` correlated GBM trajectories for `n_steps`
 - writes simulated paths to CSVs in `--out-dir`

Usage:
  python scripts/simulate_gbm.py --price-csv data/price_matrix.csv --n-paths 30 --n-steps 252 --out-dir simulations

Options include `--periods-per-year` (for annualization display) and `--seed`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def estimate_stats_and_corr(price_df: pd.DataFrame):
    prices = price_df.dropna(how="all")
    # drop columns with all NaN
    prices = prices.loc[:, prices.isna().sum() < len(prices)]
    logrets = np.log(prices / prices.shift(1)).dropna()
    mean_r = logrets.mean(axis=0)
    std_r = logrets.std(axis=0, ddof=0)
    corr = logrets.corr().values
    return mean_r.values, std_r.values, corr, logrets


def chol_with_jitter(corr: np.ndarray, max_tries: int = 10, jitter_init: float = 1e-8):
    jitter = jitter_init
    for i in range(max_tries):
        try:
            L = np.linalg.cholesky(corr + np.eye(corr.shape[0]) * jitter)
            return L, jitter
        except np.linalg.LinAlgError:
            jitter *= 10
    raise np.linalg.LinAlgError("Cholesky failed even after jitter attempts")


def simulate_paths(
    S0: np.ndarray,
    mu_per_step: np.ndarray,
    sigma_per_step: np.ndarray,
    L: np.ndarray,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    noise: str = "gaussian",
    t_df: float = 5.0,
    logrets: pd.DataFrame | None = None,
):
    rng = np.random.default_rng(seed)
    n_assets = S0.shape[0]
    dt = 1.0  # using per-step log-return stats already

    # output array: shape (n_steps+1, n_assets, n_paths)
    out = np.empty((n_steps + 1, n_assets, n_paths), dtype=float)
    out[0, :, :] = np.expand_dims(S0, axis=1)

    for p in range(n_paths):
        S = S0.copy()
        for t in range(1, n_steps + 1):
            if noise == "gaussian":
                z = rng.standard_normal(size=n_assets)
                corr_z = L @ z
            elif noise == "t":
                # Student-t scaled to unit variance (variance = df/(df-2))
                if t_df <= 2:
                    raise ValueError("t_df must be > 2 for finite variance")
                z_t = rng.standard_t(t_df, size=n_assets)
                z = z_t * np.sqrt((t_df - 2.0) / t_df)
                corr_z = L @ z
            elif noise == "bootstrap":
                if logrets is None:
                    raise ValueError("logrets must be provided for bootstrap noise")
                # sample a historical log-return row and center it
                idx = rng.integers(0, logrets.shape[0])
                r = logrets.iloc[idx].values
                # center by mean to produce innovation with zero mean
                corr_z = (r - mu_per_step) / sigma_per_step
            else:
                raise ValueError(f"unknown noise type: {noise}")

            # GBM step using per-period log-return params
            # NOTE: use mu_per_step as mean(log-returns) directly (no extra -0.5*sigma^2)
            S = S * np.exp(mu_per_step * dt + sigma_per_step * corr_z * np.sqrt(dt))
            out[t, :, p] = S
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--price-csv", required=True, help="Wide CSV with index=Date and columns=tickers")
    p.add_argument("--n-paths", type=int, default=30)
    p.add_argument("--n-steps", type=int, default=252)
    p.add_argument("--out-dir", default="simulations")
    p.add_argument("--periods-per-year", type=int, default=252)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prefix", type=str, default="sim")
    p.add_argument("--noise-dist", type=str, default="gaussian", choices=["gaussian", "t", "bootstrap"], help="Innovation distribution: gaussian, t, or bootstrap")
    p.add_argument("--t-df", type=float, default=5.0, help="Degrees of freedom for Student-t innovations (df>2)")
    args = p.parse_args()

    df = pd.read_csv(args.price_csv, index_col=0, parse_dates=True)
    tickers = list(df.columns)
    if len(tickers) == 0:
        raise SystemExit("No tickers/columns found in price CSV")

    mean_r, std_r, corr, logrets = estimate_stats_and_corr(df)

    # per-step params (log-return mean/std)
    mu_step = mean_r  # per-step mean log-return
    sigma_step = std_r

    # ensure corr is PD for Cholesky
    L, jitter = chol_with_jitter(corr)

    S0 = df.iloc[-1].astype(float).values

    sim = simulate_paths(
        S0,
        mu_step,
        sigma_step,
        L,
        args.n_steps,
        args.n_paths,
        seed=args.seed,
        noise=args.noise_dist,
        t_df=args.t_df,
        logrets=logrets,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary params
    params = {
        "tickers": tickers,
        "periods_per_year": args.periods_per_year,
        "jitter_used": jitter,
        "noise_dist": args.noise_dist,
        "t_df": args.t_df,
        "n_paths": args.n_paths,
        "n_steps": args.n_steps,
        "mu_per_step": mean_r.tolist(),
        "sigma_per_step": std_r.tolist(),
    }
    (out_dir / "sim_params.json").write_text(json.dumps(params, indent=2))

    # Save each path as a CSV with index steps and columns tickers
    for p_idx in range(args.n_paths):
        arr = sim[:, :, p_idx]
        df_out = pd.DataFrame(arr, columns=tickers)
        df_out.index.name = "step"
        fname = out_dir / f"{args.prefix}_path_{p_idx:03d}.csv"
        df_out.to_csv(fname)

    # Save aggregated percentiles (5,50,95) per step per ticker
    pct5 = np.percentile(sim, 5, axis=2)
    pct50 = np.percentile(sim, 50, axis=2)
    pct95 = np.percentile(sim, 95, axis=2)

    def save_pct(mat, name):
        dfp = pd.DataFrame(mat, columns=tickers)
        dfp.index.name = "step"
        dfp.to_csv(out_dir / f"{args.prefix}_{name}.csv")

    save_pct(pct5, "pct5")
    save_pct(pct50, "pct50")
    save_pct(pct95, "pct95")

    print(f"Wrote {args.n_paths} simulated paths to {out_dir} (params in sim_params.json)")


if __name__ == "__main__":
    main()
