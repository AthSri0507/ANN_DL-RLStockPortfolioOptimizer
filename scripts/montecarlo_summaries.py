"""Compute percentile, VaR, CVaR and terminal distribution summaries from simulations.

Reads simulated path CSVs produced by `scripts/simulate_gbm.py` (files named
`<prefix>_path_XXX.csv`) and produces summary CSV/JSON files with percentiles,
VaR and CVaR for terminal returns.

Usage:
  python scripts/montecarlo_summaries.py --sim-dir simulations --prefix sim --out-dir simulations/summaries --confidence 0.95
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_final_prices(sim_dir: Path, prefix: str):
    path_files = sorted(sim_dir.glob(f"{prefix}_path_*.csv"))
    if len(path_files) == 0:
        raise FileNotFoundError(f"No path files found in {sim_dir} with prefix {prefix}")

    dfs = [pd.read_csv(p, index_col=0) for p in path_files]
    tickers = list(dfs[0].columns)
    n_paths = len(dfs)
    # stack final rows
    finals = np.vstack([df.iloc[-1].values for df in dfs])  # shape (n_paths, n_assets)
    initials = dfs[0].iloc[0].values.astype(float)
    return initials, finals, tickers


def compute_summaries(initials: np.ndarray, finals: np.ndarray, percentiles=(1,5,25,50,75,95,99), confidence=0.95):
    # finals: shape (n_paths, n_assets)
    S0 = initials.astype(float)
    returns = finals / S0[np.newaxis, :] - 1.0  # shape (n_paths, n_assets)

    summaries = []
    alpha = confidence
    lower_pct = (1.0 - alpha) * 100.0

    for j in range(returns.shape[1]):
        r = returns[:, j]
        pct_vals = np.percentile(r, percentiles)
        mean_final = float(np.mean(finals[:, j]))
        std_final = float(np.std(finals[:, j], ddof=0))
        mean_ret = float(np.mean(r))
        std_ret = float(np.std(r, ddof=0))
        var = float(np.percentile(r, lower_pct))
        # CVaR: mean of returns <= VaR
        tail = r[r <= var]
        cvar = float(tail.mean()) if tail.size > 0 else float(var)

        row = {
            "S0": float(S0[j]),
            "mean_final": mean_final,
            "std_final": std_final,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "VaR": var,
            "CVaR": cvar,
        }
        for k, p in enumerate(percentiles):
            row[f"pct_{p}"] = float(pct_vals[k])
        summaries.append(row)

    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-dir", default="simulations", help="Directory containing simulated path CSVs")
    p.add_argument("--prefix", default="sim", help="Prefix used for simulated path files")
    p.add_argument("--out-dir", default="simulations/summaries", help="Directory to write summaries")
    p.add_argument("--percentiles", default="1,5,25,50,75,95,99", help="Comma-separated percentiles to compute")
    p.add_argument("--confidence", type=float, default=0.95, help="Confidence level for VaR/CVaR (e.g. 0.95)")
    args = p.parse_args()

    sim_dir = Path(args.sim_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    initial, finals, tickers = load_final_prices(sim_dir, args.prefix)
    pct_list = [int(x) for x in args.percentiles.split(",")]
    summaries = compute_summaries(initial, finals, percentiles=pct_list, confidence=args.confidence)

    # Save terminal distribution raw data
    finals_df = pd.DataFrame(finals, columns=tickers)
    finals_df.to_csv(out_dir / f"{args.prefix}_terminal_distribution.csv", index=False)

    # Save percentiles and VaR/CVaR summary table
    rows = []
    for tk, s in zip(tickers, summaries):
        r = {"ticker": tk}
        r.update(s)
        rows.append(r)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / f"{args.prefix}_summary.csv", index=False)

    # Save JSON summary
    json_out = {tk: rows[i] for i, tk in enumerate(tickers)}
    (out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(json_out, indent=2))

    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
