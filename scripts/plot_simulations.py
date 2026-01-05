"""Plot simulated GBM paths and percentile (fan) charts.

Reads simulation outputs produced by `scripts/simulate_gbm.py`.
If percentile CSVs (`<prefix>_pct5.csv`, `<prefix>_pct50.csv`, `<prefix>_pct95.csv`) exist, uses them
to draw fan charts. Otherwise, it will read individual path CSVs (`<prefix>_path_*.csv`) and compute percentiles.

Produces per-ticker fan chart PNGs and a multi-ticker sample paths PNG in the output folder.

Usage:
  python scripts/plot_simulations.py --sim-dir simulations --prefix sim --out-dir simulations/plots --n-sample 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_percentiles(sim_dir: Path, prefix: str):
    pct5 = sim_dir / f"{prefix}_pct5.csv"
    pct50 = sim_dir / f"{prefix}_pct50.csv"
    pct95 = sim_dir / f"{prefix}_pct95.csv"
    if pct5.exists() and pct50.exists() and pct95.exists():
        p5 = pd.read_csv(pct5, index_col=0, parse_dates=False)
        p50 = pd.read_csv(pct50, index_col=0, parse_dates=False)
        p95 = pd.read_csv(pct95, index_col=0, parse_dates=False)
        return p5, p50, p95

    # fallback: collect individual path CSVs and compute percentiles
    path_files = sorted(sim_dir.glob(f"{prefix}_path_*.csv"))
    if len(path_files) == 0:
        raise FileNotFoundError("No percentile files or path files found in sim_dir")

    # load all paths into array shape (n_steps+1, n_assets, n_paths)
    sample_df = pd.read_csv(path_files[0], index_col=0)
    steps = sample_df.shape[0]
    tickers = list(sample_df.columns)
    n_assets = len(tickers)
    n_paths = len(path_files)
    arr = np.empty((steps, n_assets, n_paths), dtype=float)
    for i, f in enumerate(path_files):
        df = pd.read_csv(f, index_col=0)
        arr[:, :, i] = df.values

    p5 = pd.DataFrame(np.percentile(arr, 5, axis=2), columns=tickers)
    p50 = pd.DataFrame(np.percentile(arr, 50, axis=2), columns=tickers)
    p95 = pd.DataFrame(np.percentile(arr, 95, axis=2), columns=tickers)
    return p5, p50, p95


def plot_fan(p5: pd.DataFrame, p50: pd.DataFrame, p95: pd.DataFrame, out_dir: Path, prefix: str):
    tickers = p50.columns
    out_dir.mkdir(parents=True, exist_ok=True)
    for tk in tickers:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(p50.index))
        ax.plot(x, p50[tk].values, color="C0", label="median")
        ax.fill_between(x, p5[tk].values, p95[tk].values, color="C0", alpha=0.2, label="5-95 pct")
        ax.set_title(f"Fan chart - {tk}")
        ax.set_xlabel("step")
        ax.set_ylabel("price")
        ax.legend()
        fname = out_dir / f"{prefix}_{tk}_fan.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)


def plot_sample_paths(sim_dir: Path, prefix: str, out_dir: Path, n_sample: int = 8):
    path_files = sorted(sim_dir.glob(f"{prefix}_path_*.csv"))
    if len(path_files) == 0:
        raise FileNotFoundError("No path files found to plot sample paths")
    sample_files = path_files[:n_sample]
    sample_dfs = [pd.read_csv(f, index_col=0) for f in sample_files]
    tickers = sample_dfs[0].columns
    steps = sample_dfs[0].shape[0]

    # multi-panel: one subplot per ticker
    n_cols = min(4, len(tickers))
    n_rows = int(np.ceil(len(tickers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()
    for i, tk in enumerate(tickers):
        ax = axes[i]
        for df in sample_dfs:
            ax.plot(df[tk].values, alpha=0.7)
        ax.set_title(f"Sample paths - {tk}")
        ax.set_xlabel("step")
        ax.set_ylabel("price")
    # hide unused axes
    for j in range(len(tickers), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fname = out_dir / f"{prefix}_sample_paths.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-dir", default="simulations", help="Directory with simulation CSV outputs")
    p.add_argument("--prefix", default="sim", help="Prefix used when saving simulation files")
    p.add_argument("--out-dir", default="simulations/plots", help="Directory to save plots")
    p.add_argument("--n-sample", type=int, default=8, help="Number of sample paths to plot")
    args = p.parse_args()

    sim_dir = Path(args.sim_dir)
    out_dir = Path(args.out_dir)
    p5, p50, p95 = load_percentiles(sim_dir, args.prefix)
    plot_fan(p5, p50, p95, out_dir, args.prefix)
    try:
        plot_sample_paths(sim_dir, args.prefix, out_dir, n_sample=args.n_sample)
    except FileNotFoundError:
        # if paths missing, skip sample paths
        pass

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
