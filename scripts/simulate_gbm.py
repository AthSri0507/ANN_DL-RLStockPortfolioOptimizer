"""Simulate correlated GBM paths using Cholesky decomposition.

This script:
 - loads a wide price CSV (index dates, columns tickers)
 - computes log-returns, per-asset mean/std and empirical correlation
 - performs Cholesky on the correlation (with jitter fallback)
 - simulates `n_paths` correlated GBM trajectories for `n_steps`
 - writes simulated paths to CSVs in `--out-dir`

Realism Enhancements (v2):
 - Regime-conditioned MC: volatility regimes (low/med/high) with Markov switching
 - Drift shrinkage: reduces noisy sample mean drift toward 0 or risk-free baseline
 - Correlation stress testing: optional blend with high-correlation matrix for crisis scenarios
 - Block bootstrap: preserves temporal dependence in bootstrap noise

Limitations:
 - No jump diffusion or leverage effects
 - Regimes are heuristic (rolling vol quantiles), not full HMM/GARCH
 - Correlations are static within regime (no DCC)

Usage:
  python scripts/simulate_gbm.py --price-csv data/price_matrix.csv --n-paths 30 --n-steps 252 --out-dir simulations

Options include `--periods-per-year`, `--seed`, `--regime-switching`, `--drift-shrinkage`, `--stress-corr`.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# REGIME DETECTION AND PARAMETERS
# =============================================================================

@dataclass
class RegimeParams:
    """Parameters for a single volatility regime."""
    name: str
    mu: np.ndarray        # per-asset drift
    sigma: np.ndarray     # per-asset volatility
    corr: np.ndarray      # correlation matrix
    L: np.ndarray         # Cholesky of correlation
    logrets: pd.DataFrame # log-returns in this regime (for bootstrap)
    

def detect_regimes(
    logrets: pd.DataFrame,
    n_regimes: int = 3,
    vol_window: int = 20,
) -> Tuple[pd.Series, List[int]]:
    """Detect volatility regimes using rolling volatility quantiles.
    
    Returns:
        regime_labels: Series with regime index (0=low, 1=med, 2=high vol)
        regime_indices: List of regime boundaries
        
    Methodology:
        - Compute rolling volatility of portfolio (equal-weighted) returns
        - Split into quantiles to define regime boundaries
        - This is a simple heuristic; not a full HMM or Markov-switching model
    """
    # Equal-weighted portfolio volatility as regime indicator
    port_ret = logrets.mean(axis=1)
    rolling_vol = port_ret.rolling(window=vol_window, min_periods=vol_window//2).std()
    rolling_vol = rolling_vol.fillna(rolling_vol.median())
    
    # Quantile-based regime assignment
    quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
    thresholds = rolling_vol.quantile(quantiles).values
    
    def assign_regime(v):
        for i, thresh in enumerate(thresholds):
            if v <= thresh:
                return i
        return n_regimes - 1
    
    regime_labels = rolling_vol.apply(assign_regime)
    return regime_labels, list(range(n_regimes))


def estimate_regime_params(
    logrets: pd.DataFrame,
    regime_labels: pd.Series,
    n_regimes: int = 3,
    drift_shrinkage: float = 0.0,
    stress_corr_alpha: float = 0.0,
) -> Dict[int, RegimeParams]:
    """Estimate (mu, sigma, corr) for each regime.
    
    Args:
        logrets: Full log-return DataFrame
        regime_labels: Series mapping dates to regime indices
        n_regimes: Number of regimes
        drift_shrinkage: Shrinkage factor (0=no shrinkage, 1=shrink to 0)
        stress_corr_alpha: Blend factor for stress correlation (0=empirical, 1=high-corr)
        
    Returns:
        Dict mapping regime index to RegimeParams
    """
    regime_params = {}
    
    for regime_idx in range(n_regimes):
        mask = regime_labels == regime_idx
        if mask.sum() < 10:
            # Not enough data; use full sample
            regime_rets = logrets
        else:
            regime_rets = logrets.loc[mask]
        
        mu = regime_rets.mean(axis=0).values
        sigma = regime_rets.std(axis=0, ddof=0).values
        corr = regime_rets.corr().values
        
        # Apply drift shrinkage: mu_shrunk = (1 - shrinkage) * mu + shrinkage * 0
        # This reduces noisy sample mean toward 0 (conservative assumption)
        if drift_shrinkage > 0:
            mu = mu * (1.0 - drift_shrinkage)
        
        # Apply correlation stress blending
        if stress_corr_alpha > 0:
            # High-correlation stress matrix (all pairwise = 0.8)
            n_assets = corr.shape[0]
            corr_high = np.full((n_assets, n_assets), 0.8)
            np.fill_diagonal(corr_high, 1.0)
            corr = (1.0 - stress_corr_alpha) * corr + stress_corr_alpha * corr_high
        
        # Ensure PSD for Cholesky
        L, _ = chol_with_jitter(corr)
        
        regime_name = ["low_vol", "med_vol", "high_vol"][regime_idx] if n_regimes == 3 else f"regime_{regime_idx}"
        regime_params[regime_idx] = RegimeParams(
            name=regime_name,
            mu=mu,
            sigma=sigma,
            corr=corr,
            L=L,
            logrets=regime_rets,
        )
    
    return regime_params


# =============================================================================
# DRIFT SHRINKAGE UTILITIES
# =============================================================================

def shrink_drift(
    mu: np.ndarray,
    shrinkage: float = 0.5,
    target: float = 0.0,
) -> np.ndarray:
    """Shrink drift estimate toward a target (default 0).
    
    Why: Sample mean return is very noisy and dominates MC outcomes unrealistically.
    Shrinking toward 0 or a risk-free baseline produces more conservative projections.
    
    Args:
        mu: Raw sample mean log-returns
        shrinkage: Factor in [0, 1]; 0=no shrinkage, 1=full shrinkage to target
        target: Shrinkage target (default 0, can be risk-free rate / 252)
        
    Returns:
        Shrunk drift estimate
    """
    return (1.0 - shrinkage) * mu + shrinkage * target


# =============================================================================
# CORE ESTIMATION
# =============================================================================

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


# =============================================================================
# BLOCK BOOTSTRAP
# =============================================================================

def sample_block_bootstrap(
    logrets: pd.DataFrame,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a contiguous block of historical returns (preserves temporal dependence).
    
    Why: Naive i.i.d. bootstrap breaks cross-asset and temporal dependence.
    Block bootstrap preserves short-term autocorrelation and cross-sectional structure.
    
    Args:
        logrets: Historical log-returns DataFrame
        block_size: Size of contiguous block to sample
        rng: Random number generator
        
    Returns:
        Block of log-returns, shape (block_size, n_assets)
    """
    n_obs = len(logrets)
    max_start = n_obs - block_size
    if max_start <= 0:
        # Not enough data; return full sample
        return logrets.values
    
    start_idx = rng.integers(0, max_start + 1)
    return logrets.iloc[start_idx:start_idx + block_size].values


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

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
    # New regime-switching parameters
    regime_params: Optional[Dict[int, RegimeParams]] = None,
    regime_switching: bool = False,
    regime_persistence: int = 20,  # avg steps before regime change
    block_size: int = 5,  # for block bootstrap
):
    """Simulate correlated GBM paths with optional regime switching.
    
    Args:
        S0: Initial prices, shape (n_assets,)
        mu_per_step: Per-step drift (used if regime_switching=False)
        sigma_per_step: Per-step volatility (used if regime_switching=False)
        L: Cholesky of correlation (used if regime_switching=False)
        n_steps: Number of simulation steps
        n_paths: Number of simulation paths
        seed: Random seed
        noise: Innovation distribution ("gaussian", "t", "bootstrap", "block_bootstrap")
        t_df: Degrees of freedom for t-distribution
        logrets: Historical log-returns (for bootstrap)
        regime_params: Dict of RegimeParams for regime-switching MC
        regime_switching: Whether to enable regime switching
        regime_persistence: Average steps before switching regime (Markov)
        block_size: Block size for block bootstrap
        
    Returns:
        Simulated paths, shape (n_steps+1, n_assets, n_paths)
    """
    rng = np.random.default_rng(seed)
    n_assets = S0.shape[0]
    dt = 1.0  # using per-step log-return stats already

    # output array: shape (n_steps+1, n_assets, n_paths)
    out = np.empty((n_steps + 1, n_assets, n_paths), dtype=float)
    out[0, :, :] = np.expand_dims(S0, axis=1)

    # Regime switching setup
    if regime_switching and regime_params is not None:
        n_regimes = len(regime_params)
        # Transition probability (simplified: uniform switch with persistence)
        # P(stay) = 1 - 1/persistence, P(switch to other) = uniform over others
        p_stay = 1.0 - 1.0 / regime_persistence
    else:
        regime_switching = False  # disable if no params

    for p in range(n_paths):
        S = S0.copy()
        
        # Initialize regime for this path (random start)
        if regime_switching:
            current_regime = rng.integers(0, n_regimes)
        
        # Block bootstrap: pre-sample blocks for this path
        block_idx = 0
        current_block = None
        
        for t in range(1, n_steps + 1):
            # Get regime-specific or global parameters
            if regime_switching:
                # Check for regime switch (Markov)
                if rng.random() > p_stay:
                    # Switch to a different regime (uniform over others)
                    other_regimes = [r for r in range(n_regimes) if r != current_regime]
                    current_regime = rng.choice(other_regimes)
                
                rp = regime_params[current_regime]
                mu = rp.mu
                sigma = rp.sigma
                L_current = rp.L
                regime_logrets = rp.logrets
            else:
                mu = mu_per_step
                sigma = sigma_per_step
                L_current = L
                regime_logrets = logrets
            
            # Generate innovation
            if noise == "gaussian":
                z = rng.standard_normal(size=n_assets)
                corr_z = L_current @ z
            elif noise == "t":
                # Student-t scaled to unit variance (variance = df/(df-2))
                if t_df <= 2:
                    raise ValueError("t_df must be > 2 for finite variance")
                z_t = rng.standard_t(t_df, size=n_assets)
                z = z_t * np.sqrt((t_df - 2.0) / t_df)
                corr_z = L_current @ z
            elif noise == "bootstrap":
                # Legacy: i.i.d. bootstrap (kept for backward compatibility)
                if regime_logrets is None:
                    raise ValueError("logrets must be provided for bootstrap noise")
                idx = rng.integers(0, len(regime_logrets))
                r = regime_logrets.iloc[idx].values
                corr_z = (r - mu) / sigma
            elif noise == "block_bootstrap":
                # Block bootstrap: preserves temporal dependence
                if regime_logrets is None:
                    raise ValueError("logrets must be provided for block_bootstrap noise")
                
                # Sample new block if needed
                if current_block is None or block_idx >= len(current_block):
                    current_block = sample_block_bootstrap(regime_logrets, block_size, rng)
                    block_idx = 0
                
                r = current_block[block_idx]
                block_idx += 1
                corr_z = (r - mu) / sigma
            else:
                raise ValueError(f"unknown noise type: {noise}")

            # GBM step using per-period log-return params
            # NOTE: use mu as mean(log-returns) directly (no extra -0.5*sigma^2)
            S = S * np.exp(mu * dt + sigma * corr_z * np.sqrt(dt))
            out[t, :, p] = S
    return out


def main():
    p = argparse.ArgumentParser(
        description="Monte Carlo GBM simulation with regime switching and drift shrinkage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Realism Enhancements:
  --regime-switching    Enable volatility regime switching (Markov-style)
  --drift-shrinkage     Shrink noisy sample drift toward 0 (0=none, 1=full)
  --stress-corr         Blend empirical correlation with high-correlation stress matrix
  --noise-dist block_bootstrap   Use block bootstrap to preserve temporal dependence
        """
    )
    p.add_argument("--price-csv", required=True, help="Wide CSV with index=Date and columns=tickers")
    p.add_argument("--n-paths", type=int, default=30)
    p.add_argument("--n-steps", type=int, default=252)
    p.add_argument("--out-dir", default="simulations")
    p.add_argument("--periods-per-year", type=int, default=252)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prefix", type=str, default="sim")
    p.add_argument("--noise-dist", type=str, default="gaussian", 
                   choices=["gaussian", "t", "bootstrap", "block_bootstrap"], 
                   help="Innovation distribution: gaussian, t, bootstrap, or block_bootstrap")
    p.add_argument("--t-df", type=float, default=5.0, help="Degrees of freedom for Student-t innovations (df>2)")
    
    # New realism parameters
    p.add_argument("--regime-switching", action="store_true", default=False,
                   help="Enable volatility regime switching (low/med/high vol)")
    p.add_argument("--n-regimes", type=int, default=3, help="Number of volatility regimes (default: 3)")
    p.add_argument("--regime-persistence", type=int, default=20,
                   help="Average steps before regime switch (default: 20)")
    p.add_argument("--drift-shrinkage", type=float, default=0.0,
                   help="Shrink drift toward 0: 0=none, 1=full shrinkage (default: 0)")
    p.add_argument("--stress-corr", type=float, default=0.0,
                   help="Blend factor for stress correlation: 0=empirical, 1=high-corr (default: 0)")
    p.add_argument("--block-size", type=int, default=5,
                   help="Block size for block_bootstrap noise (default: 5)")
    
    args = p.parse_args()

    df = pd.read_csv(args.price_csv, index_col=0, parse_dates=True)
    tickers = list(df.columns)
    if len(tickers) == 0:
        raise SystemExit("No tickers/columns found in price CSV")

    mean_r, std_r, corr, logrets = estimate_stats_and_corr(df)

    # Apply drift shrinkage to base parameters
    mu_step = shrink_drift(mean_r, shrinkage=args.drift_shrinkage)
    sigma_step = std_r

    # Apply stress correlation to base parameters
    if args.stress_corr > 0:
        n_assets = corr.shape[0]
        corr_high = np.full((n_assets, n_assets), 0.8)
        np.fill_diagonal(corr_high, 1.0)
        corr = (1.0 - args.stress_corr) * corr + args.stress_corr * corr_high

    # ensure corr is PD for Cholesky
    L, jitter = chol_with_jitter(corr)

    S0 = df.iloc[-1].astype(float).values

    # Regime switching setup
    regime_params = None
    if args.regime_switching:
        print(f"Detecting {args.n_regimes} volatility regimes...")
        regime_labels, _ = detect_regimes(logrets, n_regimes=args.n_regimes)
        regime_params = estimate_regime_params(
            logrets, 
            regime_labels, 
            n_regimes=args.n_regimes,
            drift_shrinkage=args.drift_shrinkage,
            stress_corr_alpha=args.stress_corr,
        )
        for r_idx, rp in regime_params.items():
            regime_count = (regime_labels == r_idx).sum()
            print(f"  Regime {r_idx} ({rp.name}): {regime_count} observations, "
                  f"avg_vol={rp.sigma.mean():.4f}")

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
        regime_params=regime_params,
        regime_switching=args.regime_switching,
        regime_persistence=args.regime_persistence,
        block_size=args.block_size,
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
        "mu_per_step": mu_step.tolist(),
        "sigma_per_step": sigma_step.tolist(),
        # New params
        "regime_switching": args.regime_switching,
        "n_regimes": args.n_regimes if args.regime_switching else 1,
        "regime_persistence": args.regime_persistence,
        "drift_shrinkage": args.drift_shrinkage,
        "stress_corr": args.stress_corr,
        "block_size": args.block_size if args.noise_dist == "block_bootstrap" else None,
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
