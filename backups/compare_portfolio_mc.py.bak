"""Compare Monte Carlo simulations for current vs optimized portfolios.

This module runs the Monte Carlo simulator (from simulate_gbm.py) on both
current and optimized portfolio allocations, producing side-by-side
comparisons of risk metrics (VaR, CVaR, percentiles, terminal distributions).

Key Features:
 - Uses existing GBM simulation infrastructure (no model changes)
 - Scales asset-level simulations by portfolio weights to get portfolio paths
 - Computes portfolio-level VaR, CVaR, and percentile summaries
 - Produces comparison reports showing risk/return tradeoffs

Non-goals (per Milestone 6.5 scope):
 - Does NOT change the underlying MC model
 - Does NOT modify RL rewards or environment
 - Uses pure scaling of existing asset simulations

Usage:
  python scripts/compare_portfolio_mc.py \\
      --price-csv data/price_matrix.csv \\
      --current-weights "AAPL:0.3,MSFT:0.25,GOOGL:0.25,AMZN:0.2" \\
      --target-weights "AAPL:0.2,MSFT:0.35,GOOGL:0.25,AMZN:0.2" \\
      --total-capital 100000 \\
      --n-paths 30 \\
      --n-steps 252 \\
      --out-dir simulations/comparison
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulate_gbm import estimate_stats_and_corr, chol_with_jitter, simulate_paths


@dataclass
class PortfolioMCResult:
    """Monte Carlo simulation results for a single portfolio."""
    
    name: str  # "current" or "optimized"
    weights: Dict[str, float]
    initial_value: float
    
    # Portfolio-level paths: shape (n_steps+1, n_paths)
    portfolio_paths: np.ndarray
    
    # Terminal values
    terminal_values: np.ndarray  # shape (n_paths,)
    terminal_returns: np.ndarray  # shape (n_paths,)
    
    # Summary statistics
    mean_terminal_value: float = 0.0
    std_terminal_value: float = 0.0
    mean_return: float = 0.0
    std_return: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # 5th percentile of returns (95% VaR)
    cvar_95: float = 0.0  # Expected Shortfall at 95%
    var_99: float = 0.0  # 1st percentile of returns (99% VaR)
    cvar_99: float = 0.0  # Expected Shortfall at 99%
    
    # Percentiles
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    def compute_statistics(self, percentile_list: List[int] = None):
        """Compute summary statistics from terminal values."""
        if percentile_list is None:
            percentile_list = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        self.mean_terminal_value = float(np.mean(self.terminal_values))
        self.std_terminal_value = float(np.std(self.terminal_values, ddof=0))
        self.mean_return = float(np.mean(self.terminal_returns))
        self.std_return = float(np.std(self.terminal_returns, ddof=0))
        
        # VaR at 95% (5th percentile of returns)
        self.var_95 = float(np.percentile(self.terminal_returns, 5))
        # CVaR at 95% (mean of returns below VaR)
        tail_95 = self.terminal_returns[self.terminal_returns <= self.var_95]
        self.cvar_95 = float(tail_95.mean()) if len(tail_95) > 0 else self.var_95
        
        # VaR at 99% (1st percentile of returns)
        self.var_99 = float(np.percentile(self.terminal_returns, 1))
        tail_99 = self.terminal_returns[self.terminal_returns <= self.var_99]
        self.cvar_99 = float(tail_99.mean()) if len(tail_99) > 0 else self.var_99
        
        # Percentiles
        for p in percentile_list:
            self.percentiles[p] = float(np.percentile(self.terminal_returns, p))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "weights": self.weights,
            "initial_value": float(self.initial_value),
            "mean_terminal_value": float(self.mean_terminal_value),
            "std_terminal_value": float(self.std_terminal_value),
            "mean_return": float(self.mean_return),
            "std_return": float(self.std_return),
            "var_95": float(self.var_95),
            "cvar_95": float(self.cvar_95),
            "var_99": float(self.var_99),
            "cvar_99": float(self.cvar_99),
            "percentiles": {str(k): float(v) for k, v in self.percentiles.items()},
        }


@dataclass
class PortfolioComparison:
    """Comparison of current vs optimized portfolio MC results."""
    
    current: PortfolioMCResult
    optimized: PortfolioMCResult
    
    # Comparison metrics (optimized - current)
    return_improvement: float = 0.0
    risk_reduction_var95: float = 0.0
    risk_reduction_cvar95: float = 0.0
    sharpe_current: float = 0.0
    sharpe_optimized: float = 0.0
    
    def compute_comparison(self):
        """Compute comparison metrics."""
        self.return_improvement = self.optimized.mean_return - self.current.mean_return
        self.risk_reduction_var95 = self.current.var_95 - self.optimized.var_95  # Positive = less downside
        self.risk_reduction_cvar95 = self.current.cvar_95 - self.optimized.cvar_95
        
        # Simple Sharpe-like ratio (mean return / std return)
        if self.current.std_return > 0:
            self.sharpe_current = self.current.mean_return / self.current.std_return
        if self.optimized.std_return > 0:
            self.sharpe_optimized = self.optimized.mean_return / self.optimized.std_return
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "current": self.current.to_dict(),
            "optimized": self.optimized.to_dict(),
            "comparison": {
                "return_improvement": float(self.return_improvement),
                "risk_reduction_var95": float(self.risk_reduction_var95),
                "risk_reduction_cvar95": float(self.risk_reduction_cvar95),
                "sharpe_current": float(self.sharpe_current),
                "sharpe_optimized": float(self.sharpe_optimized),
            }
        }
    
    def generate_report(self) -> str:
        """Generate a human-readable comparison report."""
        lines = [
            "=" * 80,
            "MONTE CARLO PORTFOLIO COMPARISON",
            "=" * 80,
            "",
            f"Simulation Horizon: {self.current.portfolio_paths.shape[0] - 1} steps",
            f"Number of Paths: {self.current.portfolio_paths.shape[1]}",
            "",
            "-" * 80,
            "CURRENT PORTFOLIO",
            "-" * 80,
            f"  Initial Value:        ${self.current.initial_value:>12,.2f}",
            f"  Mean Terminal Value:  ${self.current.mean_terminal_value:>12,.2f}",
            f"  Std Terminal Value:   ${self.current.std_terminal_value:>12,.2f}",
            f"  Mean Return:          {self.current.mean_return:>12.2%}",
            f"  Std Return:           {self.current.std_return:>12.2%}",
            f"  VaR (95%):            {self.current.var_95:>12.2%}",
            f"  CVaR (95%):           {self.current.cvar_95:>12.2%}",
            f"  VaR (99%):            {self.current.var_99:>12.2%}",
            f"  CVaR (99%):           {self.current.cvar_99:>12.2%}",
            "",
            "  Weights:",
        ]
        for ticker, weight in sorted(self.current.weights.items()):
            lines.append(f"    {ticker}: {weight:.2%}")
        
        lines.extend([
            "",
            "-" * 80,
            "OPTIMIZED PORTFOLIO (RL)",
            "-" * 80,
            f"  Initial Value:        ${self.optimized.initial_value:>12,.2f}",
            f"  Mean Terminal Value:  ${self.optimized.mean_terminal_value:>12,.2f}",
            f"  Std Terminal Value:   ${self.optimized.std_terminal_value:>12,.2f}",
            f"  Mean Return:          {self.optimized.mean_return:>12.2%}",
            f"  Std Return:           {self.optimized.std_return:>12.2%}",
            f"  VaR (95%):            {self.optimized.var_95:>12.2%}",
            f"  CVaR (95%):           {self.optimized.cvar_95:>12.2%}",
            f"  VaR (99%):            {self.optimized.var_99:>12.2%}",
            f"  CVaR (99%):           {self.optimized.cvar_99:>12.2%}",
            "",
            "  Weights:",
        ])
        for ticker, weight in sorted(self.optimized.weights.items()):
            lines.append(f"    {ticker}: {weight:.2%}")
        
        # Comparison section
        lines.extend([
            "",
            "-" * 80,
            "COMPARISON (Optimized vs Current)",
            "-" * 80,
            f"  Return Improvement:    {self.return_improvement:>+12.2%}",
            f"  VaR 95% Reduction:     {self.risk_reduction_var95:>+12.2%}",
            f"  CVaR 95% Reduction:    {self.risk_reduction_cvar95:>+12.2%}",
            f"  Sharpe (Current):      {self.sharpe_current:>12.3f}",
            f"  Sharpe (Optimized):    {self.sharpe_optimized:>12.3f}",
            "",
        ])
        
        # Interpretation
        lines.append("-" * 80)
        lines.append("INTERPRETATION")
        lines.append("-" * 80)
        
        if self.return_improvement > 0.01:
            lines.append(f"  ✓ Optimized portfolio shows higher expected return (+{self.return_improvement:.2%})")
        elif self.return_improvement < -0.01:
            lines.append(f"  ✗ Optimized portfolio shows lower expected return ({self.return_improvement:.2%})")
        else:
            lines.append(f"  ○ Expected returns are similar between portfolios")
        
        if self.risk_reduction_var95 > 0.01:
            lines.append(f"  ✓ Optimized portfolio reduces tail risk (VaR improved by {self.risk_reduction_var95:.2%})")
        elif self.risk_reduction_var95 < -0.01:
            lines.append(f"  ✗ Optimized portfolio has higher tail risk (VaR worse by {abs(self.risk_reduction_var95):.2%})")
        else:
            lines.append(f"  ○ Tail risk is similar between portfolios")
        
        if self.sharpe_optimized > self.sharpe_current + 0.05:
            lines.append(f"  ✓ Optimized portfolio has better risk-adjusted return (Sharpe ratio)")
        elif self.sharpe_optimized < self.sharpe_current - 0.05:
            lines.append(f"  ✗ Current portfolio has better risk-adjusted return (Sharpe ratio)")
        else:
            lines.append(f"  ○ Risk-adjusted returns are similar")
        
        lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])
        
        return "\n".join(lines)


def simulate_portfolio_paths(
    asset_sims: np.ndarray,
    weights: np.ndarray,
    tickers: List[str],
    initial_capital: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert asset-level simulations to portfolio-level paths.
    
    Args:
        asset_sims: Asset simulations, shape (n_steps+1, n_assets, n_paths)
        weights: Portfolio weights, shape (n_assets,)
        tickers: List of ticker names
        initial_capital: Initial portfolio value
        
    Returns:
        Tuple of (portfolio_paths, terminal_values, terminal_returns)
        - portfolio_paths: shape (n_steps+1, n_paths)
        - terminal_values: shape (n_paths,)
        - terminal_returns: shape (n_paths,)
    """
    n_steps_plus_1, n_assets, n_paths = asset_sims.shape
    
    # Initial prices (S0 for each asset)
    S0 = asset_sims[0, :, 0]  # shape (n_assets,)
    
    # Convert weights to number of shares (notional) based on initial capital
    # shares_i = (weight_i * initial_capital) / S0_i
    shares = (weights * initial_capital) / S0
    
    # Portfolio value at each step and path
    # V_t = sum_i (shares_i * S_t_i)
    portfolio_paths = np.zeros((n_steps_plus_1, n_paths))
    
    for t in range(n_steps_plus_1):
        for p in range(n_paths):
            portfolio_paths[t, p] = np.sum(shares * asset_sims[t, :, p])
    
    # Terminal values and returns
    terminal_values = portfolio_paths[-1, :]
    terminal_returns = (terminal_values / initial_capital) - 1.0
    
    return portfolio_paths, terminal_values, terminal_returns


def run_portfolio_comparison(
    price_df: pd.DataFrame,
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    initial_capital: float,
    n_paths: int = 30,
    n_steps: int = 252,
    seed: Optional[int] = None,
    noise: str = "gaussian",
    t_df: float = 5.0,
    # New realism parameters
    regime_switching: bool = False,
    n_regimes: int = 3,
    regime_persistence: int = 20,
    drift_shrinkage: float = 0.0,
    stress_corr: float = 0.0,
    block_size: int = 5,
) -> PortfolioComparison:
    """Run Monte Carlo comparison for current vs optimized portfolios.
    
    Args:
        price_df: Historical price data (dates x tickers)
        current_weights: Current portfolio weights {ticker: weight}
        target_weights: Optimized portfolio weights {ticker: weight}
        initial_capital: Initial portfolio value
        n_paths: Number of simulation paths
        n_steps: Number of simulation steps
        seed: Random seed for reproducibility
        noise: Noise distribution ("gaussian", "t", "bootstrap", or "block_bootstrap")
        t_df: Degrees of freedom for t-distribution
        regime_switching: Enable volatility regime switching
        n_regimes: Number of volatility regimes
        regime_persistence: Average steps before regime switch
        drift_shrinkage: Shrink drift toward 0 (0=none, 1=full)
        stress_corr: Blend factor for stress correlation (0=empirical, 1=high-corr)
        block_size: Block size for block_bootstrap
        
    Returns:
        PortfolioComparison with results for both portfolios
    """
    tickers = list(price_df.columns)
    
    # Prepare weight arrays aligned to tickers
    current_w = np.array([current_weights.get(t, 0.0) for t in tickers])
    target_w = np.array([target_weights.get(t, 0.0) for t in tickers])
    
    # Normalize weights
    if current_w.sum() > 0:
        current_w = current_w / current_w.sum()
    if target_w.sum() > 0:
        target_w = target_w / target_w.sum()
    
    # Estimate GBM parameters from historical data
    mean_r, std_r, corr, logrets = estimate_stats_and_corr(price_df)
    
    # Apply drift shrinkage
    if drift_shrinkage > 0:
        mean_r = mean_r * (1.0 - drift_shrinkage)
    
    # Apply stress correlation
    if stress_corr > 0:
        n_assets = corr.shape[0]
        corr_high = np.full((n_assets, n_assets), 0.8)
        np.fill_diagonal(corr_high, 1.0)
        corr = (1.0 - stress_corr) * corr + stress_corr * corr_high
    
    L, jitter = chol_with_jitter(corr)
    S0 = price_df.iloc[-1].astype(float).values
    
    # Regime switching setup
    regime_params = None
    if regime_switching:
        from simulate_gbm import detect_regimes, estimate_regime_params
        regime_labels, _ = detect_regimes(logrets, n_regimes=n_regimes)
        regime_params = estimate_regime_params(
            logrets, 
            regime_labels, 
            n_regimes=n_regimes,
            drift_shrinkage=drift_shrinkage,
            stress_corr_alpha=stress_corr,
        )
    
    # Run asset-level simulations (same for both portfolios)
    asset_sims = simulate_paths(
        S0=S0,
        mu_per_step=mean_r,
        sigma_per_step=std_r,
        L=L,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        noise=noise,
        t_df=t_df,
        logrets=logrets,
        regime_params=regime_params,
        regime_switching=regime_switching,
        regime_persistence=regime_persistence,
        block_size=block_size,
    )
    
    # Convert to portfolio-level paths for current weights
    current_paths, current_terminal, current_returns = simulate_portfolio_paths(
        asset_sims=asset_sims,
        weights=current_w,
        tickers=tickers,
        initial_capital=initial_capital,
    )
    
    # Convert to portfolio-level paths for optimized weights
    target_paths, target_terminal, target_returns = simulate_portfolio_paths(
        asset_sims=asset_sims,
        weights=target_w,
        tickers=tickers,
        initial_capital=initial_capital,
    )
    
    # Create result objects
    current_result = PortfolioMCResult(
        name="current",
        weights={t: float(w) for t, w in zip(tickers, current_w)},
        initial_value=initial_capital,
        portfolio_paths=current_paths,
        terminal_values=current_terminal,
        terminal_returns=current_returns,
    )
    current_result.compute_statistics()
    
    target_result = PortfolioMCResult(
        name="optimized",
        weights={t: float(w) for t, w in zip(tickers, target_w)},
        initial_value=initial_capital,
        portfolio_paths=target_paths,
        terminal_values=target_terminal,
        terminal_returns=target_returns,
    )
    target_result.compute_statistics()
    
    # Create comparison
    comparison = PortfolioComparison(current=current_result, optimized=target_result)
    comparison.compute_comparison()
    
    return comparison


def save_comparison_results(
    comparison: PortfolioComparison,
    out_dir: str,
    prefix: str = "portfolio_comparison",
) -> Dict[str, str]:
    """Save comparison results to files.
    
    Args:
        comparison: PortfolioComparison results
        out_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dictionary of output file paths
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save JSON summary
    json_path = out_path / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)
    paths["json"] = str(json_path)
    
    # Save text report
    report_path = out_path / f"{prefix}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(comparison.generate_report())
    paths["report"] = str(report_path)
    
    # Save current portfolio paths
    current_paths_df = pd.DataFrame(
        comparison.current.portfolio_paths,
        columns=[f"path_{i:03d}" for i in range(comparison.current.portfolio_paths.shape[1])]
    )
    current_paths_df.index.name = "step"
    current_csv = out_path / f"{prefix}_current_paths.csv"
    current_paths_df.to_csv(current_csv)
    paths["current_paths"] = str(current_csv)
    
    # Save optimized portfolio paths
    target_paths_df = pd.DataFrame(
        comparison.optimized.portfolio_paths,
        columns=[f"path_{i:03d}" for i in range(comparison.optimized.portfolio_paths.shape[1])]
    )
    target_paths_df.index.name = "step"
    target_csv = out_path / f"{prefix}_optimized_paths.csv"
    target_paths_df.to_csv(target_csv)
    paths["optimized_paths"] = str(target_csv)
    
    # Save terminal distribution comparison
    terminal_df = pd.DataFrame({
        "current_value": comparison.current.terminal_values,
        "current_return": comparison.current.terminal_returns,
        "optimized_value": comparison.optimized.terminal_values,
        "optimized_return": comparison.optimized.terminal_returns,
    })
    terminal_csv = out_path / f"{prefix}_terminal_distribution.csv"
    terminal_df.to_csv(terminal_csv, index=False)
    paths["terminal_distribution"] = str(terminal_csv)
    
    # Save summary comparison table
    summary_rows = []
    for result in [comparison.current, comparison.optimized]:
        row = {
            "portfolio": result.name,
            "initial_value": result.initial_value,
            "mean_terminal_value": result.mean_terminal_value,
            "std_terminal_value": result.std_terminal_value,
            "mean_return": result.mean_return,
            "std_return": result.std_return,
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
            "var_99": result.var_99,
            "cvar_99": result.cvar_99,
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_path / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    paths["summary"] = str(summary_csv)
    
    return paths


def parse_weights_string(weights_str: str) -> Dict[str, float]:
    """Parse a weights string like 'AAPL:0.3,MSFT:0.25' into a dict."""
    weights = {}
    for pair in weights_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            ticker, weight = pair.split(":", 1)
            weights[ticker.strip()] = float(weight.strip())
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Compare Monte Carlo simulations for current vs optimized portfolios"
    )
    parser.add_argument("--price-csv", required=True, help="Wide CSV with index=Date and columns=tickers")
    parser.add_argument(
        "--current-weights",
        required=True,
        help="Current weights as 'TICKER:weight,TICKER:weight' or path to CSV"
    )
    parser.add_argument(
        "--target-weights",
        required=True,
        help="Target/optimized weights as 'TICKER:weight,TICKER:weight' or path to CSV"
    )
    parser.add_argument("--total-capital", type=float, default=100000, help="Total capital to simulate")
    parser.add_argument("--n-paths", type=int, default=30, help="Number of simulation paths")
    parser.add_argument("--n-steps", type=int, default=252, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out-dir", default="simulations/comparison", help="Output directory")
    parser.add_argument("--noise-dist", type=str, default="gaussian", 
                       choices=["gaussian", "t", "bootstrap"], help="Innovation distribution")
    parser.add_argument("--t-df", type=float, default=5.0, help="Degrees of freedom for t-distribution")
    
    args = parser.parse_args()
    
    # Load price data
    price_df = pd.read_csv(args.price_csv, index_col=0, parse_dates=True)
    
    # Parse current weights
    if Path(args.current_weights).exists():
        wdf = pd.read_csv(args.current_weights)
        current_weights = dict(zip(wdf.iloc[:, 0], wdf.iloc[:, 1]))
    else:
        current_weights = parse_weights_string(args.current_weights)
    
    # Parse target weights
    if Path(args.target_weights).exists():
        wdf = pd.read_csv(args.target_weights)
        target_weights = dict(zip(wdf.iloc[:, 0], wdf.iloc[:, 1]))
    else:
        target_weights = parse_weights_string(args.target_weights)
    
    print(f"Running Monte Carlo comparison...")
    print(f"  Price data: {args.price_csv} ({len(price_df)} rows, {len(price_df.columns)} assets)")
    print(f"  Initial capital: ${args.total_capital:,.2f}")
    print(f"  Simulation: {args.n_paths} paths × {args.n_steps} steps")
    print(f"  Noise distribution: {args.noise_dist}")
    print()
    
    # Run comparison
    comparison = run_portfolio_comparison(
        price_df=price_df,
        current_weights=current_weights,
        target_weights=target_weights,
        initial_capital=args.total_capital,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=args.seed,
        noise=args.noise_dist,
        t_df=args.t_df,
    )
    
    # Print report
    print(comparison.generate_report())
    
    # Save outputs
    paths = save_comparison_results(comparison, args.out_dir)
    print(f"\nOutputs saved to {args.out_dir}:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
