"""
Evaluate Holdings Module

Evaluates user's current holdings in monetary terms and compares 
to RL-optimized target allocations. Produces detailed comparison reports.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_parser import CapitalConfig, parse_excel, parse_capital_config
from allocation_converter import (
    normalize_weights,
    weights_to_allocations,
    allocations_to_weights,
    compare_allocations,
    holdings_to_weights,
    AllocationResult,
    PortfolioComparison,
)


def fetch_current_prices(tickers: List[str], cache_dir: str = "data_store") -> Dict[str, float]:
    """Fetch current prices for tickers using yfinance.
    
    Args:
        tickers: List of ticker symbols
        cache_dir: Directory for caching (not used for live prices)
        
    Returns:
        Dictionary mapping ticker -> current price
    """
    import yfinance as yf
    
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                prices[ticker] = float(hist["Close"].iloc[-1])
            else:
                # Try getting info as fallback
                info = stock.info
                prices[ticker] = info.get("regularMarketPrice", 0.0) or info.get("previousClose", 0.0)
        except Exception as e:
            print(f"Warning: Could not fetch price for {ticker}: {e}")
            prices[ticker] = 0.0
    
    return prices


def load_prices_from_csv(csv_path: str, date_col: str = None) -> Dict[str, float]:
    """Load latest prices from a price matrix CSV.
    
    Args:
        csv_path: Path to price matrix CSV (dates as rows, tickers as columns)
        date_col: Name of date column (if not index)
        
    Returns:
        Dictionary mapping ticker -> latest price
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Get the last row (most recent prices)
    last_prices = df.iloc[-1]
    
    return last_prices.to_dict()


@dataclass
class HoldingsEvaluation:
    """Complete evaluation of user holdings vs optimized allocation."""
    
    # Holdings information
    holdings_df: pd.DataFrame
    prices: Dict[str, float]
    
    # Current portfolio state
    current_values: np.ndarray  # Market value per asset
    current_weights: np.ndarray
    total_holdings_value: float
    
    # Capital configuration
    capital_config: CapitalConfig
    
    # Optimized portfolio (if provided)
    optimized_weights: Optional[np.ndarray] = None
    comparison: Optional[PortfolioComparison] = None
    
    # Metadata
    evaluation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def tickers(self) -> List[str]:
        return self.holdings_df["Ticker"].tolist()
    
    @property
    def quantities(self) -> np.ndarray:
        return self.holdings_df["Quantity"].values
    
    @property
    def uninvested_cash(self) -> float:
        """Cash available but not currently invested in holdings."""
        return max(0, self.capital_config.investable_capital - self.total_holdings_value)
    
    @property
    def investment_ratio(self) -> float:
        """Fraction of investable capital currently invested."""
        if self.capital_config.investable_capital <= 0:
            return 0.0
        return min(1.0, self.total_holdings_value / self.capital_config.investable_capital)
    
    def holdings_summary(self) -> pd.DataFrame:
        """Generate detailed holdings summary DataFrame."""
        return pd.DataFrame({
            "Ticker": self.tickers,
            "Quantity": self.quantities,
            "Price": [self.prices.get(t, 0.0) for t in self.tickers],
            "Market_Value": self.current_values,
            "Weight": self.current_weights,
            "Weight_Pct": self.current_weights * 100,
        })
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "evaluation_date": self.evaluation_date,
            "capital": {
                "total_capital": self.capital_config.total_capital,
                "reserved_cash": self.capital_config.reserved_cash,
                "investable_capital": self.capital_config.investable_capital,
            },
            "holdings": {
                "total_value": self.total_holdings_value,
                "uninvested_cash": self.uninvested_cash,
                "investment_ratio": self.investment_ratio,
                "positions": [
                    {
                        "ticker": t,
                        "quantity": float(q),
                        "price": self.prices.get(t, 0.0),
                        "value": float(v),
                        "weight": float(w),
                    }
                    for t, q, v, w in zip(
                        self.tickers, self.quantities, 
                        self.current_values, self.current_weights
                    )
                ]
            }
        }
        
        if self.optimized_weights is not None:
            result["optimized"] = {
                "weights": self.optimized_weights.tolist(),
            }
        
        if self.comparison is not None:
            result["comparison"] = {
                "weight_deltas": self.comparison.weight_deltas.tolist(),
                "allocation_deltas": self.comparison.allocation_deltas.tolist(),
            }
        
        return result
    
    def summary_report(self) -> str:
        """Generate human-readable summary report."""
        lines = [
            "=" * 80,
            "PORTFOLIO HOLDINGS EVALUATION",
            "=" * 80,
            f"Evaluation Date: {self.evaluation_date}",
            "",
            "CAPITAL SUMMARY",
            "-" * 40,
            f"  Total Capital:       ${self.capital_config.total_capital:>15,.2f}",
            f"  Reserved Cash:       ${self.capital_config.reserved_cash:>15,.2f}",
            f"  Investable Capital:  ${self.capital_config.investable_capital:>15,.2f}",
            "",
            "HOLDINGS SUMMARY",
            "-" * 40,
            f"  Total Holdings Value: ${self.total_holdings_value:>14,.2f}",
            f"  Uninvested Cash:      ${self.uninvested_cash:>14,.2f}",
            f"  Investment Ratio:     {self.investment_ratio:>14.1%}",
            "",
            "POSITIONS",
            "-" * 80,
            f"{'Ticker':<10} {'Qty':>10} {'Price':>12} {'Value':>14} {'Weight':>10}",
            "-" * 80,
        ]
        
        for i, ticker in enumerate(self.tickers):
            lines.append(
                f"{ticker:<10} "
                f"{self.quantities[i]:>10.2f} "
                f"${self.prices.get(ticker, 0):>11,.2f} "
                f"${self.current_values[i]:>13,.2f} "
                f"{self.current_weights[i]:>9.1%}"
            )
        
        lines.append("-" * 80)
        lines.append(
            f"{'TOTAL':<10} "
            f"{'':>10} "
            f"{'':>12} "
            f"${self.total_holdings_value:>13,.2f} "
            f"{sum(self.current_weights):>9.1%}"
        )
        
        if self.comparison is not None:
            lines.extend([
                "",
                "=" * 80,
                "COMPARISON: CURRENT vs OPTIMIZED ALLOCATION",
                "=" * 80,
                "",
                self.comparison.summary(),
            ])
        
        return "\n".join(lines)


def evaluate_holdings(
    holdings_df: pd.DataFrame,
    prices: Dict[str, float],
    capital_config: CapitalConfig,
    optimized_weights: Optional[np.ndarray] = None,
) -> HoldingsEvaluation:
    """Evaluate current holdings in monetary terms.
    
    Args:
        holdings_df: DataFrame with 'Ticker' and 'Quantity' columns
        prices: Current prices per ticker
        capital_config: User's capital configuration
        optimized_weights: Optional RL-optimized weights for comparison
        
    Returns:
        HoldingsEvaluation with complete analysis
    """
    tickers = holdings_df["Ticker"].tolist()
    quantities = holdings_df["Quantity"].values.astype(float)
    
    # Get prices for each ticker
    ticker_prices = np.array([prices.get(t, 0.0) for t in tickers])
    
    # Compute market values
    current_values = quantities * ticker_prices
    total_value = current_values.sum()
    
    # Compute weights relative to investable capital
    investable = capital_config.investable_capital
    if investable > 0:
        current_weights = current_values / investable
    else:
        current_weights = np.zeros(len(current_values))
    
    # Create comparison if optimized weights provided
    comparison = None
    if optimized_weights is not None:
        # Normalize current weights to sum to 1 for fair comparison
        # (assuming user wants to be fully invested)
        current_weights_normalized = normalize_weights(current_weights)
        optimized_weights_normalized = normalize_weights(optimized_weights)
        
        comparison = compare_allocations(
            current_weights=current_weights_normalized,
            optimized_weights=optimized_weights_normalized,
            capital_config=capital_config,
            tickers=tickers,
        )
    
    return HoldingsEvaluation(
        holdings_df=holdings_df,
        prices=prices,
        current_values=current_values,
        current_weights=current_weights,
        total_holdings_value=total_value,
        capital_config=capital_config,
        optimized_weights=optimized_weights,
        comparison=comparison,
    )


def load_optimized_weights(
    model_path: str,
    price_df: pd.DataFrame,
    n_assets: int,
) -> Tuple[np.ndarray, List[str]]:
    """Load RL model and get optimized weights for current state.
    
    Args:
        model_path: Path to trained PPO model
        price_df: Price DataFrame for environment
        n_assets: Number of assets
        
    Returns:
        Tuple of (optimized_weights, tickers)
    """
    from stable_baselines3 import PPO
    from portfolio_env import PortfolioEnv
    
    env = PortfolioEnv(
        n_assets=n_assets,
        price_df=price_df,
        transaction_cost=0.001,
    )
    
    model = PPO.load(model_path)
    
    # Reset to get initial observation
    obs, _ = env.reset()
    
    # Get model's recommended action (weights)
    action, _ = model.predict(obs, deterministic=True)
    
    # Normalize to proper weights
    weights = normalize_weights(action)
    tickers = list(price_df.columns)
    
    return weights, tickers


def evaluate_from_excel(
    excel_path: str,
    capital_config: CapitalConfig,
    price_csv: Optional[str] = None,
    model_path: Optional[str] = None,
    fetch_live_prices: bool = True,
) -> HoldingsEvaluation:
    """Convenience function to evaluate holdings from Excel file.
    
    Args:
        excel_path: Path to Excel file with holdings
        capital_config: Capital configuration
        price_csv: Optional path to price CSV (uses last row)
        model_path: Optional path to RL model for comparison
        fetch_live_prices: If True and no price_csv, fetch live prices
        
    Returns:
        HoldingsEvaluation with complete analysis
    """
    # Parse holdings from Excel
    holdings_df = parse_excel(excel_path)
    tickers = holdings_df["Ticker"].tolist()
    
    # Get prices
    if price_csv:
        prices = load_prices_from_csv(price_csv)
    elif fetch_live_prices:
        prices = fetch_current_prices(tickers)
    else:
        raise ValueError("Must provide price_csv or enable fetch_live_prices")
    
    # Get optimized weights if model provided
    optimized_weights = None
    if model_path and price_csv:
        price_df = pd.read_csv(price_csv, index_col=0, parse_dates=True)
        n_assets = len(tickers)
        optimized_weights, _ = load_optimized_weights(model_path, price_df, n_assets)
    
    return evaluate_holdings(
        holdings_df=holdings_df,
        prices=prices,
        capital_config=capital_config,
        optimized_weights=optimized_weights,
    )


def save_evaluation(
    evaluation: HoldingsEvaluation,
    out_dir: str,
    prefix: str = "holdings_eval",
) -> Dict[str, str]:
    """Save evaluation results to files.
    
    Args:
        evaluation: HoldingsEvaluation to save
        out_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dictionary of output file paths
    """
    os.makedirs(out_dir, exist_ok=True)
    
    paths = {}
    
    # Save JSON summary
    json_path = os.path.join(out_dir, f"{prefix}.json")
    with open(json_path, "w") as f:
        json.dump(evaluation.to_dict(), f, indent=2)
    paths["json"] = json_path
    
    # Save holdings CSV
    holdings_csv = os.path.join(out_dir, f"{prefix}_holdings.csv")
    evaluation.holdings_summary().to_csv(holdings_csv, index=False)
    paths["holdings_csv"] = holdings_csv
    
    # Save comparison CSV if available
    if evaluation.comparison is not None:
        comparison_csv = os.path.join(out_dir, f"{prefix}_comparison.csv")
        evaluation.comparison.to_dataframe().to_csv(comparison_csv, index=False)
        paths["comparison_csv"] = comparison_csv
    
    # Save text report
    report_path = os.path.join(out_dir, f"{prefix}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(evaluation.summary_report())
    paths["report"] = report_path
    
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate current holdings and compare to optimized allocation"
    )
    parser.add_argument(
        "--holdings", type=str,
        help="Path to Excel file with holdings (Ticker, Quantity columns)"
    )
    parser.add_argument(
        "--price-csv", type=str,
        help="Path to price matrix CSV (uses latest prices)"
    )
    parser.add_argument(
        "--total-capital", type=float, required=True,
        help="Total capital available"
    )
    parser.add_argument(
        "--reserved-cash", type=float, default=0.0,
        help="Amount to keep as cash reserve"
    )
    parser.add_argument(
        "--model-path", type=str,
        help="Path to trained RL model for comparison"
    )
    parser.add_argument(
        "--optimized-weights", type=float, nargs="+",
        help="Manual optimized weights for comparison (space-separated)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--fetch-prices", action="store_true",
        help="Fetch live prices from yfinance"
    )
    
    args = parser.parse_args()
    
    # Create capital config
    capital_config = parse_capital_config(args.total_capital, args.reserved_cash)
    
    print(f"Capital Configuration:")
    print(f"  Total Capital:      ${capital_config.total_capital:,.2f}")
    print(f"  Reserved Cash:      ${capital_config.reserved_cash:,.2f}")
    print(f"  Investable Capital: ${capital_config.investable_capital:,.2f}")
    print()
    
    # Load holdings or create sample
    if args.holdings:
        holdings_df = parse_excel(args.holdings)
    else:
        # Demo with sample holdings
        print("No holdings file provided. Using sample holdings for demo.")
        holdings_df = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "Quantity": [50, 30, 10, 20, 40],
        })
    
    tickers = holdings_df["Ticker"].tolist()
    print(f"Holdings: {len(tickers)} positions")
    print(holdings_df.to_string(index=False))
    print()
    
    # Get prices
    if args.price_csv:
        print(f"Loading prices from: {args.price_csv}")
        prices = load_prices_from_csv(args.price_csv)
    elif args.fetch_prices:
        print("Fetching live prices from yfinance...")
        prices = fetch_current_prices(tickers)
    else:
        # Use dummy prices for demo
        print("Using dummy prices for demo (use --price-csv or --fetch-prices for real data)")
        prices = {t: 100.0 + i * 50 for i, t in enumerate(tickers)}
    
    print("\nCurrent Prices:")
    for t, p in prices.items():
        if t in tickers:
            print(f"  {t}: ${p:,.2f}")
    print()
    
    # Get optimized weights
    optimized_weights = None
    if args.optimized_weights:
        optimized_weights = np.array(args.optimized_weights)
        print(f"Using provided optimized weights: {optimized_weights}")
    elif args.model_path and args.price_csv:
        print(f"Loading optimized weights from model: {args.model_path}")
        price_df = pd.read_csv(args.price_csv, index_col=0, parse_dates=True)
        optimized_weights, _ = load_optimized_weights(
            args.model_path, price_df, len(tickers)
        )
    
    # Evaluate holdings
    evaluation = evaluate_holdings(
        holdings_df=holdings_df,
        prices=prices,
        capital_config=capital_config,
        optimized_weights=optimized_weights,
    )
    
    # Print report
    print("\n" + evaluation.summary_report())
    
    # Save results
    print(f"\nSaving results to: {args.out_dir}")
    paths = save_evaluation(evaluation, args.out_dir)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
