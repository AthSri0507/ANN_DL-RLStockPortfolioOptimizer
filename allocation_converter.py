"""
Allocation Converter Module

Converts relative portfolio weights (from RL agent) to currency allocations
using investable capital. The RL agent operates on normalized weights (0-1),
and this module scales them to actual dollar amounts.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from input_parser import CapitalConfig


@dataclass
class AllocationResult:
    """Result of converting weights to currency allocations.
    
    Attributes:
        tickers: List of asset tickers
        weights: Relative weights (sum to 1.0)
        allocations: Currency amounts per asset
        investable_capital: Total capital being allocated
        capital_config: Original capital configuration
    """
    tickers: List[str]
    weights: np.ndarray
    allocations: np.ndarray
    investable_capital: float
    capital_config: CapitalConfig
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with ticker, weight, and allocation columns."""
        return pd.DataFrame({
            "Ticker": self.tickers,
            "Weight": self.weights,
            "Allocation": self.allocations,
            "Allocation_Pct": self.weights * 100,
        })
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "tickers": self.tickers,
            "weights": self.weights.tolist(),
            "allocations": self.allocations.tolist(),
            "investable_capital": self.investable_capital,
            "total_capital": self.capital_config.total_capital,
            "reserved_cash": self.capital_config.reserved_cash,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Portfolio Allocation Summary",
            f"=" * 40,
            f"Total Capital:      ${self.capital_config.total_capital:>12,.2f}",
            f"Reserved Cash:      ${self.capital_config.reserved_cash:>12,.2f}",
            f"Investable Capital: ${self.investable_capital:>12,.2f}",
            f"",
            f"{'Ticker':<10} {'Weight':>10} {'Allocation':>15}",
            f"-" * 40,
        ]
        for ticker, weight, alloc in zip(self.tickers, self.weights, self.allocations):
            lines.append(f"{ticker:<10} {weight:>10.2%} ${alloc:>14,.2f}")
        lines.append(f"-" * 40)
        lines.append(f"{'Total':<10} {np.sum(self.weights):>10.2%} ${np.sum(self.allocations):>14,.2f}")
        return "\n".join(lines)


def normalize_weights(weights: np.ndarray, min_weight: float = 0.0) -> np.ndarray:
    """Normalize weights to sum to 1.0.
    
    Args:
        weights: Raw weight array (can be any positive values)
        min_weight: Minimum weight threshold; weights below this are set to 0
        
    Returns:
        Normalized weights summing to 1.0
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)  # Ensure non-negative
    
    # Apply minimum weight threshold
    if min_weight > 0:
        weights[weights < min_weight] = 0.0
    
    total = weights.sum()
    if total <= 0:
        # If all weights are zero, return equal weights
        return np.ones(len(weights)) / len(weights)
    
    return weights / total


def weights_to_allocations(
    weights: Union[np.ndarray, List[float]],
    capital_config: CapitalConfig,
    tickers: Optional[List[str]] = None,
    normalize: bool = True,
) -> AllocationResult:
    """Convert relative weights to currency allocations.
    
    The RL agent outputs weights in [0, 1] range that sum to 1.0.
    This function multiplies them by investable_capital to get
    actual currency amounts.
    
    Args:
        weights: Relative portfolio weights from RL agent
        capital_config: User's capital configuration
        tickers: Optional list of ticker symbols (for labeling)
        normalize: If True, normalize weights to sum to 1.0
        
    Returns:
        AllocationResult with weights and currency allocations
    """
    weights = np.asarray(weights, dtype=np.float64)
    
    if normalize:
        weights = normalize_weights(weights)
    
    investable = capital_config.investable_capital
    allocations = weights * investable
    
    if tickers is None:
        tickers = [f"Asset_{i}" for i in range(len(weights))]
    
    return AllocationResult(
        tickers=tickers,
        weights=weights,
        allocations=allocations,
        investable_capital=investable,
        capital_config=capital_config,
    )


def allocations_to_weights(
    allocations: Union[np.ndarray, List[float], Dict[str, float]],
    capital_config: Optional[CapitalConfig] = None,
) -> np.ndarray:
    """Convert currency allocations back to relative weights.
    
    Useful for converting user's current holdings to weights
    that can be compared with RL-optimized weights.
    
    Args:
        allocations: Currency amounts per asset (array or dict)
        capital_config: If provided, uses investable_capital as denominator;
                       otherwise uses sum of allocations
                       
    Returns:
        Normalized weights array
    """
    if isinstance(allocations, dict):
        allocations = np.array(list(allocations.values()))
    else:
        allocations = np.asarray(allocations, dtype=np.float64)
    
    if capital_config is not None:
        total = capital_config.investable_capital
    else:
        total = allocations.sum()
    
    if total <= 0:
        return np.ones(len(allocations)) / len(allocations)
    
    return allocations / total


@dataclass
class PortfolioComparison:
    """Comparison between current and optimized portfolio allocations."""
    tickers: List[str]
    current_weights: np.ndarray
    optimized_weights: np.ndarray
    current_allocations: np.ndarray
    optimized_allocations: np.ndarray
    weight_deltas: np.ndarray
    allocation_deltas: np.ndarray
    capital_config: CapitalConfig
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        return pd.DataFrame({
            "Ticker": self.tickers,
            "Current_Weight": self.current_weights,
            "Optimized_Weight": self.optimized_weights,
            "Weight_Delta": self.weight_deltas,
            "Current_Allocation": self.current_allocations,
            "Optimized_Allocation": self.optimized_allocations,
            "Allocation_Delta": self.allocation_deltas,
        })
    
    def summary(self) -> str:
        """Generate human-readable comparison summary."""
        lines = [
            f"Portfolio Comparison",
            f"=" * 70,
            f"Investable Capital: ${self.capital_config.investable_capital:,.2f}",
            f"",
            f"{'Ticker':<8} {'Curr Wt':>8} {'Opt Wt':>8} {'Δ Wt':>8} {'Curr $':>12} {'Opt $':>12} {'Δ $':>12}",
            f"-" * 70,
        ]
        for i, ticker in enumerate(self.tickers):
            lines.append(
                f"{ticker:<8} "
                f"{self.current_weights[i]:>7.1%} "
                f"{self.optimized_weights[i]:>7.1%} "
                f"{self.weight_deltas[i]:>+7.1%} "
                f"${self.current_allocations[i]:>11,.0f} "
                f"${self.optimized_allocations[i]:>11,.0f} "
                f"${self.allocation_deltas[i]:>+11,.0f}"
            )
        lines.append(f"-" * 70)
        
        # Totals
        lines.append(
            f"{'Total':<8} "
            f"{np.sum(self.current_weights):>7.1%} "
            f"{np.sum(self.optimized_weights):>7.1%} "
            f"{'':>8} "
            f"${np.sum(self.current_allocations):>11,.0f} "
            f"${np.sum(self.optimized_allocations):>11,.0f} "
            f"${np.sum(self.allocation_deltas):>+11,.0f}"
        )
        return "\n".join(lines)
    
    def get_buys(self, min_delta: float = 0.0) -> pd.DataFrame:
        """Get assets that should be bought (positive allocation delta)."""
        df = self.to_dataframe()
        return df[df["Allocation_Delta"] > min_delta].sort_values(
            "Allocation_Delta", ascending=False
        )
    
    def get_sells(self, min_delta: float = 0.0) -> pd.DataFrame:
        """Get assets that should be sold (negative allocation delta)."""
        df = self.to_dataframe()
        return df[df["Allocation_Delta"] < -min_delta].sort_values(
            "Allocation_Delta", ascending=True
        )


def compare_allocations(
    current_weights: Union[np.ndarray, List[float]],
    optimized_weights: Union[np.ndarray, List[float]],
    capital_config: CapitalConfig,
    tickers: Optional[List[str]] = None,
) -> PortfolioComparison:
    """Compare current portfolio to optimized portfolio.
    
    Both inputs should be relative weights. This function computes
    the weight deltas and scales to currency amounts.
    
    Args:
        current_weights: Current portfolio weights (relative)
        optimized_weights: RL-optimized weights (relative)
        capital_config: User's capital configuration
        tickers: Optional ticker labels
        
    Returns:
        PortfolioComparison with deltas and recommendations
    """
    current_weights = normalize_weights(np.asarray(current_weights))
    optimized_weights = normalize_weights(np.asarray(optimized_weights))
    
    n_assets = len(current_weights)
    if len(optimized_weights) != n_assets:
        raise ValueError("Weight arrays must have same length")
    
    if tickers is None:
        tickers = [f"Asset_{i}" for i in range(n_assets)]
    
    investable = capital_config.investable_capital
    
    current_alloc = current_weights * investable
    optimized_alloc = optimized_weights * investable
    
    weight_deltas = optimized_weights - current_weights
    alloc_deltas = optimized_alloc - current_alloc
    
    return PortfolioComparison(
        tickers=tickers,
        current_weights=current_weights,
        optimized_weights=optimized_weights,
        current_allocations=current_alloc,
        optimized_allocations=optimized_alloc,
        weight_deltas=weight_deltas,
        allocation_deltas=alloc_deltas,
        capital_config=capital_config,
    )


def holdings_to_weights(
    holdings_df: pd.DataFrame,
    prices: Union[pd.Series, Dict[str, float]],
    capital_config: Optional[CapitalConfig] = None,
) -> AllocationResult:
    """Convert holdings (shares/quantities) to weights using current prices.
    
    Args:
        holdings_df: DataFrame with 'Ticker' and 'Quantity' columns
        prices: Current prices per ticker (Series or dict)
        capital_config: If provided, computes weights relative to investable_capital;
                       otherwise uses total holdings value
                       
    Returns:
        AllocationResult with weights and allocations
    """
    if isinstance(prices, dict):
        prices = pd.Series(prices)
    
    tickers = holdings_df["Ticker"].tolist()
    quantities = holdings_df["Quantity"].values
    
    # Get prices for each ticker
    ticker_prices = np.array([prices.get(t, 0.0) for t in tickers])
    
    # Compute market values
    allocations = quantities * ticker_prices
    
    # Compute weights
    if capital_config is not None:
        total = capital_config.investable_capital
    else:
        total = allocations.sum()
        # Create a synthetic capital config
        capital_config = CapitalConfig(total_capital=total, reserved_cash=0.0)
    
    weights = allocations / total if total > 0 else np.ones(len(allocations)) / len(allocations)
    
    return AllocationResult(
        tickers=tickers,
        weights=weights,
        allocations=allocations,
        investable_capital=capital_config.investable_capital,
        capital_config=capital_config,
    )


if __name__ == "__main__":
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert RL weights to currency allocations")
    parser.add_argument("--total-capital", type=float, default=100000,
                        help="Total capital available")
    parser.add_argument("--reserved-cash", type=float, default=10000,
                        help="Amount to keep as cash reserve")
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[0.3, 0.25, 0.2, 0.15, 0.1],
                        help="Portfolio weights (will be normalized)")
    parser.add_argument("--tickers", type=str, nargs="+",
                        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        help="Ticker symbols")
    args = parser.parse_args()
    
    # Create capital config
    capital = CapitalConfig(
        total_capital=args.total_capital,
        reserved_cash=args.reserved_cash,
    )
    
    print(f"Capital Configuration:")
    print(f"  Total Capital:      ${capital.total_capital:,.2f}")
    print(f"  Reserved Cash:      ${capital.reserved_cash:,.2f}")
    print(f"  Investable Capital: ${capital.investable_capital:,.2f}")
    print()
    
    # Convert weights to allocations
    result = weights_to_allocations(
        weights=args.weights,
        capital_config=capital,
        tickers=args.tickers,
    )
    
    print(result.summary())
    print()
    
    # Demo comparison with different optimized weights
    print("\n" + "=" * 70)
    print("Example: Comparing current vs optimized allocation")
    print("=" * 70 + "\n")
    
    current = [0.3, 0.25, 0.2, 0.15, 0.1]
    optimized = [0.25, 0.30, 0.20, 0.10, 0.15]  # RL suggested different weights
    
    comparison = compare_allocations(
        current_weights=current,
        optimized_weights=optimized,
        capital_config=capital,
        tickers=args.tickers,
    )
    
    print(comparison.summary())
    print("\nBuy recommendations:")
    print(comparison.get_buys(min_delta=100))
    print("\nSell recommendations:")
    print(comparison.get_sells(min_delta=100))
