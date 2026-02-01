"""
Trade Delta Calculator Module

Computes suggested buy/sell deltas per asset, respecting:
- Reserved cash constraints
- Transaction costs
- Tiny-delta filtering (below a threshold)
- Optional grouping of small trades
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_parser import CapitalConfig


class TradeAction(Enum):
    """Type of trade action."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeDelta:
    """Represents a single trade recommendation."""
    ticker: str
    action: TradeAction
    current_value: float
    target_value: float
    delta_value: float  # Positive = buy, negative = sell
    delta_shares: Optional[float] = None  # If price provided
    current_weight: float = 0.0
    target_weight: float = 0.0
    delta_weight: float = 0.0
    transaction_cost: float = 0.0
    net_delta: float = 0.0  # delta_value minus transaction cost
    is_filtered: bool = False  # True if below min threshold
    filter_reason: Optional[str] = None
    
    @property
    def abs_delta(self) -> float:
        return abs(self.delta_value)
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "action": self.action.value,
            "current_value": float(self.current_value),
            "target_value": float(self.target_value),
            "delta_value": float(self.delta_value),
            "delta_shares": float(self.delta_shares) if self.delta_shares is not None else None,
            "current_weight": float(self.current_weight),
            "target_weight": float(self.target_weight),
            "delta_weight": float(self.delta_weight),
            "transaction_cost": float(self.transaction_cost),
            "net_delta": float(self.net_delta),
            "is_filtered": bool(self.is_filtered),
            "filter_reason": self.filter_reason,
        }


@dataclass
class TradeDeltas:
    """Collection of trade deltas with summary statistics."""
    trades: List[TradeDelta]
    capital_config: CapitalConfig
    transaction_cost_rate: float
    min_trade_value: float
    min_trade_pct: float
    
    # Computed summaries
    total_buys: float = 0.0
    total_sells: float = 0.0
    total_transaction_costs: float = 0.0
    net_cash_flow: float = 0.0  # Positive = net sell, negative = net buy
    
    def __post_init__(self):
        """Compute summary statistics."""
        self.total_buys = sum(t.delta_value for t in self.trades if t.action == TradeAction.BUY and not t.is_filtered)
        self.total_sells = sum(abs(t.delta_value) for t in self.trades if t.action == TradeAction.SELL and not t.is_filtered)
        self.total_transaction_costs = sum(t.transaction_cost for t in self.trades if not t.is_filtered)
        self.net_cash_flow = self.total_sells - self.total_buys
    
    @property
    def active_trades(self) -> List[TradeDelta]:
        """Trades that are not filtered out."""
        return [t for t in self.trades if not t.is_filtered]
    
    @property
    def filtered_trades(self) -> List[TradeDelta]:
        """Trades that were filtered out."""
        return [t for t in self.trades if t.is_filtered]
    
    @property
    def buy_trades(self) -> List[TradeDelta]:
        """Active buy trades sorted by delta (largest first)."""
        return sorted(
            [t for t in self.active_trades if t.action == TradeAction.BUY],
            key=lambda t: t.delta_value,
            reverse=True
        )
    
    @property
    def sell_trades(self) -> List[TradeDelta]:
        """Active sell trades sorted by delta (largest sell first)."""
        return sorted(
            [t for t in self.active_trades if t.action == TradeAction.SELL],
            key=lambda t: t.delta_value  # More negative = larger sell
        )
    
    @property
    def is_cash_feasible(self) -> bool:
        """Check if trades are feasible given reserved cash constraint."""
        # Net buys should not exceed available cash (sells + uninvested)
        return self.net_cash_flow >= -self.capital_config.reserved_cash
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def active_trades_df(self) -> pd.DataFrame:
        """DataFrame of active (non-filtered) trades."""
        if not self.active_trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.active_trades])
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "TRADE RECOMMENDATIONS",
            "=" * 70,
            f"Transaction Cost Rate: {self.transaction_cost_rate:.2%}",
            f"Min Trade Value:       ${self.min_trade_value:,.2f}",
            f"Min Trade Percent:     {self.min_trade_pct:.2%}",
            "",
            "SUMMARY",
            "-" * 40,
            f"  Total Buys:           ${self.total_buys:>12,.2f}",
            f"  Total Sells:          ${self.total_sells:>12,.2f}",
            f"  Net Cash Flow:        ${self.net_cash_flow:>+12,.2f}",
            f"  Transaction Costs:    ${self.total_transaction_costs:>12,.2f}",
            f"  Active Trades:        {len(self.active_trades):>12}",
            f"  Filtered Trades:      {len(self.filtered_trades):>12}",
            f"  Cash Feasible:        {'Yes' if self.is_cash_feasible else 'NO - EXCEEDS RESERVE':>12}",
            "",
        ]
        
        if self.buy_trades:
            lines.extend([
                "BUY RECOMMENDATIONS",
                "-" * 70,
                f"{'Ticker':<8} {'Current':>12} {'Target':>12} {'Delta':>12} {'Cost':>10}",
                "-" * 70,
            ])
            for t in self.buy_trades:
                lines.append(
                    f"{t.ticker:<8} "
                    f"${t.current_value:>11,.0f} "
                    f"${t.target_value:>11,.0f} "
                    f"${t.delta_value:>+11,.0f} "
                    f"${t.transaction_cost:>9,.2f}"
                )
            lines.append("")
        
        if self.sell_trades:
            lines.extend([
                "SELL RECOMMENDATIONS",
                "-" * 70,
                f"{'Ticker':<8} {'Current':>12} {'Target':>12} {'Delta':>12} {'Cost':>10}",
                "-" * 70,
            ])
            for t in self.sell_trades:
                lines.append(
                    f"{t.ticker:<8} "
                    f"${t.current_value:>11,.0f} "
                    f"${t.target_value:>11,.0f} "
                    f"${t.delta_value:>+11,.0f} "
                    f"${t.transaction_cost:>9,.2f}"
                )
            lines.append("")
        
        if self.filtered_trades:
            lines.extend([
                "FILTERED (Below Threshold)",
                "-" * 70,
            ])
            for t in self.filtered_trades:
                lines.append(f"  {t.ticker}: ${t.delta_value:+,.0f} - {t.filter_reason}")
        
        return "\n".join(lines)


def compute_trade_deltas(
    tickers: List[str],
    current_values: np.ndarray,
    target_values: np.ndarray,
    capital_config: CapitalConfig,
    prices: Optional[Dict[str, float]] = None,
    transaction_cost_rate: float = 0.001,
    min_trade_value: float = 100.0,
    min_trade_pct: float = 0.005,
) -> TradeDeltas:
    """Compute trade deltas with filtering and transaction costs.
    
    Args:
        tickers: List of ticker symbols
        current_values: Current allocation values per asset
        target_values: Target allocation values per asset
        capital_config: Capital configuration
        prices: Optional current prices (for share calculations)
        transaction_cost_rate: Transaction cost as fraction of trade value
        min_trade_value: Minimum absolute trade value to execute
        min_trade_pct: Minimum trade as percentage of investable capital
        
    Returns:
        TradeDeltas with all trade recommendations
    """
    current_values = np.asarray(current_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    
    investable = capital_config.investable_capital
    min_pct_value = min_trade_pct * investable
    effective_min = max(min_trade_value, min_pct_value)
    
    # Compute weights
    total_current = current_values.sum()
    total_target = target_values.sum()
    
    current_weights = current_values / total_current if total_current > 0 else np.zeros(len(current_values))
    target_weights = target_values / total_target if total_target > 0 else np.zeros(len(target_values))
    
    trades = []
    
    for i, ticker in enumerate(tickers):
        delta = target_values[i] - current_values[i]
        delta_weight = target_weights[i] - current_weights[i]
        abs_delta = abs(delta)
        
        # Determine action
        if abs_delta < 1e-6:
            action = TradeAction.HOLD
        elif delta > 0:
            action = TradeAction.BUY
        else:
            action = TradeAction.SELL
        
        # Calculate transaction cost
        tx_cost = abs_delta * transaction_cost_rate
        
        # Calculate shares if price available
        delta_shares = None
        if prices and ticker in prices and prices[ticker] > 0:
            delta_shares = delta / prices[ticker]
        
        # Apply filtering
        is_filtered = False
        filter_reason = None
        
        if action != TradeAction.HOLD:
            if abs_delta < min_trade_value:
                is_filtered = True
                filter_reason = f"Below min value ${min_trade_value:.0f}"
            elif abs_delta < min_pct_value:
                is_filtered = True
                filter_reason = f"Below min pct {min_trade_pct:.1%}"
        
        trade = TradeDelta(
            ticker=ticker,
            action=action,
            current_value=current_values[i],
            target_value=target_values[i],
            delta_value=delta,
            delta_shares=delta_shares,
            current_weight=current_weights[i],
            target_weight=target_weights[i],
            delta_weight=delta_weight,
            transaction_cost=tx_cost if not is_filtered else 0.0,
            net_delta=delta - tx_cost if delta > 0 else delta + tx_cost,
            is_filtered=is_filtered,
            filter_reason=filter_reason,
        )
        trades.append(trade)
    
    return TradeDeltas(
        trades=trades,
        capital_config=capital_config,
        transaction_cost_rate=transaction_cost_rate,
        min_trade_value=min_trade_value,
        min_trade_pct=min_trade_pct,
    )


def compute_trade_deltas_from_weights(
    tickers: List[str],
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    capital_config: CapitalConfig,
    prices: Optional[Dict[str, float]] = None,
    transaction_cost_rate: float = 0.001,
    min_trade_value: float = 100.0,
    min_trade_pct: float = 0.005,
) -> TradeDeltas:
    """Compute trade deltas from portfolio weights.
    
    Args:
        tickers: List of ticker symbols
        current_weights: Current portfolio weights (should sum to ~1)
        target_weights: Target portfolio weights (should sum to ~1)
        capital_config: Capital configuration
        prices: Optional current prices (for share calculations)
        transaction_cost_rate: Transaction cost as fraction of trade value
        min_trade_value: Minimum absolute trade value to execute
        min_trade_pct: Minimum trade as percentage of investable capital
        
    Returns:
        TradeDeltas with all trade recommendations
    """
    current_weights = np.asarray(current_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)
    
    # Normalize weights
    if current_weights.sum() > 0:
        current_weights = current_weights / current_weights.sum()
    if target_weights.sum() > 0:
        target_weights = target_weights / target_weights.sum()
    
    investable = capital_config.investable_capital
    
    # Convert to values
    current_values = current_weights * investable
    target_values = target_weights * investable
    
    return compute_trade_deltas(
        tickers=tickers,
        current_values=current_values,
        target_values=target_values,
        capital_config=capital_config,
        prices=prices,
        transaction_cost_rate=transaction_cost_rate,
        min_trade_value=min_trade_value,
        min_trade_pct=min_trade_pct,
    )


def adjust_for_cash_constraint(
    deltas: TradeDeltas,
    available_cash: float,
) -> TradeDeltas:
    """Adjust trade deltas to respect available cash constraint.
    
    If net buys exceed available cash (from sells + uninvested cash),
    scale down buy orders proportionally.
    
    Args:
        deltas: Original trade deltas
        available_cash: Cash available for net buying
        
    Returns:
        Adjusted TradeDeltas (may scale down buys)
    """
    # Calculate net buy requirement
    total_buys = deltas.total_buys
    total_sells = deltas.total_sells
    net_buy = total_buys - total_sells
    
    if net_buy <= available_cash:
        # No adjustment needed
        return deltas
    
    # Need to scale down buys
    if total_buys <= 0:
        return deltas
    
    # Scale factor to bring net buy within available cash
    # net_buy_new = scale * total_buys - total_sells = available_cash
    # scale = (available_cash + total_sells) / total_buys
    scale = (available_cash + total_sells) / total_buys
    scale = max(0, min(1, scale))  # Clamp to [0, 1]
    
    adjusted_trades = []
    for trade in deltas.trades:
        if trade.action == TradeAction.BUY and not trade.is_filtered:
            # Scale down buy
            new_delta = trade.delta_value * scale
            new_target = trade.current_value + new_delta
            new_tx_cost = abs(new_delta) * deltas.transaction_cost_rate
            
            adjusted = TradeDelta(
                ticker=trade.ticker,
                action=trade.action,
                current_value=trade.current_value,
                target_value=new_target,
                delta_value=new_delta,
                delta_shares=trade.delta_shares * scale if trade.delta_shares else None,
                current_weight=trade.current_weight,
                target_weight=trade.target_weight * scale,
                delta_weight=trade.delta_weight * scale,
                transaction_cost=new_tx_cost,
                net_delta=new_delta - new_tx_cost,
                is_filtered=trade.is_filtered,
                filter_reason=f"Scaled to {scale:.0%} due to cash constraint" if scale < 1 else trade.filter_reason,
            )
            adjusted_trades.append(adjusted)
        else:
            adjusted_trades.append(trade)
    
    return TradeDeltas(
        trades=adjusted_trades,
        capital_config=deltas.capital_config,
        transaction_cost_rate=deltas.transaction_cost_rate,
        min_trade_value=deltas.min_trade_value,
        min_trade_pct=deltas.min_trade_pct,
    )


def group_small_trades(
    deltas: TradeDeltas,
    group_threshold: float = 500.0,
) -> Tuple[TradeDeltas, Dict]:
    """Group small trades into an 'OTHER' bucket.
    
    Trades below group_threshold are combined into a single line item.
    This simplifies execution for portfolios with many small positions.
    
    Args:
        deltas: Original trade deltas
        group_threshold: Trades below this value are grouped
        
    Returns:
        Tuple of (modified TradeDeltas, grouping info dict)
    """
    large_trades = []
    small_buys = []
    small_sells = []
    
    for trade in deltas.trades:
        if trade.is_filtered:
            large_trades.append(trade)
            continue
            
        if trade.abs_delta >= group_threshold:
            large_trades.append(trade)
        elif trade.action == TradeAction.BUY:
            small_buys.append(trade)
        elif trade.action == TradeAction.SELL:
            small_sells.append(trade)
        else:
            large_trades.append(trade)
    
    grouping_info = {
        "grouped_buys": [t.ticker for t in small_buys],
        "grouped_sells": [t.ticker for t in small_sells],
        "total_grouped_buy_value": sum(t.delta_value for t in small_buys),
        "total_grouped_sell_value": sum(t.delta_value for t in small_sells),
    }
    
    # Create grouped trade entries
    if small_buys:
        total_buy = sum(t.delta_value for t in small_buys)
        total_buy_cost = sum(t.transaction_cost for t in small_buys)
        grouped_buy = TradeDelta(
            ticker="OTHER_BUYS",
            action=TradeAction.BUY,
            current_value=sum(t.current_value for t in small_buys),
            target_value=sum(t.target_value for t in small_buys),
            delta_value=total_buy,
            delta_shares=None,
            current_weight=sum(t.current_weight for t in small_buys),
            target_weight=sum(t.target_weight for t in small_buys),
            delta_weight=sum(t.delta_weight for t in small_buys),
            transaction_cost=total_buy_cost,
            net_delta=total_buy - total_buy_cost,
            is_filtered=False,
            filter_reason=f"Grouped {len(small_buys)} small buys",
        )
        large_trades.append(grouped_buy)
    
    if small_sells:
        total_sell = sum(t.delta_value for t in small_sells)
        total_sell_cost = sum(t.transaction_cost for t in small_sells)
        grouped_sell = TradeDelta(
            ticker="OTHER_SELLS",
            action=TradeAction.SELL,
            current_value=sum(t.current_value for t in small_sells),
            target_value=sum(t.target_value for t in small_sells),
            delta_value=total_sell,
            delta_shares=None,
            current_weight=sum(t.current_weight for t in small_sells),
            target_weight=sum(t.target_weight for t in small_sells),
            delta_weight=sum(t.delta_weight for t in small_sells),
            transaction_cost=total_sell_cost,
            net_delta=total_sell + total_sell_cost,
            is_filtered=False,
            filter_reason=f"Grouped {len(small_sells)} small sells",
        )
        large_trades.append(grouped_sell)
    
    result = TradeDeltas(
        trades=large_trades,
        capital_config=deltas.capital_config,
        transaction_cost_rate=deltas.transaction_cost_rate,
        min_trade_value=deltas.min_trade_value,
        min_trade_pct=deltas.min_trade_pct,
    )
    
    return result, grouping_info


def save_trade_deltas(
    deltas: TradeDeltas,
    out_dir: str,
    prefix: str = "trade_deltas",
) -> Dict[str, str]:
    """Save trade deltas to files.
    
    Args:
        deltas: TradeDeltas to save
        out_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dictionary of output file paths
    """
    import json
    os.makedirs(out_dir, exist_ok=True)
    
    paths = {}
    
    # Save all trades CSV
    all_csv = os.path.join(out_dir, f"{prefix}_all.csv")
    deltas.to_dataframe().to_csv(all_csv, index=False)
    paths["all_csv"] = all_csv
    
    # Save active trades CSV
    active_csv = os.path.join(out_dir, f"{prefix}_active.csv")
    active_df = deltas.active_trades_df()
    if not active_df.empty:
        active_df.to_csv(active_csv, index=False)
        paths["active_csv"] = active_csv
    
    # Save summary JSON
    summary = {
        "total_buys": float(deltas.total_buys),
        "total_sells": float(deltas.total_sells),
        "net_cash_flow": float(deltas.net_cash_flow),
        "total_transaction_costs": float(deltas.total_transaction_costs),
        "num_active_trades": len(deltas.active_trades),
        "num_filtered_trades": len(deltas.filtered_trades),
        "is_cash_feasible": bool(deltas.is_cash_feasible),
        "config": {
            "transaction_cost_rate": float(deltas.transaction_cost_rate),
            "min_trade_value": float(deltas.min_trade_value),
            "min_trade_pct": float(deltas.min_trade_pct),
        },
        "trades": [t.to_dict() for t in deltas.active_trades],
    }
    json_path = os.path.join(out_dir, f"{prefix}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    paths["json"] = json_path
    
    # Save text report
    report_path = os.path.join(out_dir, f"{prefix}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(deltas.summary())
    paths["report"] = report_path
    
    return paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute trade deltas with filtering")
    parser.add_argument("--total-capital", type=float, default=100000)
    parser.add_argument("--reserved-cash", type=float, default=10000)
    parser.add_argument("--tx-cost", type=float, default=0.001, help="Transaction cost rate")
    parser.add_argument("--min-value", type=float, default=100, help="Min trade value")
    parser.add_argument("--min-pct", type=float, default=0.005, help="Min trade as pct of capital")
    parser.add_argument("--out-dir", type=str, default="trade_results")
    args = parser.parse_args()
    
    # Demo with sample data
    capital = CapitalConfig(args.total_capital, args.reserved_cash)
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    current_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    target_weights = np.array([0.25, 0.30, 0.20, 0.12, 0.13])
    
    prices = {"AAPL": 150, "MSFT": 300, "GOOGL": 140, "AMZN": 180, "META": 350}
    
    print(f"Capital: ${capital.total_capital:,.0f} (Reserved: ${capital.reserved_cash:,.0f})")
    print(f"Investable: ${capital.investable_capital:,.0f}")
    print()
    
    deltas = compute_trade_deltas_from_weights(
        tickers=tickers,
        current_weights=current_weights,
        target_weights=target_weights,
        capital_config=capital,
        prices=prices,
        transaction_cost_rate=args.tx_cost,
        min_trade_value=args.min_value,
        min_trade_pct=args.min_pct,
    )
    
    print(deltas.summary())
    
    # Save results
    paths = save_trade_deltas(deltas, args.out_dir)
    print(f"\nSaved to: {args.out_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
