"""Integrated Recommendations — Combine RL weight deltas with asset-level metrics.

This module combines:
 - RL weight deltas (from trade_delta_calculator)
 - Asset-level signal scores (from recommendation_engine)
 
To produce ranked buy/reduce recommendations with short rationales explaining
why each trade is recommended.

The composite ranking considers:
 - Trade delta magnitude (from RL optimization)
 - Asset signal score (momentum, risk-adj return, diversification, etc.)
 - Overall confidence based on alignment of RL and signal recommendations

Usage:
  python scripts/integrated_recommendations.py \\
      --price-csv data/price_matrix.csv \\
      --current-weights "AAPL:0.3,MSFT:0.25,GOOGL:0.25,AMZN:0.2" \\
      --target-weights "AAPL:0.25,MSFT:0.35,GOOGL:0.2,AMZN:0.2" \\
      --total-capital 100000 \\
      --reserved-cash 10000 \\
      --out-dir simulations/integrated_recommendations
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

from input_parser import CapitalConfig
from recommendation_engine import (
    generate_recommendations,
    compute_signal_scores,
    compute_composite_score,
    DEFAULT_SIGNAL_WEIGHTS,
)
from trade_delta_calculator import (
    compute_trade_deltas_from_weights,
    TradeAction,
    TradeDelta,
    TradeDeltas,
)


@dataclass
class IntegratedRecommendation:
    """A single integrated recommendation combining RL and signal analysis."""
    
    ticker: str
    action: str  # "BUY", "SELL", "HOLD"
    
    # Trade details
    delta_value: float
    delta_weight: float
    current_value: float
    target_value: float
    current_weight: float
    target_weight: float
    
    # Signal analysis
    signal_score: float  # Composite signal score [-1, +1]
    signal_recommendation: str  # Signal-based recommendation
    
    # Integrated ranking
    integrated_score: float  # Combined RL + signal score
    rank: int  # 1 = highest priority
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    
    # Rationale components
    rationale: str  # Short text rationale
    signal_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "rank": self.rank,
            "confidence": self.confidence,
            "delta_value": float(self.delta_value),
            "delta_weight": float(self.delta_weight),
            "current_value": float(self.current_value),
            "target_value": float(self.target_value),
            "current_weight": float(self.current_weight),
            "target_weight": float(self.target_weight),
            "signal_score": float(self.signal_score),
            "signal_recommendation": self.signal_recommendation,
            "integrated_score": float(self.integrated_score),
            "rationale": self.rationale,
            "signal_breakdown": {k: float(v) for k, v in self.signal_breakdown.items()},
        }


@dataclass
class IntegratedRecommendations:
    """Collection of integrated recommendations."""
    
    recommendations: List[IntegratedRecommendation]
    capital_config: CapitalConfig
    summary: Dict = field(default_factory=dict)
    
    @property
    def buy_recommendations(self) -> List[IntegratedRecommendation]:
        """Get buy recommendations sorted by rank."""
        return sorted(
            [r for r in self.recommendations if r.action == "BUY"],
            key=lambda r: r.rank
        )
    
    @property
    def sell_recommendations(self) -> List[IntegratedRecommendation]:
        """Get sell recommendations sorted by rank."""
        return sorted(
            [r for r in self.recommendations if r.action == "SELL"],
            key=lambda r: r.rank
        )
    
    @property
    def hold_recommendations(self) -> List[IntegratedRecommendation]:
        """Get hold recommendations."""
        return [r for r in self.recommendations if r.action == "HOLD"]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        if not self.recommendations:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.recommendations])
    
    def generate_report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 80,
            "INTEGRATED PORTFOLIO RECOMMENDATIONS",
            "Combining RL Optimization with Asset Signal Analysis",
            "=" * 80,
            "",
            f"Total Capital:      ${self.capital_config.total_capital:>12,.2f}",
            f"Reserved Cash:      ${self.capital_config.reserved_cash:>12,.2f}",
            f"Investable Capital: ${self.capital_config.investable_capital:>12,.2f}",
            "",
        ]
        
        # Buy recommendations
        buys = self.buy_recommendations
        if buys:
            lines.extend([
                "-" * 80,
                "BUY RECOMMENDATIONS (Ranked by Priority)",
                "-" * 80,
            ])
            for rec in buys:
                lines.extend([
                    f"\n  #{rec.rank} {rec.ticker} — {rec.confidence} Confidence",
                    f"     Action: BUY ${rec.delta_value:,.2f} ({rec.delta_weight:+.2%} weight)",
                    f"     Current: ${rec.current_value:,.2f} ({rec.current_weight:.2%}) → "
                    f"Target: ${rec.target_value:,.2f} ({rec.target_weight:.2%})",
                    f"     Signal Score: {rec.signal_score:+.3f} ({rec.signal_recommendation})",
                    f"     Rationale: {rec.rationale}",
                ])
            lines.append("")
        
        # Sell recommendations
        sells = self.sell_recommendations
        if sells:
            lines.extend([
                "-" * 80,
                "SELL RECOMMENDATIONS (Ranked by Priority)",
                "-" * 80,
            ])
            for rec in sells:
                lines.extend([
                    f"\n  #{rec.rank} {rec.ticker} — {rec.confidence} Confidence",
                    f"     Action: SELL ${abs(rec.delta_value):,.2f} ({rec.delta_weight:+.2%} weight)",
                    f"     Current: ${rec.current_value:,.2f} ({rec.current_weight:.2%}) → "
                    f"Target: ${rec.target_value:,.2f} ({rec.target_weight:.2%})",
                    f"     Signal Score: {rec.signal_score:+.3f} ({rec.signal_recommendation})",
                    f"     Rationale: {rec.rationale}",
                ])
            lines.append("")
        
        # Hold positions
        holds = self.hold_recommendations
        if holds:
            lines.extend([
                "-" * 80,
                "HOLD POSITIONS (No Change Recommended)",
                "-" * 80,
            ])
            for rec in holds:
                lines.append(
                    f"  {rec.ticker}: ${rec.current_value:,.2f} ({rec.current_weight:.2%}) "
                    f"— Signal: {rec.signal_score:+.3f}"
                )
            lines.append("")
        
        lines.extend([
            "=" * 80,
            "END OF RECOMMENDATIONS",
            "=" * 80,
        ])
        
        return "\n".join(lines)


def generate_rationale(
    ticker: str,
    action: str,
    delta_weight: float,
    signal_score: float,
    signal_breakdown: Dict[str, float],
    confidence: str,
) -> str:
    """Generate a short text rationale for a recommendation.
    
    Args:
        ticker: Asset ticker
        action: "BUY", "SELL", or "HOLD"
        delta_weight: Weight change
        signal_score: Composite signal score
        signal_breakdown: Individual signal scores
        confidence: Confidence level
        
    Returns:
        Short rationale string
    """
    parts = []
    
    # Main action reason based on RL
    if action == "BUY":
        parts.append(f"RL optimization suggests increasing {ticker} by {delta_weight:+.1%}")
    elif action == "SELL":
        parts.append(f"RL optimization suggests reducing {ticker} by {abs(delta_weight):.1%}")
    else:
        parts.append(f"RL optimization suggests maintaining current {ticker} position")
    
    # Signal alignment
    signal_direction = "positive" if signal_score > 0.1 else ("negative" if signal_score < -0.1 else "neutral")
    
    if action == "BUY" and signal_score > 0.1:
        parts.append("supported by favorable signals")
    elif action == "BUY" and signal_score < -0.1:
        parts.append("despite mixed signals (proceed with caution)")
    elif action == "SELL" and signal_score < -0.1:
        parts.append("confirmed by unfavorable signals")
    elif action == "SELL" and signal_score > 0.1:
        parts.append("overriding positive signals for rebalancing")
    
    # Top contributing signals
    if signal_breakdown:
        sorted_signals = sorted(
            signal_breakdown.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_signals = sorted_signals[:2]
        
        signal_reasons = []
        for signal_name, score in top_signals:
            if abs(score) > 0.15:
                direction = "strong" if abs(score) > 0.5 else "moderate"
                polarity = "positive" if score > 0 else "negative"
                # Format signal name for display
                display_name = signal_name.replace("_", " ")
                signal_reasons.append(f"{direction} {polarity} {display_name}")
        
        if signal_reasons:
            parts.append(f"({', '.join(signal_reasons)})")
    
    return ". ".join(parts) + "."


def compute_integrated_score(
    delta_weight: float,
    signal_score: float,
    action: str,
    rl_weight: float = 0.6,
    signal_weight: float = 0.4,
) -> Tuple[float, str]:
    """Compute integrated score combining RL delta and signal score.
    
    The integrated score prioritizes trades that are both:
    - Larger in magnitude (from RL optimization)
    - Supported by signal analysis
    
    Args:
        delta_weight: Weight change from RL (-1 to +1)
        signal_score: Composite signal score (-1 to +1)
        action: Trade action ("BUY", "SELL", "HOLD")
        rl_weight: Weight for RL component
        signal_weight: Weight for signal component
        
    Returns:
        Tuple of (integrated_score, confidence_level)
    """
    if action == "HOLD":
        return 0.0, "LOW"
    
    # Normalize delta weight to [-1, +1] scale
    # Assume typical rebalancing deltas are within ±20%
    normalized_delta = np.clip(delta_weight / 0.20, -1, 1)
    
    # For buys, positive signal is confirming
    # For sells, negative signal is confirming
    if action == "BUY":
        # Higher delta and higher signal = higher score
        rl_component = abs(normalized_delta)
        signal_alignment = (signal_score + 1) / 2  # Convert to [0, 1]
    else:  # SELL
        # More negative delta and more negative signal = higher score
        rl_component = abs(normalized_delta)
        signal_alignment = (-signal_score + 1) / 2  # Convert to [0, 1], inverted
    
    integrated = rl_weight * rl_component + signal_weight * signal_alignment
    
    # Determine confidence based on alignment
    if action == "BUY":
        aligned = signal_score > 0
    else:
        aligned = signal_score < 0
    
    if aligned and abs(signal_score) > 0.3:
        confidence = "HIGH"
    elif aligned or abs(signal_score) < 0.1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return integrated, confidence


def generate_integrated_recommendations(
    price_df: pd.DataFrame,
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    capital_config: CapitalConfig,
    signal_weights: Optional[Dict[str, float]] = None,
    periods_per_year: int = 252,
    momentum_window: int = 20,
    min_trade_value: float = 100.0,
    min_trade_pct: float = 0.005,
) -> IntegratedRecommendations:
    """Generate integrated recommendations combining RL and signals.
    
    Args:
        price_df: Price matrix (dates x tickers)
        current_weights: Current portfolio weights dict {ticker: weight}
        target_weights: Target/optimized weights dict {ticker: weight}
        capital_config: Capital configuration
        signal_weights: Optional weights for signal components
        periods_per_year: Trading days per year
        momentum_window: Window for momentum calculation
        min_trade_value: Minimum trade value threshold
        min_trade_pct: Minimum trade as percentage of capital
        
    Returns:
        IntegratedRecommendations with ranked buy/sell lists and rationales
    """
    # Get tickers from price data
    tickers = list(price_df.columns)
    
    # Convert weight dicts to arrays, filling missing with 0
    current_w = np.array([current_weights.get(t, 0.0) for t in tickers])
    target_w = np.array([target_weights.get(t, 0.0) for t in tickers])
    
    # Normalize weights
    if current_w.sum() > 0:
        current_w = current_w / current_w.sum()
    if target_w.sum() > 0:
        target_w = target_w / target_w.sum()
    
    # Generate signal-based recommendations
    current_weights_series = pd.Series(current_w, index=tickers)
    recs_df, signal_scores_df = generate_recommendations(
        price_df,
        weights=current_weights_series,
        signal_weights=signal_weights,
        periods_per_year=periods_per_year,
        momentum_window=momentum_window,
    )
    
    # Compute trade deltas
    trade_deltas = compute_trade_deltas_from_weights(
        tickers=tickers,
        current_weights=current_w,
        target_weights=target_w,
        capital_config=capital_config,
        min_trade_value=min_trade_value,
        min_trade_pct=min_trade_pct,
    )
    
    # Combine into integrated recommendations
    integrated_recs = []
    
    for trade in trade_deltas.trades:
        ticker = trade.ticker
        
        if ticker not in recs_df.index:
            continue
        
        # Get signal data
        signal_score = recs_df.loc[ticker, "composite_score"]
        signal_rec = recs_df.loc[ticker, "recommendation"]
        
        # Get signal breakdown
        signal_breakdown = {}
        for col in signal_scores_df.columns:
            if ticker in signal_scores_df.index:
                signal_breakdown[col] = signal_scores_df.loc[ticker, col]
        
        # Determine action
        if trade.action == TradeAction.BUY:
            action = "BUY"
        elif trade.action == TradeAction.SELL:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Skip filtered trades but include HOLD
        if trade.is_filtered and action != "HOLD":
            continue
        
        # Compute integrated score
        integrated_score, confidence = compute_integrated_score(
            delta_weight=trade.delta_weight,
            signal_score=signal_score,
            action=action,
        )
        
        # Generate rationale
        rationale = generate_rationale(
            ticker=ticker,
            action=action,
            delta_weight=trade.delta_weight,
            signal_score=signal_score,
            signal_breakdown=signal_breakdown,
            confidence=confidence,
        )
        
        rec = IntegratedRecommendation(
            ticker=ticker,
            action=action,
            delta_value=trade.delta_value,
            delta_weight=trade.delta_weight,
            current_value=trade.current_value,
            target_value=trade.target_value,
            current_weight=trade.current_weight,
            target_weight=trade.target_weight,
            signal_score=signal_score,
            signal_recommendation=signal_rec,
            integrated_score=integrated_score,
            rank=0,  # Will be assigned later
            confidence=confidence,
            rationale=rationale,
            signal_breakdown=signal_breakdown,
        )
        integrated_recs.append(rec)
    
    # Rank buy recommendations (highest integrated score first)
    buys = [r for r in integrated_recs if r.action == "BUY"]
    buys.sort(key=lambda r: r.integrated_score, reverse=True)
    for i, rec in enumerate(buys, 1):
        rec.rank = i
    
    # Rank sell recommendations (highest integrated score first)
    sells = [r for r in integrated_recs if r.action == "SELL"]
    sells.sort(key=lambda r: r.integrated_score, reverse=True)
    for i, rec in enumerate(sells, 1):
        rec.rank = i
    
    # Holds don't need ranking
    holds = [r for r in integrated_recs if r.action == "HOLD"]
    for rec in holds:
        rec.rank = 0
    
    # Compute summary
    summary = {
        "total_buys": len(buys),
        "total_sells": len(sells),
        "total_holds": len(holds),
        "high_confidence_count": sum(1 for r in integrated_recs if r.confidence == "HIGH"),
        "total_buy_value": sum(r.delta_value for r in buys),
        "total_sell_value": sum(abs(r.delta_value) for r in sells),
    }
    
    return IntegratedRecommendations(
        recommendations=integrated_recs,
        capital_config=capital_config,
        summary=summary,
    )


def save_integrated_recommendations(
    recs: IntegratedRecommendations,
    out_dir: str,
    prefix: str = "integrated_recs",
) -> Dict[str, str]:
    """Save integrated recommendations to files.
    
    Args:
        recs: IntegratedRecommendations to save
        out_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dictionary of output file paths
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save full recommendations CSV
    csv_path = out_path / f"{prefix}.csv"
    recs.to_dataframe().to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)
    
    # Save buy recommendations
    buys_df = pd.DataFrame([r.to_dict() for r in recs.buy_recommendations])
    if not buys_df.empty:
        buys_path = out_path / f"{prefix}_buy.csv"
        buys_df.to_csv(buys_path, index=False)
        paths["buy_csv"] = str(buys_path)
    
    # Save sell recommendations
    sells_df = pd.DataFrame([r.to_dict() for r in recs.sell_recommendations])
    if not sells_df.empty:
        sells_path = out_path / f"{prefix}_sell.csv"
        sells_df.to_csv(sells_path, index=False)
        paths["sell_csv"] = str(sells_path)
    
    # Save JSON summary
    json_data = {
        "summary": recs.summary,
        "capital_config": {
            "total_capital": recs.capital_config.total_capital,
            "reserved_cash": recs.capital_config.reserved_cash,
            "investable_capital": recs.capital_config.investable_capital,
        },
        "buy_recommendations": [r.to_dict() for r in recs.buy_recommendations],
        "sell_recommendations": [r.to_dict() for r in recs.sell_recommendations],
        "hold_positions": [r.to_dict() for r in recs.hold_recommendations],
    }
    json_path = out_path / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    paths["json"] = str(json_path)
    
    # Save text report
    report_path = out_path / f"{prefix}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(recs.generate_report())
    paths["report"] = str(report_path)
    
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
        description="Generate integrated recommendations combining RL and signal analysis"
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
    parser.add_argument("--total-capital", type=float, default=100000)
    parser.add_argument("--reserved-cash", type=float, default=10000)
    parser.add_argument("--out-dir", default="simulations/integrated_recommendations")
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--min-trade-value", type=float, default=100)
    parser.add_argument("--min-trade-pct", type=float, default=0.005)
    
    # Signal weight overrides
    parser.add_argument("--w-momentum", type=float, default=0.25)
    parser.add_argument("--w-sharpe", type=float, default=0.25)
    parser.add_argument("--w-diversification", type=float, default=0.20)
    parser.add_argument("--w-drawdown", type=float, default=0.15)
    parser.add_argument("--w-vol", type=float, default=0.15)
    
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
    
    # Build capital config
    capital_config = CapitalConfig(args.total_capital, args.reserved_cash)
    
    # Build signal weights
    signal_weights = {
        "momentum": args.w_momentum,
        "risk_adjusted_return": args.w_sharpe,
        "diversification": args.w_diversification,
        "drawdown": args.w_drawdown,
        "vol_contribution": args.w_vol,
    }
    
    print(f"Generating integrated recommendations...")
    print(f"  Price data: {args.price_csv} ({len(price_df)} rows, {len(price_df.columns)} assets)")
    print(f"  Total capital: ${capital_config.total_capital:,.2f}")
    print(f"  Reserved cash: ${capital_config.reserved_cash:,.2f}")
    print(f"  Investable: ${capital_config.investable_capital:,.2f}")
    print()
    
    # Generate recommendations
    recs = generate_integrated_recommendations(
        price_df=price_df,
        current_weights=current_weights,
        target_weights=target_weights,
        capital_config=capital_config,
        signal_weights=signal_weights,
        periods_per_year=args.periods_per_year,
        momentum_window=args.momentum_window,
        min_trade_value=args.min_trade_value,
        min_trade_pct=args.min_trade_pct,
    )
    
    # Print report
    print(recs.generate_report())
    
    # Save outputs
    paths = save_integrated_recommendations(recs, args.out_dir)
    print(f"\nOutputs saved to {args.out_dir}:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
