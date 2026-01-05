"""Recommendation Engine — Combine signals into exposure scores.

Combines multiple asset-level signals into a composite score for each asset,
indicating whether exposure should be "increased" or "decreased".

Signals used:
 - momentum: positive momentum -> increase exposure
 - risk-adjusted return (Sharpe-like): high return/vol -> increase
 - diversification benefit: low correlation with portfolio -> increase
 - drawdown: recent drawdown -> cautious / decrease
 - volatility contribution: high vol contribution -> decrease

The composite score is a weighted average of normalized signal scores,
producing a value in [-1, +1] where:
 - Positive score: recommend increasing exposure
 - Negative score: recommend decreasing exposure
 - Magnitude indicates confidence/strength of recommendation

Usage:
  python scripts/recommendation_engine.py --price-csv data/price_matrix.csv --out-dir simulations/recommendations
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Import asset metrics computation
import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_asset_metrics import compute_asset_metrics


# Default signal weights for the composite score
DEFAULT_SIGNAL_WEIGHTS = {
    "momentum": 0.30,
    "risk_adjusted_return": 0.25,
    "diversification": 0.20,
    "drawdown": 0.15,
    "vol_contribution": 0.10,
}


def normalize_signal(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normalize a signal to [-1, +1] using robust z-score normalization.
    
    Uses median and MAD (median absolute deviation) for robustness to outliers.
    
    Args:
        values: Raw signal values
        higher_is_better: If True, higher raw values produce positive scores
        
    Returns:
        Normalized scores in [-1, +1]
    """
    if values.isna().all() or len(values) == 0:
        return pd.Series(0.0, index=values.index)
    
    median = values.median()
    mad = np.abs(values - median).median()
    
    if mad == 0 or np.isnan(mad):
        # Fallback to standard z-score if MAD is 0
        std = values.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=values.index)
        z = (values - values.mean()) / std
    else:
        # Robust z-score using MAD (scaled to be comparable to std)
        z = (values - median) / (mad * 1.4826)
    
    # Clip to [-3, +3] then scale to [-1, +1]
    z = z.clip(-3, 3) / 3
    
    if not higher_is_better:
        z = -z
    
    return z


def compute_signal_scores(metrics_df: pd.DataFrame, momentum_col: str = "momentum_20") -> pd.DataFrame:
    """Compute individual signal scores from asset metrics.
    
    Args:
        metrics_df: DataFrame with asset metrics (from compute_asset_metrics)
        momentum_col: Name of the momentum column (default: momentum_20)
        
    Returns:
        DataFrame with normalized signal scores for each asset
    """
    scores = pd.DataFrame(index=metrics_df.index)
    
    # 1. Momentum score: positive momentum is good
    if momentum_col in metrics_df.columns:
        scores["momentum"] = normalize_signal(metrics_df[momentum_col], higher_is_better=True)
    else:
        # Try to find any momentum column
        mom_cols = [c for c in metrics_df.columns if "momentum" in c.lower()]
        if mom_cols:
            scores["momentum"] = normalize_signal(metrics_df[mom_cols[0]], higher_is_better=True)
        else:
            scores["momentum"] = 0.0
    
    # 2. Risk-adjusted return (Sharpe-like): annual_return / annual_vol
    if "annual_return" in metrics_df.columns and "annual_vol" in metrics_df.columns:
        vol = metrics_df["annual_vol"].replace(0, np.nan)
        sharpe_like = metrics_df["annual_return"] / vol
        scores["risk_adjusted_return"] = normalize_signal(sharpe_like.fillna(0), higher_is_better=True)
    else:
        scores["risk_adjusted_return"] = 0.0
    
    # 3. Diversification benefit: lower correlation with portfolio is better
    if "corr_with_portfolio" in metrics_df.columns:
        scores["diversification"] = normalize_signal(
            metrics_df["corr_with_portfolio"], 
            higher_is_better=False  # Lower correlation = higher diversification benefit
        )
    else:
        scores["diversification"] = 0.0
    
    # 4. Drawdown penalty: larger drawdown is bad
    if "max_drawdown" in metrics_df.columns:
        scores["drawdown"] = normalize_signal(
            metrics_df["max_drawdown"].abs(),  # Drawdown is typically negative
            higher_is_better=False  # Lower (less negative) drawdown is better
        )
    else:
        scores["drawdown"] = 0.0
    
    # 5. Volatility contribution: high contribution to portfolio vol is bad
    if "pct_contribution_variance" in metrics_df.columns:
        scores["vol_contribution"] = normalize_signal(
            metrics_df["pct_contribution_variance"],
            higher_is_better=False  # Lower vol contribution is better
        )
    elif "contribution_vol" in metrics_df.columns:
        scores["vol_contribution"] = normalize_signal(
            metrics_df["contribution_vol"],
            higher_is_better=False
        )
    else:
        scores["vol_contribution"] = 0.0
    
    return scores


def compute_composite_score(
    signal_scores: pd.DataFrame,
    weights: Dict[str, float] | None = None
) -> pd.Series:
    """Compute weighted composite score from individual signal scores.
    
    Args:
        signal_scores: DataFrame with normalized signal scores
        weights: Dict mapping signal name to weight (must sum to 1)
        
    Returns:
        Series of composite scores in [-1, +1]
    """
    if weights is None:
        weights = DEFAULT_SIGNAL_WEIGHTS.copy()
    
    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    composite = pd.Series(0.0, index=signal_scores.index)
    
    for signal_name, weight in weights.items():
        if signal_name in signal_scores.columns:
            composite += weight * signal_scores[signal_name].fillna(0)
    
    # Ensure output is in [-1, +1]
    return composite.clip(-1, 1)


def generate_recommendations(
    price_df: pd.DataFrame,
    weights: pd.Series | None = None,
    signal_weights: Dict[str, float] | None = None,
    periods_per_year: int = 252,
    momentum_window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate asset recommendations combining multiple signals.
    
    Args:
        price_df: Price matrix (dates x tickers)
        weights: Optional portfolio weights per asset
        signal_weights: Optional weights for each signal in composite
        periods_per_year: Trading days per year for annualization
        momentum_window: Window for momentum calculation
        
    Returns:
        Tuple of (recommendations_df, signal_scores_df)
        - recommendations_df has columns: composite_score, recommendation, raw metrics
        - signal_scores_df has normalized signal scores
    """
    # Compute base asset metrics
    metrics = compute_asset_metrics(
        price_df, 
        weights=weights,
        periods_per_year=periods_per_year,
        window=momentum_window
    )
    
    # Find the momentum column name
    momentum_col = f"momentum_{momentum_window}"
    
    # Compute normalized signal scores
    signal_scores = compute_signal_scores(metrics, momentum_col=momentum_col)
    
    # Compute composite score
    composite = compute_composite_score(signal_scores, weights=signal_weights)
    
    # Build recommendations DataFrame
    recs = metrics.copy()
    recs["composite_score"] = composite
    
    # Generate recommendation labels
    def score_to_recommendation(score: float) -> str:
        if score >= 0.3:
            return "STRONG_INCREASE"
        elif score >= 0.1:
            return "INCREASE"
        elif score <= -0.3:
            return "STRONG_DECREASE"
        elif score <= -0.1:
            return "DECREASE"
        else:
            return "HOLD"
    
    recs["recommendation"] = composite.apply(score_to_recommendation)
    
    # Add signal scores to recommendations for transparency
    for col in signal_scores.columns:
        recs[f"signal_{col}"] = signal_scores[col]
    
    return recs, signal_scores


def get_ranked_lists(
    recommendations: pd.DataFrame,
    top_n: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract ranked lists of assets to increase and decrease exposure.
    
    Args:
        recommendations: DataFrame from generate_recommendations
        top_n: Number of top assets to return in each list
        
    Returns:
        Tuple of (increase_list, decrease_list) DataFrames
    """
    # Sort by composite score
    sorted_recs = recommendations.sort_values("composite_score", ascending=False)
    
    # Top N to increase (highest positive scores)
    increase = sorted_recs[sorted_recs["composite_score"] > 0].head(top_n)
    
    # Top N to decrease (most negative scores)
    decrease = sorted_recs[sorted_recs["composite_score"] < 0].tail(top_n).sort_values("composite_score")
    
    return increase, decrease


def main():
    parser = argparse.ArgumentParser(description="Generate asset exposure recommendations")
    parser.add_argument("--price-csv", required=True, help="Wide CSV with index=Date and columns=tickers")
    parser.add_argument("--out-dir", default="simulations/recommendations", help="Directory to save output")
    parser.add_argument("--weights-csv", default=None, help="Optional CSV with portfolio weights")
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=5, help="Number of top assets in ranked lists")
    
    # Signal weight overrides
    parser.add_argument("--w-momentum", type=float, default=0.25, help="Weight for momentum signal")
    parser.add_argument("--w-sharpe", type=float, default=0.25, help="Weight for risk-adjusted return signal")
    parser.add_argument("--w-diversification", type=float, default=0.20, help="Weight for diversification signal")
    parser.add_argument("--w-drawdown", type=float, default=0.15, help="Weight for drawdown signal")
    parser.add_argument("--w-vol", type=float, default=0.15, help="Weight for volatility contribution signal")
    
    args = parser.parse_args()
    
    # Load price data
    price_df = pd.read_csv(args.price_csv, index_col=0, parse_dates=True)
    
    # Load weights if provided
    weights = None
    if args.weights_csv:
        wdf = pd.read_csv(args.weights_csv)
        if "ticker" in wdf.columns.str.lower():
            # Find ticker and weight columns case-insensitively
            ticker_col = [c for c in wdf.columns if c.lower() == "ticker"][0]
            weight_col = [c for c in wdf.columns if c.lower() == "weight"][0]
            weights = pd.Series(wdf[weight_col].values, index=wdf[ticker_col].values)
        else:
            # Assume two-column headerless format
            weights = pd.Series(wdf.iloc[:, 1].values, index=wdf.iloc[:, 0].values)
    
    # Build signal weights dict
    signal_weights = {
        "momentum": args.w_momentum,
        "risk_adjusted_return": args.w_sharpe,
        "diversification": args.w_diversification,
        "drawdown": args.w_drawdown,
        "vol_contribution": args.w_vol,
    }
    
    # Generate recommendations
    recs, signal_scores = generate_recommendations(
        price_df,
        weights=weights,
        signal_weights=signal_weights,
        periods_per_year=args.periods_per_year,
        momentum_window=args.momentum_window,
    )
    
    # Get ranked lists
    increase_list, decrease_list = get_ranked_lists(recs, top_n=args.top_n)
    
    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    recs.to_csv(out_dir / "recommendations.csv")
    signal_scores.to_csv(out_dir / "signal_scores.csv")
    increase_list.to_csv(out_dir / "increase_exposure.csv")
    decrease_list.to_csv(out_dir / "decrease_exposure.csv")
    
    # Print summary
    print("=" * 60)
    print("ASSET EXPOSURE RECOMMENDATIONS")
    print("=" * 60)
    print(f"\nSignal weights used:")
    for k, v in signal_weights.items():
        print(f"  {k}: {v:.2f}")
    
    print(f"\n{'='*60}")
    print("TOP ASSETS TO INCREASE EXPOSURE")
    print("=" * 60)
    if len(increase_list) > 0:
        for ticker in increase_list.index:
            score = increase_list.loc[ticker, "composite_score"]
            rec = increase_list.loc[ticker, "recommendation"]
            print(f"  {ticker}: score={score:.3f} ({rec})")
    else:
        print("  No assets recommended for increased exposure")
    
    print(f"\n{'='*60}")
    print("TOP ASSETS TO DECREASE EXPOSURE")
    print("=" * 60)
    if len(decrease_list) > 0:
        for ticker in decrease_list.index:
            score = decrease_list.loc[ticker, "composite_score"]
            rec = decrease_list.loc[ticker, "recommendation"]
            print(f"  {ticker}: score={score:.3f} ({rec})")
    else:
        print("  No assets recommended for decreased exposure")
    
    print(f"\nFull recommendations saved to: {out_dir / 'recommendations.csv'}")
    print(f"Signal scores saved to: {out_dir / 'signal_scores.csv'}")


if __name__ == "__main__":
    main()
