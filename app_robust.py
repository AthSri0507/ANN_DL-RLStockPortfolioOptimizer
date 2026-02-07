"""
RL Portfolio Optimizer — Streamlit Dashboard (Robust Inference)
A minimal, modern frontend for RL-based portfolio optimization with robust inference techniques.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Project imports
from input_parser import CapitalConfig
from scripts.compare_portfolio_mc import run_portfolio_comparison
from scripts.integrated_recommendations import generate_integrated_recommendations
from stable_baselines3 import PPO
from portfolio_env import PortfolioEnv

# ============================================================================
# ROBUST INFERENCE CONFIGURATION (Fixed - Not User Editable)
# ============================================================================

ROBUST_CONFIG = {
    # Start-step ensemble: run from multiple start offsets and aggregate
    "use_start_ensemble": True,
    "start_offsets": [-4, -3, -2, -1, 0],  # 5 different starting points
    "ensemble_agg": "median",  # Median is robust to outlier runs
    
    # Last-K smoothed weights: use median of last K steps' weights
    "use_last_k_smoothing": True,
    "last_k": 5,  # ~1 trading week of smoothing
    
    # Cap and floor: limit per-asset allocation
    "use_cap_floor": True,
    "cap": 0.15,   # Max 15% per asset (conservative)
    "floor": 0.02, # Min 2% per asset (ensures diversification)
    
    # Softmax temperature: flatten extreme allocations
    "use_softmax_temp": True,
    "temperature": 2.0,
    
    # Monte Carlo evaluation parameters
    "n_paths": 2000,
    "noise": "block_bootstrap",
    "n_steps": 252,
}

# ============================================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================================
st.set_page_config(
    page_title="RL Portfolio Optimizer (Robust)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for purple theme and modern styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #6c5ce7;
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(108, 92, 231, 0.1);
        border: 1px solid rgba(108, 92, 231, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .metric-card-green {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .metric-card-red {
        background: rgba(255, 71, 87, 0.1);
        border: 1px solid rgba(255, 71, 87, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .robust-card {
        background: rgba(0, 200, 150, 0.1);
        border: 1px solid rgba(0, 200, 150, 0.4);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #a29bfe !important;
    }
    
    /* Text */
    p, span, label {
        color: #dfe6e9 !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: rgba(108, 92, 231, 0.1) !important;
        border: 1px solid rgba(108, 92, 231, 0.3) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 92, 231, 0.4);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #a29bfe !important;
        font-size: 1.8rem !important;
    }
    
    [data-testid="stMetricDelta"] > div {
        color: #00ff88 !important;
    }
    
    /* Tables */
    .dataframe {
        background: rgba(108, 92, 231, 0.05) !important;
        border-radius: 12px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(108, 92, 231, 0.1) !important;
        border-radius: 8px !important;
        color: #a29bfe !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(108, 92, 231, 0.3) !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1) !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
    }
    
    .stError {
        background: rgba(255, 71, 87, 0.1) !important;
        border: 1px solid rgba(255, 71, 87, 0.3) !important;
    }
    
    /* Info box */
    .info-box {
        background: rgba(108, 92, 231, 0.15);
        border-left: 4px solid #6c5ce7;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Hero title */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe, #00c896);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #b2bec3 !important;
        font-size: 1.1rem;
        margin-top: 5px;
    }
    
    .robust-badge {
        background: linear-gradient(135deg, #00c896, #00a896);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ROBUST INFERENCE FUNCTIONS
# ============================================================================

def apply_cap_floor(weights: np.ndarray, cap: float, floor: float) -> np.ndarray:
    """Apply cap and floor to weights, then renormalize."""
    adjusted = weights.copy()
    positive_mask = adjusted > 0
    adjusted[positive_mask] = np.maximum(adjusted[positive_mask], floor)
    adjusted = np.minimum(adjusted, cap)
    
    if adjusted.sum() > 0:
        adjusted = adjusted / adjusted.sum()
    else:
        adjusted = np.ones_like(adjusted) / len(adjusted)
    
    return adjusted


def apply_softmax_temperature(weights: np.ndarray, temperature: float) -> np.ndarray:
    """Apply softmax with temperature to flatten extreme allocations."""
    eps = 1e-8
    logits = np.log(weights + eps)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - scaled_logits.max())
    return exp_logits / exp_logits.sum()


def run_single_rollout(
    price_df: pd.DataFrame,
    model: PPO,
    n_assets: int,
    n_steps: int,
    seed: int,
    start_step: int = None,
    collect_smoothed: bool = True,
) -> tuple:
    """Run a single deterministic rollout and collect weights."""
    env = PortfolioEnv(
        n_assets=n_assets,
        price_df=price_df,
        transaction_cost=0.001,
        reward_alpha=0.0,
        window=20,
        start_step=start_step,
    )
    
    obs, info = env.reset(seed=seed)
    
    smoothed_weights_history = []
    final_action = None
    
    for step_idx in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        final_action = action
        
        if collect_smoothed and "smoothed_weights" in info:
            smoothed_weights_history.append(info["smoothed_weights"].copy())
        else:
            normalized = action / action.sum() if action.sum() > 0 else np.ones_like(action) / len(action)
            smoothed_weights_history.append(normalized)
        
        if terminated or truncated:
            break
    
    return final_action, smoothed_weights_history


def get_robust_weights(
    price_df: pd.DataFrame,
    model: PPO,
    all_tickers: list,
    config: dict,
    seed: int = 42,
) -> np.ndarray:
    """Extract robust RL weights using configured techniques."""
    n_assets = len(all_tickers)
    window = 20
    max_steps = len(price_df) - window - 2
    n_rollout_steps = min(50, max_steps)
    
    all_weight_sets = []
    
    # --- Start-step ensemble ---
    if config.get("use_start_ensemble", False):
        start_offsets = config.get("start_offsets", [0])
        base_start = max(window, max_steps - n_rollout_steps)
        
        for offset in start_offsets:
            start_step = max(window, base_start + offset)
            final_action, smoothed_history = run_single_rollout(
                price_df, model, n_assets, n_rollout_steps, seed, start_step=start_step
            )
            
            if config.get("use_last_k_smoothing", False) and len(smoothed_history) > 0:
                last_k = config.get("last_k", 5)
                last_k_weights = smoothed_history[-last_k:]
                weights = np.median(last_k_weights, axis=0)
            else:
                weights = final_action / final_action.sum() if final_action.sum() > 0 else np.ones(n_assets) / n_assets
            
            all_weight_sets.append(weights)
    else:
        final_action, smoothed_history = run_single_rollout(
            price_df, model, n_assets, n_rollout_steps, seed
        )
        
        if config.get("use_last_k_smoothing", False) and len(smoothed_history) > 0:
            last_k = config.get("last_k", 5)
            last_k_weights = smoothed_history[-last_k:]
            weights = np.median(last_k_weights, axis=0)
        else:
            weights = final_action / final_action.sum() if final_action.sum() > 0 else np.ones(n_assets) / n_assets
        
        all_weight_sets.append(weights)
    
    # --- Aggregate across ensemble ---
    if len(all_weight_sets) > 1:
        agg_method = config.get("ensemble_agg", "median")
        if agg_method == "median":
            weights = np.median(all_weight_sets, axis=0)
        else:
            weights = np.mean(all_weight_sets, axis=0)
    else:
        weights = all_weight_sets[0]
    
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_assets) / n_assets
    
    # --- Apply softmax temperature ---
    if config.get("use_softmax_temp", False):
        temperature = config.get("temperature", 2.0)
        weights = apply_softmax_temperature(weights, temperature)
    
    # --- Apply cap + floor ---
    if config.get("use_cap_floor", False):
        cap = config.get("cap", 0.15)
        floor = config.get("floor", 0.02)
        weights = apply_cap_floor(weights, cap, floor)
    
    return weights


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_rl_model():
    """Load the trained RL model."""
    MODEL_PATH = None
    candidates = []
    models_dir = project_root / 'models'
    archive_dir = project_root / 'logs' / 'archive'
    
    if models_dir.exists():
        candidates += sorted(models_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    if archive_dir.exists():
        candidates += sorted(archive_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    candidates = [p for p in candidates if 'ppo' in p.name.lower() or 'model' in p.name.lower()]
    
    if candidates:
        MODEL_PATH = candidates[0]
        model = PPO.load(MODEL_PATH)
        return model, MODEL_PATH
    return None, None


@st.cache_data
def load_training_data():
    """Load the training price data."""
    RL_PRICE_CSV = project_root / 'data' / 'rl_training_price_data_india.csv'
    if RL_PRICE_CSV.exists():
        return pd.read_csv(RL_PRICE_CSV, index_col=0, parse_dates=True)
    return None


@st.cache_data(ttl=3600)
def fetch_price_data(tickers, start_date, end_date):
    """Fetch price data from Yahoo Finance."""
    try:
        raw_data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        
        if len(tickers) == 1:
            price_df = raw_data[['Close']].copy()
            price_df.columns = tickers
        else:
            price_df = raw_data['Close'].copy()
        
        price_df = price_df.dropna()
        price_df = price_df[tickers]
        return price_df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def get_rl_weights_robust(model, training_df, user_tickers, seed=42):
    """Run RL policy with robust inference to get optimized weights."""
    rl_tickers = list(training_df.columns)
    
    # Get robust weights using ensemble + smoothing + cap/floor + temperature
    robust_weights = get_robust_weights(
        price_df=training_df,
        model=model,
        all_tickers=rl_tickers,
        config=ROBUST_CONFIG,
        seed=seed,
    )
    
    full_weights = {t: float(w) for t, w in zip(rl_tickers, robust_weights)}
    
    # Filter to user's tickers and renormalize
    common_tickers = [t for t in user_tickers if t in rl_tickers]
    if not common_tickers:
        return None, []
    
    common_weights = {t: full_weights[t] for t in common_tickers}
    total = sum(common_weights.values())
    normalized_weights = {t: w / total for t, w in common_weights.items()}
    
    return normalized_weights, common_tickers


def compute_shares_to_trade(current_weights, target_weights, tickers, price_df, investable_capital):
    """Compute integer number of shares to buy/sell."""
    trades = []
    latest_prices = price_df.iloc[-1]
    
    for ticker in tickers:
        current_w = current_weights.get(ticker, 0)
        target_w = target_weights.get(ticker, 0)
        delta_w = target_w - current_w
        
        delta_value = delta_w * investable_capital
        price = latest_prices[ticker]
        shares = int(delta_value / price)
        
        trades.append({
            'Ticker': ticker,
            'Current Weight': f"{current_w:.2%}",
            'Target Weight': f"{target_w:.2%}",
            'Weight Change': f"{delta_w:+.2%}",
            'Current Price': f"₹{price:,.2f}",
            'Shares to Trade': shares,
            'Trade Value': f"₹{abs(shares * price):,.2f}",
            'Action': '🟢 BUY' if shares > 0 else ('🔴 SELL' if shares < 0 else '⚪ HOLD')
        })
    
    return pd.DataFrame(trades)


def create_fan_chart(result, title, color):
    """Create a fan chart for portfolio paths."""
    paths = result.portfolio_paths
    n_steps = paths.shape[0]
    x = np.arange(n_steps)
    
    pct5 = np.percentile(paths, 5, axis=1)
    pct25 = np.percentile(paths, 25, axis=1)
    pct50 = np.percentile(paths, 50, axis=1)
    pct75 = np.percentile(paths, 75, axis=1)
    pct95 = np.percentile(paths, 95, axis=1)
    
    fig = go.Figure()
    
    # 5-95% band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([pct95, pct5[::-1]]),
        fill='toself',
        fillcolor=f'rgba({color}, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='5-95%',
        showlegend=True
    ))
    
    # 25-75% band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([pct75, pct25[::-1]]),
        fill='toself',
        fillcolor=f'rgba({color}, 0.25)',
        line=dict(color='rgba(0,0,0,0)'),
        name='25-75%',
        showlegend=True
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=x, y=pct50,
        mode='lines',
        line=dict(color=f'rgb({color})', width=3),
        name='Median'
    ))
    
    # Initial value line
    fig.add_hline(y=result.initial_value, line_dash="dash", line_color="gray",
                  annotation_text="Initial Value")
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#a29bfe')),
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value (₹)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dfe6e9'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_comparison_bar_chart(comparison):
    """Create bar chart comparing current vs optimized metrics."""
    metrics = ['Mean Return', 'Sharpe Ratio', 'VaR (95%)', 'CVaR (95%)']
    current_vals = [
        comparison.current.mean_return * 100,
        comparison.sharpe_current,
        comparison.current.var_95 * 100,
        comparison.current.cvar_95 * 100
    ]
    optimized_vals = [
        comparison.optimized.mean_return * 100,
        comparison.sharpe_optimized,
        comparison.optimized.var_95 * 100,
        comparison.optimized.cvar_95 * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Portfolio',
        x=metrics,
        y=current_vals,
        marker_color='#6c5ce7',
        text=[f'{v:.2f}' for v in current_vals],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Optimized Portfolio (Robust RL)',
        x=metrics,
        y=optimized_vals,
        marker_color='#00c896',
        text=[f'{v:.2f}' for v in optimized_vals],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text='Portfolio Metrics Comparison', font=dict(size=18, color='#a29bfe')),
        barmode='group',
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dfe6e9'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_allocation_pie_charts(current_weights, target_weights):
    """Create side-by-side pie charts for allocation comparison."""
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]],
                        subplot_titles=('Current Allocation', 'Optimized Allocation (Robust RL)'))
    
    tickers = list(current_weights.keys())
    current_vals = list(current_weights.values())
    target_vals = [target_weights.get(t, 0) for t in tickers]
    
    fig.add_trace(go.Pie(
        labels=tickers,
        values=current_vals,
        hole=0.4,
        marker_colors=px.colors.sequential.Purples[2:],
        textinfo='label+percent',
        textfont=dict(color='white')
    ), row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=tickers,
        values=target_vals,
        hole=0.4,
        marker_colors=px.colors.sequential.Greens[2:],
        textinfo='label+percent',
        textfont=dict(color='white')
    ), row=1, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dfe6e9'),
        showlegend=False
    )
    
    return fig


def create_terminal_distribution(comparison):
    """Create terminal return distribution histogram."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=comparison.current.terminal_returns * 100,
        name='Current Portfolio',
        marker_color='#6c5ce7',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=comparison.optimized.terminal_returns * 100,
        name='Optimized Portfolio (Robust RL)',
        marker_color='#00c896',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Break-even")
    fig.add_vline(x=comparison.current.var_95 * 100, line_dash="dot", line_color="#6c5ce7",
                  annotation_text="VaR 95% (Current)")
    fig.add_vline(x=comparison.optimized.var_95 * 100, line_dash="dot", line_color="#00c896",
                  annotation_text="VaR 95% (Optimized)")
    
    fig.update_layout(
        title=dict(text='Terminal Return Distribution', font=dict(size=18, color='#a29bfe')),
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        barmode='overlay',
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dfe6e9'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# ============================================================================
# SIDEBAR - Model Info & Robust Parameters
# ============================================================================

with st.sidebar:
    st.markdown("### 🤖 Model Information")
    
    model, model_path = load_rl_model()
    training_df = load_training_data()
    
    if model is not None:
        st.success("✓ RL Model Loaded")
        st.caption(f"Path: `{model_path.name}`")
    else:
        st.error("✗ No model found")
    
    st.markdown("---")
    
    # ========================================================================
    # ROBUST INFERENCE PARAMETERS (Display Only)
    # ========================================================================
    
    st.markdown("### 🛡️ Robust Inference Parameters")
    st.caption("*These parameters are optimized for stable weight extraction and reliable tail-risk metrics.*")
    
    with st.expander("📊 Start-Step Ensemble", expanded=True):
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Enabled | ✅ Yes |
        | Start Offsets | `{ROBUST_CONFIG['start_offsets']}` |
        | Aggregation | `{ROBUST_CONFIG['ensemble_agg']}` |
        
        **Why?** Running the policy from multiple starting points captures variation across market contexts. Median aggregation is robust to outlier runs.
        """)
    
    with st.expander("🔄 Last-K Smoothing", expanded=True):
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Enabled | ✅ Yes |
        | K (steps) | `{ROBUST_CONFIG['last_k']}` |
        
        **Why?** Using median of last 5 steps (~1 trading week) smooths daily noise while preserving the policy's recent signal.
        """)
    
    with st.expander("📏 Cap & Floor", expanded=True):
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Enabled | ✅ Yes |
        | Max Weight (Cap) | `{ROBUST_CONFIG['cap']*100:.0f}%` |
        | Min Weight (Floor) | `{ROBUST_CONFIG['floor']*100:.0f}%` |
        
        **Why?** Conservative cap prevents concentration that kills CVaR. Floor ensures minimum diversification across all assets.
        """)
    
    with st.expander("🌡️ Softmax Temperature", expanded=True):
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Enabled | ✅ Yes |
        | Temperature (T) | `{ROBUST_CONFIG['temperature']}` |
        
        **Why?** Flattens extreme allocations. T=2.0 provides moderate smoothing without destroying the policy's signal.
        """)
    
    with st.expander("🎲 Monte Carlo Settings", expanded=True):
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Paths | `{ROBUST_CONFIG['n_paths']}` |
        | Steps | `{ROBUST_CONFIG['n_steps']}` |
        | Noise Type | `{ROBUST_CONFIG['noise']}` |
        
        **Why?** 2000 paths provide reliable tail metrics (VaR/CVaR need 1000+). Block bootstrap preserves temporal dependence in returns.
        """)
    
    st.markdown("---")
    
    with st.expander("📊 Training Parameters", expanded=False):
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Algorithm | PPO (Proximal Policy Optimization) |
        | Learning Rate | 3e-4 |
        | Batch Size | 64 |
        | N Steps | 2048 |
        | Gamma | 0.99 |
        | GAE Lambda | 0.95 |
        | Clip Range | 0.2 |
        | Entropy Coef | 0.01 |
        | Training Steps | ~500K |
        """)
    
    with st.expander("🏛️ Portfolio Environment", expanded=False):
        st.markdown("""
        **State Space (Observation):**
        - Window of past 20 days returns
        - Current portfolio weights
        - Technical indicators (momentum, volatility)
        
        **Action Space:**
        - Continuous weights for each asset
        - Softmax normalized to sum to 1
        
        **Parameters:**
        - Transaction Cost: 0.1%
        - Window Size: 20 days
        - Assets: 25 Indian stocks (NSE)
        """)
    
    with st.expander("🧠 RL Architecture", expanded=False):
        st.markdown("""
        **Policy Network:**
        ```
        Input → Dense(64, ReLU)
              → Dense(64, ReLU)
              → Dense(n_assets, Softmax)
        ```
        
        **Value Network:**
        ```
        Input → Dense(64, ReLU)
              → Dense(64, ReLU)
              → Dense(1)
        ```
        
        **Shared Features:** No (separate networks)
        """)
    
    with st.expander("🎯 Reward Function", expanded=False):
        st.markdown("""
        ```python
        raw_reward = portfolio_return 
                   - alpha * (vol ** 2) 
                   - trans_cost 
                   - turnover_pen 
                   - cvar_penalty
        ```
        
        Where:
        - `portfolio_return`: Daily log return
        - `alpha`: Risk aversion (0.0 = return only)
        - `transaction_cost`: 0.1% per trade
        - `turnover`: Sum of absolute weight changes
        - `cvar_penalty`: Conditional Value at Risk penalty
        
        **Objective:** Maximize risk-adjusted returns while minimizing trading costs and tail risk.
        """)
    
    if training_df is not None:
        with st.expander("📈 Training Data", expanded=False):
            st.markdown(f"""
            - **Tickers:** {len(training_df.columns)} Indian stocks
            - **Date Range:** {training_df.index[0].strftime('%Y-%m-%d')} to {training_df.index[-1].strftime('%Y-%m-%d')}
            - **Total Days:** {len(training_df)}
            """)
            st.caption("Tickers: " + ", ".join(training_df.columns[:10].tolist()) + "...")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Hero Title
st.markdown('<h1 class="hero-title">🛡️ RL Portfolio Optimizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Optimize your portfolio using Reinforcement Learning with <span class="robust-badge">ROBUST INFERENCE</span></p>', unsafe_allow_html=True)

st.markdown("---")

# Info box about robust inference
st.markdown("""
<div class="info-box">
    <p><strong>🛡️ Robust Inference Enabled</strong></p>
    <p>This app uses advanced techniques to extract stable portfolio weights:</p>
    <ul>
        <li><strong>Start-step ensemble</strong> — Run policy from 5 different market contexts</li>
        <li><strong>Last-K smoothing</strong> — Median of last 5 steps removes final-step noise</li>
        <li><strong>Cap + Floor (15%/2%)</strong> — Prevents extreme concentration</li>
        <li><strong>Softmax temperature (T=2.0)</strong> — Flattens overconfident allocations</li>
        <li><strong>2000 MC paths</strong> — Reliable tail metrics (VaR/CVaR)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# USER INPUTS
# ============================================================================

st.markdown("### 💼 Configure Your Portfolio")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    
    # Available tickers from training data
    available_tickers = list(training_df.columns) if training_df is not None else []
    
    selected_tickers = st.multiselect(
        "Select Stocks (from training universe)",
        options=available_tickers,
        default=["HDFCBANK.NS", "TCS.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS"] if available_tickers else [],
        help="Select stocks that were part of the RL training data"
    )
    
    st.markdown("**Set Current Weights (must sum to 100%)**")
    
    weights_input = {}
    if selected_tickers:
        default_weight = 100.0 / len(selected_tickers)
        cols = st.columns(min(len(selected_tickers), 4))
        for i, ticker in enumerate(selected_tickers):
            with cols[i % 4]:
                weights_input[ticker] = st.number_input(
                    f"{ticker.replace('.NS', '')}",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_weight,
                    step=1.0,
                    key=f"weight_{ticker}"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    
    total_capital = st.number_input(
        "💰 Total Capital (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    reserved_cash = st.number_input(
        "🏦 Reserved Cash (₹)",
        min_value=0,
        max_value=int(total_capital * 0.5),
        value=10000,
        step=1000
    )
    
    investable = total_capital - reserved_cash
    st.metric("Investable Capital", f"₹{investable:,.0f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Validate weights
total_weight = sum(weights_input.values()) if weights_input else 0
if selected_tickers:
    if abs(total_weight - 100.0) > 0.1:
        st.warning(f"⚠️ Weights sum to {total_weight:.1f}% (should be 100%)")
    else:
        st.success(f"✓ Weights sum to {total_weight:.1f}%")

st.markdown("---")

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

if st.button("🚀 Run Robust Portfolio Optimization", use_container_width=True):
    
    if not selected_tickers:
        st.error("Please select at least one stock!")
    elif abs(total_weight - 100.0) > 0.1:
        st.error("Please ensure weights sum to 100%!")
    elif model is None:
        st.error("No RL model found! Please ensure model is in 'models/' folder.")
    else:
        # Normalize user weights
        current_weights = {t: w / 100.0 for t, w in weights_input.items()}
        
        with st.spinner("Fetching price data from Yahoo Finance..."):
            price_df = fetch_price_data(
                selected_tickers,
                "2021-01-01",
                pd.Timestamp.today().strftime("%Y-%m-%d")
            )
        
        if price_df is None or price_df.empty:
            st.error("Failed to fetch price data!")
        else:
            st.success(f"✓ Loaded {len(price_df)} days of price data")
            
            # Get ROBUST RL weights
            with st.spinner("Running RL policy with ROBUST inference (ensemble + smoothing + cap/floor)..."):
                target_weights, common_tickers = get_rl_weights_robust(
                    model, training_df, selected_tickers
                )
            
            if target_weights is None:
                st.error("None of your selected tickers are in the RL training data!")
            else:
                st.success("✓ Robust weight extraction complete")
                
                # Filter to common tickers
                current_weights = {t: current_weights.get(t, 0) for t in common_tickers}
                total_cw = sum(current_weights.values())
                current_weights = {t: w / total_cw for t, w in current_weights.items()}
                
                # Run Monte Carlo simulation with ROBUST settings
                with st.spinner(f"Running Monte Carlo simulation ({ROBUST_CONFIG['n_paths']} paths, {ROBUST_CONFIG['n_steps']} days, {ROBUST_CONFIG['noise']} noise)..."):
                    capital_config = CapitalConfig(total_capital, reserved_cash)
                    
                    comparison = run_portfolio_comparison(
                        price_df=price_df[common_tickers],
                        current_weights=current_weights,
                        target_weights=target_weights,
                        initial_capital=capital_config.investable_capital,
                        n_paths=ROBUST_CONFIG['n_paths'],
                        n_steps=ROBUST_CONFIG['n_steps'],
                        seed=42,
                        noise=ROBUST_CONFIG['noise'],
                    )
                
                st.success("✓ Optimization complete!")
                
                st.markdown("---")
                
                # ============================================================
                # RESULTS DISPLAY
                # ============================================================
                
                # Section 1: Weight Comparison
                st.markdown("### 📊 Portfolio Allocation")
                
                fig_pie = create_allocation_pie_charts(current_weights, target_weights)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Section 2: Metrics Comparison
                st.markdown("### 📈 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    delta_return = (comparison.optimized.mean_return - comparison.current.mean_return) * 100
                    st.metric(
                        "Mean Return",
                        f"{comparison.optimized.mean_return:.2%}",
                        f"{delta_return:+.2f}%"
                    )
                
                with col2:
                    delta_sharpe = comparison.sharpe_optimized - comparison.sharpe_current
                    st.metric(
                        "Sharpe Ratio",
                        f"{comparison.sharpe_optimized:.3f}",
                        f"{delta_sharpe:+.3f}"
                    )
                
                with col3:
                    delta_var = (comparison.optimized.var_95 - comparison.current.var_95) * 100
                    st.metric(
                        "VaR (95%)",
                        f"{comparison.optimized.var_95:.2%}",
                        f"{delta_var:+.2f}%",
                        delta_color="inverse"
                    )
                
                with col4:
                    delta_cvar = (comparison.optimized.cvar_95 - comparison.current.cvar_95) * 100
                    st.metric(
                        "CVaR (95%)",
                        f"{comparison.optimized.cvar_95:.2%}",
                        f"{delta_cvar:+.2f}%",
                        delta_color="inverse"
                    )
                
                # Comparison bar chart
                fig_bar = create_comparison_bar_chart(comparison)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Metric-by-metric winner
                st.markdown("#### 🏆 Metric-by-Metric Winner")
                rl_wins = 0
                metric_results = []
                
                if comparison.optimized.mean_return > comparison.current.mean_return:
                    metric_results.append(("Mean Return", "✅ Robust RL", "#00c896"))
                    rl_wins += 1
                else:
                    metric_results.append(("Mean Return", "❌ Baseline", "#ff4757"))
                
                if comparison.sharpe_optimized > comparison.sharpe_current:
                    metric_results.append(("Sharpe Ratio", "✅ Robust RL", "#00c896"))
                    rl_wins += 1
                else:
                    metric_results.append(("Sharpe Ratio", "❌ Baseline", "#ff4757"))
                
                if comparison.optimized.var_95 > comparison.current.var_95:
                    metric_results.append(("VaR (95%)", "✅ Robust RL", "#00c896"))
                    rl_wins += 1
                else:
                    metric_results.append(("VaR (95%)", "❌ Baseline", "#ff4757"))
                
                if comparison.optimized.cvar_95 > comparison.current.cvar_95:
                    metric_results.append(("CVaR (95%)", "✅ Robust RL", "#00c896"))
                    rl_wins += 1
                else:
                    metric_results.append(("CVaR (95%)", "❌ Baseline", "#ff4757"))
                
                cols = st.columns(4)
                for i, (metric, winner, color) in enumerate(metric_results):
                    with cols[i]:
                        st.markdown(f"**{metric}**")
                        st.markdown(f"<span style='color: {color}'>{winner}</span>", unsafe_allow_html=True)
                
                st.markdown(f"### 🏆 Robust RL wins **{rl_wins}/4** metrics")
                
                # Detailed Metrics Table in Expander
                with st.expander("📊 View Detailed Metrics Table", expanded=False):
                    metrics_data = {
                        "Metric": [
                            "Initial Value",
                            "Mean Terminal Value",
                            "Std Terminal Value",
                            "Mean Return",
                            "Std Return",
                            "VaR (95%)",
                            "CVaR (95%)",
                            "VaR (99%)",
                            "CVaR (99%)",
                            "Sharpe Ratio",
                        ],
                        "Current Portfolio": [
                            f"₹{comparison.current.initial_value:,.2f}",
                            f"₹{comparison.current.mean_terminal_value:,.2f}",
                            f"₹{comparison.current.std_terminal_value:,.2f}",
                            f"{comparison.current.mean_return:.2%}",
                            f"{comparison.current.std_return:.2%}",
                            f"{comparison.current.var_95:.2%}",
                            f"{comparison.current.cvar_95:.2%}",
                            f"{comparison.current.var_99:.2%}",
                            f"{comparison.current.cvar_99:.2%}",
                            f"{comparison.sharpe_current:.3f}",
                        ],
                        "Optimized Portfolio (Robust RL)": [
                            f"₹{comparison.optimized.initial_value:,.2f}",
                            f"₹{comparison.optimized.mean_terminal_value:,.2f}",
                            f"₹{comparison.optimized.std_terminal_value:,.2f}",
                            f"{comparison.optimized.mean_return:.2%}",
                            f"{comparison.optimized.std_return:.2%}",
                            f"{comparison.optimized.var_95:.2%}",
                            f"{comparison.optimized.cvar_95:.2%}",
                            f"{comparison.optimized.var_99:.2%}",
                            f"{comparison.optimized.cvar_99:.2%}",
                            f"{comparison.sharpe_optimized:.3f}",
                        ],
                    }
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Style the table
                    def highlight_better(row):
                        styles = [''] * len(row)
                        metric = row['Metric']
                        current_str = row['Current Portfolio']
                        opt_str = row['Optimized Portfolio (Robust RL)']
                        
                        # Parse numeric values
                        try:
                            if '₹' in current_str:
                                current_val = float(current_str.replace('₹', '').replace(',', ''))
                                opt_val = float(opt_str.replace('₹', '').replace(',', ''))
                            elif '%' in current_str:
                                current_val = float(current_str.replace('%', ''))
                                opt_val = float(opt_str.replace('%', ''))
                            else:
                                current_val = float(current_str)
                                opt_val = float(opt_str)
                            
                            # For VaR/CVaR, less negative is better
                            if 'VaR' in metric or 'CVaR' in metric:
                                if opt_val > current_val:
                                    styles[2] = 'background-color: rgba(0, 200, 150, 0.3); font-weight: bold'
                                elif opt_val < current_val:
                                    styles[2] = 'background-color: rgba(255, 71, 87, 0.3)'
                            # For Std, lower is better
                            elif 'Std' in metric:
                                if opt_val < current_val:
                                    styles[2] = 'background-color: rgba(0, 200, 150, 0.3); font-weight: bold'
                                elif opt_val > current_val:
                                    styles[2] = 'background-color: rgba(255, 71, 87, 0.3)'
                            # For others, higher is better
                            else:
                                if opt_val > current_val:
                                    styles[2] = 'background-color: rgba(0, 200, 150, 0.3); font-weight: bold'
                                elif opt_val < current_val:
                                    styles[2] = 'background-color: rgba(255, 71, 87, 0.3)'
                        except:
                            pass
                        
                        return styles
                    
                    styled_metrics = metrics_df.style.apply(highlight_better, axis=1)
                    st.dataframe(styled_metrics, use_container_width=True, hide_index=True)
                    
                    # Summary stats
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Return Improvement", f"{comparison.return_improvement:+.2%}")
                    with col2:
                        var_change = (comparison.optimized.var_95 - comparison.current.var_95) * 100
                        st.metric("VaR 95% Change", f"{var_change:+.2f}%", delta_color="inverse")
                    with col3:
                        cvar_change = (comparison.optimized.cvar_95 - comparison.current.cvar_95) * 100
                        st.metric("CVaR 95% Change", f"{cvar_change:+.2f}%", delta_color="inverse")
                
                st.markdown("---")
                
                # Section 3: Fan Charts
                st.markdown("### 📉 Simulated Portfolio Paths")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_fan_current = create_fan_chart(comparison.current, "Current Portfolio", "108, 92, 231")
                    st.plotly_chart(fig_fan_current, use_container_width=True)
                
                with col2:
                    fig_fan_opt = create_fan_chart(comparison.optimized, "Optimized Portfolio (Robust RL)", "0, 200, 150")
                    st.plotly_chart(fig_fan_opt, use_container_width=True)
                
                # Terminal distribution
                fig_dist = create_terminal_distribution(comparison)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.markdown("---")
                
                # Section 4: Trade Recommendations with Confidence Scores
                st.markdown("### 📋 Buy/Sell Recommendations")
                
                # Generate integrated recommendations
                with st.spinner("Generating recommendations..."):
                    recs = generate_integrated_recommendations(
                        price_df=price_df[common_tickers],
                        current_weights=current_weights,
                        target_weights=target_weights,
                        capital_config=capital_config,
                        min_trade_value=100,
                        min_trade_pct=0.005,
                    )
                
                # Buy Recommendations Table
                if recs.buy_recommendations:
                    st.markdown("#### 🟢 Buy Recommendations")
                    buy_data = []
                    for rec in recs.buy_recommendations:
                        buy_data.append({
                            'Rank': rec.rank,
                            'Ticker': rec.ticker,
                            'Confidence': rec.confidence,
                            'Delta Value': f'₹{rec.delta_value:,.2f}',
                            'Weight Change': f'{rec.delta_weight:+.2%}',
                            'Signal Score': f'{rec.signal_score:+.3f}',
                            'Rationale': rec.rationale[:60] + '...' if len(rec.rationale) > 60 else rec.rationale,
                        })
                    buy_df = pd.DataFrame(buy_data)
                    
                    def style_confidence_buy(val):
                        if val == 'HIGH':
                            return 'background-color: rgba(0, 200, 150, 0.4); color: white; font-weight: bold'
                        elif val == 'MEDIUM':
                            return 'background-color: rgba(241, 196, 15, 0.4); color: white; font-weight: bold'
                        else:
                            return 'background-color: rgba(108, 92, 231, 0.3); color: white'
                    
                    st.dataframe(
                        buy_df.style.applymap(style_confidence_buy, subset=['Confidence']),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No buy recommendations.")
                
                # Sell Recommendations Table
                if recs.sell_recommendations:
                    st.markdown("#### 🔴 Sell Recommendations")
                    sell_data = []
                    for rec in recs.sell_recommendations:
                        sell_data.append({
                            'Rank': rec.rank,
                            'Ticker': rec.ticker,
                            'Confidence': rec.confidence,
                            'Delta Value': f'₹{rec.delta_value:,.2f}',
                            'Weight Change': f'{rec.delta_weight:+.2%}',
                            'Signal Score': f'{rec.signal_score:+.3f}',
                            'Rationale': rec.rationale[:60] + '...' if len(rec.rationale) > 60 else rec.rationale,
                        })
                    sell_df = pd.DataFrame(sell_data)
                    
                    def style_confidence_sell(val):
                        if val == 'HIGH':
                            return 'background-color: rgba(255, 71, 87, 0.5); color: white; font-weight: bold'
                        elif val == 'MEDIUM':
                            return 'background-color: rgba(241, 196, 15, 0.4); color: white; font-weight: bold'
                        else:
                            return 'background-color: rgba(108, 92, 231, 0.3); color: white'
                    
                    st.dataframe(
                        sell_df.style.applymap(style_confidence_sell, subset=['Confidence']),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No sell recommendations.")
                
                st.markdown("---")
                
                # Section 5: Shares to Trade
                st.markdown("### 🛒 Shares to Trade")
                
                trades_df = compute_shares_to_trade(
                    current_weights, target_weights, common_tickers,
                    price_df[common_tickers], capital_config.investable_capital
                )
                
                # Style the dataframe
                def highlight_action(row):
                    if 'BUY' in row['Action']:
                        return ['background-color: rgba(0, 200, 150, 0.2)'] * len(row)
                    elif 'SELL' in row['Action']:
                        return ['background-color: rgba(255, 71, 87, 0.2)'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    trades_df.style.apply(highlight_action, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Trade summary
                buys = trades_df[trades_df['Action'].str.contains('BUY')]
                sells = trades_df[trades_df['Action'].str.contains('SELL')]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card-green">
                        <h4>🟢 Total Buys</h4>
                        <p style="font-size: 1.5rem; font-weight: bold;">{len(buys)} stocks</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card-red">
                        <h4>🔴 Total Sells</h4>
                        <p style="font-size: 1.5rem; font-weight: bold;">{len(sells)} stocks</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    holds = len(trades_df) - len(buys) - len(sells)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⚪ Holds</h4>
                        <p style="font-size: 1.5rem; font-weight: bold;">{holds} stocks</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Section 6: Why Robust Inference?
                st.markdown("### 💡 Why Robust Inference?")
                
                st.markdown("""
                <div class="info-box">
                    <p><strong>Traditional RL weight extraction can be unstable due to:</strong></p>
                    <ul>
                        <li>📉 <strong>Sensitivity to start conditions</strong> — Single rollout may hit an outlier market context</li>
                        <li>🎲 <strong>Final-step noise</strong> — Last action can be noisy; doesn't reflect stable policy</li>
                        <li>📊 <strong>Extreme allocations</strong> — Policy may be overconfident on a few assets</li>
                        <li>🔢 <strong>Unreliable tail metrics</strong> — VaR/CVaR need 1000+ paths for stability</li>
                    </ul>
                    
                    <p><strong>Robust inference fixes these issues by:</strong></p>
                    <ul>
                        <li>🔄 <strong>Start-step ensemble</strong> — Run from 5 different starting points, take median</li>
                        <li>📈 <strong>Last-K smoothing</strong> — Use median of last 5 steps' weights</li>
                        <li>📏 <strong>Cap + Floor</strong> — Limit max/min per-asset allocation (15%/2%)</li>
                        <li>🌡️ <strong>Softmax temperature</strong> — Flatten extreme allocations (T=2.0)</li>
                        <li>🎲 <strong>2000 MC paths</strong> — Reliable tail metrics with block bootstrap</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Weight change reasons
                with st.expander("📋 Detailed Weight Changes", expanded=True):
                    for ticker in common_tickers:
                        curr_w = current_weights.get(ticker, 0)
                        targ_w = target_weights.get(ticker, 0)
                        delta = targ_w - curr_w
                        
                        if delta > 0.01:
                            st.markdown(f"""
                            **{ticker}**: {curr_w:.1%} → {targ_w:.1%} (↑ {delta:.1%})  
                            *Reason: Robust RL ensemble recommends increased exposure based on risk-return profile.*
                            """)
                        elif delta < -0.01:
                            st.markdown(f"""
                            **{ticker}**: {curr_w:.1%} → {targ_w:.1%} (↓ {abs(delta):.1%})  
                            *Reason: Robust RL recommends reducing exposure due to risk/correlation factors.*
                            """)
                        else:
                            st.markdown(f"""
                            **{ticker}**: {curr_w:.1%} → {targ_w:.1%} (≈ no change)  
                            *Reason: Current allocation is near optimal according to robust ensemble.*
                            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #636e72; font-size: 0.9rem;">
    Built with ❤️ using Streamlit | RL Model: PPO (Stable-Baselines3) | <span style="color: #00c896">Robust Inference</span> | Data: Yahoo Finance
</div>
""", unsafe_allow_html=True)
