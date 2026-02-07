# AnNDL: Reinforcement Learning Portfolio Optimizer

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![Stable Baselines3](https://img.shields.io/badge/stable--baselines3-1.8.0%2B-brightgreen?style=flat-square)](https://stable-baselines3.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.0%2B-FF4B4B?style=flat-square)](https://streamlit.io/)

An end-to-end reinforcement learning framework for intelligent portfolio allocation, Monte Carlo simulation, and quantitative evaluation. Train RL agents to optimize portfolio weights, backtest against baselines, and explore allocation strategies through interactive Streamlit dashboards.

## Overview

AnNDL combines modern reinforcement learning with quantitative finance to help investors make data-driven portfolio decisions. The system ingests portfolio data, trains PPO agents to optimize risk-adjusted returns, simulates multiple market scenarios via Monte Carlo methods, and provides actionable allocation recommendations compared to baseline strategies.

### Key Capabilities

- **RL-Driven Allocation**: Train PPO agents to discover optimal portfolio weights using risk-adjusted reward signals
- **Backtesting & Evaluation**: Compare RL agents against buy-and-hold, equal-weight, and other baseline strategies  
- **Monte Carlo Simulation**: Generate 30+ correlated price paths with full GBM calibration and risk metrics
- **Interactive Dashboard**: Web-based Streamlit UI for uploading portfolios, visualizing results, and exploring recommendations
- **Notebook Demos**: Pre-built Jupyter notebooks for experimentation and result reproducibility

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Virtual environment (recommended)

### Installation

Clone the repository:

```bash
git clone https://github.com/AthSri0507/ANN_DL-RLStockPortfolioOptimizer
cd anndl
```

Create and activate a virtual environment:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Web Dashboard

Launch the interactive Streamlit app to upload portfolios and explore RL recommendations:

```bash
streamlit run app_robust.py
```

Open your browser to `http://localhost:8501` to see the dashboard in action.

### Train a Custom RL Agent

Train an agent on your price data (provide your own CSV file):

```bash
python train_agent.py \
  --price-csv prices.csv \
  --n-assets 10 \
  --total-timesteps 100000 \
  --out agent_model.zip
```

For a complete list of training options:

```bash
python train_agent.py --help
```

### Evaluate Agent Performance

Evaluate trained agents using Python or Jupyter:

```python
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("agent_model.zip")

# Run evaluation on test environment
env = PortfolioEnv(price_csv="test_prices.csv")
obs, info = env.reset()
episode_return = 0
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward
    if terminated or truncated:
        break
print(f"Episode return: {episode_return}")
```

### Explore with Jupyter Notebooks

Open notebooks to experiment with the full pipeline:

```bash
jupyter notebook notebooks/Equal_Weight_test.ipynb
```

Available notebooks include baseline strategy analysis, unequal and equal weight tests, and custom scenario exploration.

## Data Universe

The system includes a comprehensive **25-asset Indian stock universe** (NSE tickers) spanning multiple sectors:

- **Banks** (5): HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, AXISBANK.NS, KOTAKBANK.NS
- **IT/Tech** (5): TCS.NS, INFY.NS, HCLTECH.NS, WIPRO.NS, TECHM.NS
- **Pharma** (3): SUNPHARMA.NS, DRREDDY.NS, CIPLA.NS
- **Consumer/FMCG** (4): HINDUNILVR.NS, ITC.NS, NESTLEIND.NS, ASIANPAINT.NS
- **Auto** (3): MARUTI.NS, TATAMOTORS.NS, M&M.NS
- **Energy/Infrastructure** (4): RELIANCE.NS, LT.NS, ONGC.NS, NTPC.NS
- **Large-cap Leaders** (1): BHARTIARTL.NS

The framework supports:
- **Sector-level analysis**: Test RL against baselines on 6+ sector subsets
- **Portfolio diversification**: Evaluate risk-adjusted returns across market-cap tiers
- **Cross-market testing**: Compare equal-weight, market-cap-weighted, and RL-optimized allocations
- **Robust inference**: Ensemble predictions with smoothing, capping, and temperature adjustment


anndl/
├── app_robust.py                   # Robust Streamlit dashboard
├── train_agent.py                  # RL training harness (PPO/SAC)
├── portfolio_env.py                # Gym-compatible environment
├── eval_metrics.py                 # Performance metric calculations
├── data_fetcher.py                 # Yahoo Finance price retrieval
├── input_parser.py                 # Excel portfolio parser
├── currency.py                     # Multi-currency support
├── allocation_converter.py         # Weight <-> quantity conversion
├── features.py                     # Feature engineering
├── global_features.py              # Global portfolio features
│
├── scripts/                        # Utility modules
│   ├── build_price_matrix.py       # Construct training price matrices
│   ├── simulate_gbm.py             # Monte Carlo simulation engine
│   ├── compare_portfolio_mc.py      # Portfolio comparison via MC
│   ├── integrated_recommendations.py # Recommendation engine
│   ├── recommendation_engine.py     # Scoring and ranking
│   └── ...
│
├── notebooks/                      # Jupyter notebooks
│   ├── Equal_Weight_test.ipynb     # Baseline strategy
│   ├── Unequal_Weight_test.ipynb   # Unequal allocation analysis
│   ├── test_book.ipynb             # Testing & validation
│   └── ...
│
├── tests/                          # tests to make sure modules are functional
├── docs/                           # Documentation & images
└── requirements.txt                # Python dependencies
```

## Features in Detail

### 1. Portfolio Environment (`PortfolioEnv`)
- Continuous action space for weight allocation
- Normalized, stock-agnostic feature representation
- Configurable reward signals (risk-adjusted returns, Sharpe-like objectives)
- Transaction cost modeling and penalty shaping
- Multiple evaluation windows (in-sample, out-of-sample, walk-forward)

### 2. RL Training (`train_agent.py`)
- Multi-algorithm support (PPO default, SAC available)
- Vectorized environment for parallel training
- Checkpoint callbacks and model saving
- Hyperparameter control via CLI arguments
- Reproducible training with seeding

### 3. Monte Carlo Simulator
- Per-asset GBM calibration from historical returns
- Cholesky-based correlated path generation
- Configurable portfolio rebalancing
- Risk metrics: VaR, CVaR, terminal value percentiles
- noise - Gaussian,Students-t,bootstrap,block bootstrap

### 4. Evaluation Module
- Backtest against buy-and-hold and equal-weight baselines
- Portfolio metrics: Sharpe ratio, max drawdown, Calmar ratio, Sortino ratio
- Asset-level contribution analysis
- Transaction cost accounting

### 5. Recommendation Engine
- Score-based asset ranking (return, volatility, correlation, momentum)
- Top-3 allocation increase / decrease recommendations
- Risk-adjusted signals with rationales
- Deployment Side regularization with Ensemble methods,Cap and floor for weights and Softmax Temperature

### 6. Interactive Dashboards
- Allocation visualization 
- Performance charts with Plotly
- Real-time comparison to Current Portfolio via Monte Carlo
- Metrics comparison including returns,sharpe ratio,cvar & var at various confidence thresholds


## Usage Examples

### Scenario 1: Load Portfolio from Excel

```python
from input_parser import CapitalConfig, excel_to_capital_config

config = excel_to_capital_config("portfolio.xlsx")
print(config.tickers)
print(config.quantities)
```

### Scenario 2: Fetch and Save Price Data

```python
from data_fetcher import fetch_prices

# Fetch historical price data via Yahoo Finance
# Example: Indian market universe
prices_df = fetch_prices(
    tickers=["INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"],
    start="2020-01-01",
    end="2024-01-01"
)

# Save to CSV for training
prices_df.to_csv("my_prices.csv")
print(prices_df)
```

### Scenario 3: Train and Evaluate Agent

```python
import numpy as np
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO

# Create environment
env = PortfolioEnv(price_csv="prices.csv")

# Train agent
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64)
model.learn(total_timesteps=100000)

# Save and load
model.save("agent_model")
model = PPO.load("agent_model")
```

## Dashboard Interface

The Streamlit dashboard provides an intuitive interface for portfolio optimization:

![Dashboard Overview](docs/images/img1.svg)
*User Portfolio Configuration and Budget Constraint*

![Analysis Tools](docs/images/img2.svg)
*Rl Predicted buy and sell recommendation based on user stock and budget*

![Recommendations](docs/images/img3.svg)
*Monte carlo based Fan-Charts for current vs optimized portfolio*

## Configuration & Customization

### Training Hyperparameters

Edit parameters in `train_agent.py` or pass via command line:

- `--n-steps`: Rollout steps per training update (default: 2048)
- `--batch-size`: Gradient step batch size (default: 64)
- `--learning-rate`: Policy learning rate (default: 3e-4)
- `--reward-alpha`: Risk aversion weight (default: 0.5)
- `--transaction-cost`: Cost per unit turnover (default: 0.001)

### Environment Configuration

Modify `PortfolioEnv` in `portfolio_env.py`:

- Action clipping and normalization
- Feature engineering and normalization
- Reward function shaping
- Episode horizons and step limits

## Documentation

- **Demo Walkthrough**: Open any notebook in `notebooks/` for worked examples
- **API Docs**: Inline docstrings throughout codebase provide function signatures and usage

## Requirements

See [requirements.txt](requirements.txt) for the complete list. Key dependencies:

- **Data**: pandas, numpy, yfinance, scipy
- **RL**: stable-baselines3, gym, gymnasium, torch
- **Visualization**: streamlit, plotly, matplotlib
- **Utilities**: openpyxl (Excel parsing)
- **Testing**: pytest

## Support & Troubleshooting

> [!NOTE]
> **Common Issues:**

**Module not found error**

Make sure you've installed dependencies and the project root is in your Python path. The Streamlit app and scripts handle this automatically.

**Training is very slow**

Consider increasing `--n-positions` or reducing `--total-timesteps` for faster iteration. Use CPU-capable vectorized environments on multi-core machines.

**Simulation output looks unrealistic**

Check that your price data is clean (no NaN, split-adjusted, survivorship bias removed). Ensure historical returns are properly normalized and asset correlation is reasonable.

For additional help, refer to the demo notebooks or open an issue on GitHub.

## References & Resources

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **OpenAI Gym**: https://gymnasium.farama.org/
- **yfinance**: https://github.com/ranaroussi/yfinance
- **Geometric Brownian Motion**: https://en.wikipedia.org/wiki/Geometric_Brownian_motion
- **Reinforcement Learning**: Sutton & Barto, _Reinforcement Learning: An Introduction_

---

**Status**: Active development. Performance and API stability not guaranteed.

For updates, feature requests, or bug reports, please open an issue or submit a pull request.
