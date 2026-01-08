"""
Generate a synthetic India price CSV for RL training.
Saves to: data/rl_training_price_data_india.csv

Usage:
  python scripts/generate_india_rl_csv.py

This script simulates realistic-ish daily adjusted close prices
for 25 Indian tickers using geometric Brownian motion with
sector-aware vol/drift defaults. Seeded for reproducibility.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "rl_training_price_data_india.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 25 tickers provided by user
TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "RELIANCE.NS", "LT.NS", "ONGC.NS", "NTPC.NS", "HINDUNILVR.NS",
    "ITC.NS", "NESTLEIND.NS", "ASIANPAINT.NS", "MARUTI.NS", "TATAMOTORS.NS",
    "M&M.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "BHARTIARTL.NS",
]

# Sector-based defaults (annualized drift, annualized vol)
SECTOR_PARAMS = {
    "bank": (0.08, 0.30),
    "it": (0.10, 0.25),
    "energy": (0.06, 0.28),
    "industrial": (0.07, 0.26),
    "fmcg": (0.05, 0.18),
    "auto": (0.06, 0.30),
    "pharma": (0.07, 0.22),
    "telecom": (0.05, 0.24),
}

SECTOR_MAP = {
    "HDFCBANK.NS": "bank",
    "ICICIBANK.NS": "bank",
    "SBIN.NS": "bank",
    "AXISBANK.NS": "bank",
    "KOTAKBANK.NS": "bank",
    "TCS.NS": "it",
    "INFY.NS": "it",
    "HCLTECH.NS": "it",
    "WIPRO.NS": "it",
    "TECHM.NS": "it",
    "RELIANCE.NS": "energy",
    "LT.NS": "industrial",
    "ONGC.NS": "energy",
    "NTPC.NS": "energy",
    "HINDUNILVR.NS": "fmcg",
    "ITC.NS": "fmcg",
    "NESTLEIND.NS": "fmcg",
    "ASIANPAINT.NS": "fmcg",
    "MARUTI.NS": "auto",
    "TATAMOTORS.NS": "auto",
    "M&M.NS": "auto",
    "SUNPHARMA.NS": "pharma",
    "DRREDDY.NS": "pharma",
    "CIPLA.NS": "pharma",
    "BHARTIARTL.NS": "telecom",
}

# Simulation params
SEED = 42
N_DAYS = 1000  # business days (~4 years)
START_DATE = "2018-01-01"

np.random.seed(SEED)

dates = pd.bdate_range(start=START_DATE, periods=N_DAYS)

prices = pd.DataFrame(index=dates, columns=TICKERS, dtype=float)

for t in TICKERS:
    sector = SECTOR_MAP.get(t, "fmcg")
    mu_ann, sigma_ann = SECTOR_PARAMS.get(sector, (0.06, 0.22))
    mu = mu_ann / 252.0
    sigma = sigma_ann / np.sqrt(252.0)

    # initial price sampled to be realistic (INR): 200 - 3000
    s0 = float(np.round(10 ** np.random.uniform(np.log10(50), np.log10(3000)), 2))

    # generate standard normal shocks
    zs = np.random.normal(loc=0.0, scale=1.0, size=N_DAYS)

    log_returns = (mu - 0.5 * sigma ** 2) + sigma * zs
    log_price = np.cumsum(log_returns) + np.log(s0)
    series = np.exp(log_price)

    # introduce gentle mean-reversion / seasonality small component
    season = 1.0 + 0.01 * np.sin(np.linspace(0, 6 * np.pi, N_DAYS))
    series = series * season

    prices[t] = np.round(series, 2)

# Save CSV (index as dates)
prices.to_csv(OUT_PATH)
print(f"Wrote synthetic RL training CSV to: {OUT_PATH}")
print(f"Shape: {prices.shape[0]} rows x {prices.shape[1]} columns")
