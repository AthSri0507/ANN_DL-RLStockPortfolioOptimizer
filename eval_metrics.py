# eval_metrics.py
import argparse
import glob, os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from portfolio_env import PortfolioEnv

def max_drawdown(returns):
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    return drawdown.min()

def historical_var(returns, alpha=0.95):
    q = np.percentile(returns, (1 - alpha) * 100)
    return q

def historical_cvar(returns, alpha=0.95):
    q = historical_var(returns, alpha)
    tail = returns[returns <= q]
    return tail.mean() if len(tail) > 0 else q

def find_latest_model(path="models"):
    files = glob.glob(os.path.join(path, "**", "*.zip"), recursive=True) + glob.glob(os.path.join(path, "*.zip"))
    if not files:
        raise FileNotFoundError("No .zip model files found in models/")
    return max(files, key=os.path.getmtime)

def evaluate(model_path, price_csv, n_assets, trading_days=252, transaction_cost=0.001):
    price_df = pd.read_csv(price_csv, index_col=0, parse_dates=True)
    env = PortfolioEnv(n_assets=n_assets, price_df=price_df, transaction_cost=transaction_cost)
    model = PPO.load(model_path)

    # reset and step loop
    try:
        obs, _ = env.reset()
    except Exception:
        obs = env.reset()

    returns = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        # support both (obs, reward, done, info) and (obs, reward, done, _, info)
        if len(step_out) == 4:
            obs, reward, done, info = step_out
        else:
            obs, reward, done, _, info = step_out
        # prefer explicit portfolio return from info, fallback to reward
        port_ret = info.get("portfolio_return", reward)
        returns.append(port_ret)

    returns = np.asarray(returns)
    n = len(returns)
    cumulative = np.prod(1 + returns) - 1
    annualized_return = (1 + cumulative) ** (trading_days / max(1, n)) - 1
    annualized_vol = returns.std(ddof=0) * (trading_days ** 0.5)
    sharpe = annualized_return / (annualized_vol + 1e-9)
    mdd = max_drawdown(returns)
    var95 = historical_var(returns, alpha=0.95)
    cvar95 = historical_cvar(returns, alpha=0.95)

    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "VaR_95": var95,
        "CVaR_95": cvar95,
        "n_steps": n
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Path to model .zip (default: latest in models/)")
    p.add_argument("--price-csv", default="data/price_matrix.csv")
    p.add_argument("--n-assets", type=int, required=True)
    p.add_argument("--trading-days", type=int, default=252)
    args = p.parse_args()

    model_path = args.model or find_latest_model("models")
    print("Using model:", model_path)
    metrics = evaluate(model_path, args.price_csv, args.n_assets, trading_days=args.trading_days)

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")