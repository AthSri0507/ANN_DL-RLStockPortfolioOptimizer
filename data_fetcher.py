import os
from typing import List, Dict

import pandas as pd
import yfinance as yf


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cache_path(cache_dir: str, ticker: str) -> str:
    safe = ticker.replace("/", "_")
    return os.path.join(cache_dir, f"{safe}.csv")


def fetch_price_history(tickers: List[str], start=None, end=None, interval='1d', cache_dir='data_store', force=False) -> Dict[str, pd.DataFrame]:
    """Fetch historical price data for tickers via yfinance and cache per-ticker CSVs.

    Returns a dict ticker -> DataFrame.
    """
    ensure_dir(cache_dir)
    result: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = cache_path(cache_dir, t)
        if (not force) and os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                result[t] = df
                continue
            except Exception:
                pass
        # use yfinance to download
        df = yf.download(t, start=start, end=end, interval=interval, progress=False)
        if df is None or df.empty:
            # save an empty frame to indicate attempted fetch
            df = pd.DataFrame()
        else:
            df.to_csv(path)
        result[t] = df
    return result


def load_cached(ticker: str, cache_dir='data_store') -> pd.DataFrame:
    path = cache_path(cache_dir, ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--cache_dir', default='data_store')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    res = fetch_price_history(args.tickers, start=args.start, end=args.end, cache_dir=args.cache_dir, force=args.force)
    for k, df in res.items():
        print(k, df.shape if (df is not None and not df.empty) else 'empty')
