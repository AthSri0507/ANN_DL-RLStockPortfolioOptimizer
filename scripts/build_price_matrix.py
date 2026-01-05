# scripts/build_price_matrix.py
import yfinance as yf
import pandas as pd

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]  # update list
start = "2018-01-01"
end = "2024-12-31"

data = yf.download(tickers, start=start, end=end, progress=False, group_by='column', auto_adjust=False)
# If multi-column returned, take 'Adj Close'; otherwise 'Close'
if ("Adj Close" in data.columns.levels[0]) or isinstance(data.columns, pd.MultiIndex):
    price_df = data["Adj Close"].copy()
else:
    price_df = data[[f"{t}" for t in tickers]].copy()

# keep dates ascending, align, forward-fill short gaps
price_df = price_df.sort_index().ffill().dropna(how="all")

price_df.to_csv("data/price_matrix.csv")
print("Wrote data/price_matrix.csv with shape", price_df.shape)