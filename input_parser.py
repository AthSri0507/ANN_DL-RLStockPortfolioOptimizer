import pandas as pd
from typing import List

REQUIRED_COLUMNS = ["Ticker", "Quantity"]


def parse_excel(path: str) -> pd.DataFrame:
    """Read an Excel file and return validated portfolio DataFrame.

    Expects columns `Ticker` and `Quantity`.
    """
    df = pd.read_excel(path, engine="openpyxl")
    return from_dataframe(df)


def from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a DataFrame containing portfolio info.

    Returns a DataFrame with columns `Ticker` (uppercased str) and `Quantity` (numeric).
    """
    if "Ticker" not in df.columns or "Quantity" not in df.columns:
        raise ValueError("Excel must contain 'Ticker' and 'Quantity' columns")
    out = df.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    if out["Quantity"].isnull().any():
        raise ValueError("Some quantities could not be converted to numeric")
    out = out[["Ticker", "Quantity"]].reset_index(drop=True)
    return out


def tickers_from_excel(path: str) -> List[str]:
    df = parse_excel(path)
    return df["Ticker"].tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to portfolio Excel (Ticker, Quantity)")
    args = parser.parse_args()
    print(parse_excel(args.path))
