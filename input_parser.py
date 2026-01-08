import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

REQUIRED_COLUMNS = ["Ticker", "Quantity"]


@dataclass
class CapitalConfig:
    """Configuration for user capital and allocation constraints.
    
    Attributes:
        total_capital: Total capital available (in currency units)
        reserved_cash: Amount to keep as cash reserve (not invested)
        investable_capital: Computed as total_capital - reserved_cash
    """
    total_capital: float
    reserved_cash: float = 0.0
    
    def __post_init__(self):
        """Validate inputs and compute investable capital."""
        if self.total_capital < 0:
            raise ValueError("total_capital must be non-negative")
        if self.reserved_cash < 0:
            raise ValueError("reserved_cash must be non-negative")
        if self.reserved_cash > self.total_capital:
            raise ValueError("reserved_cash cannot exceed total_capital")
    
    @property
    def investable_capital(self) -> float:
        """Capital available for investment after reserving cash."""
        return self.total_capital - self.reserved_cash
    
    @property
    def reserve_ratio(self) -> float:
        """Fraction of total capital held as reserve."""
        if self.total_capital == 0:
            return 0.0
        return self.reserved_cash / self.total_capital
    
    @property
    def investable_ratio(self) -> float:
        """Fraction of total capital available for investment."""
        return 1.0 - self.reserve_ratio
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_capital": self.total_capital,
            "reserved_cash": self.reserved_cash,
            "investable_capital": self.investable_capital,
            "reserve_ratio": self.reserve_ratio,
            "investable_ratio": self.investable_ratio,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CapitalConfig":
        """Create CapitalConfig from dictionary."""
        return cls(
            total_capital=d["total_capital"],
            reserved_cash=d.get("reserved_cash", 0.0),
        )
    
    def __repr__(self) -> str:
        return (
            f"CapitalConfig(total_capital={self.total_capital:,.2f}, "
            f"reserved_cash={self.reserved_cash:,.2f}, "
            f"investable_capital={self.investable_capital:,.2f})"
        )


def parse_capital_config(
    total_capital: float,
    reserved_cash: float = 0.0,
) -> CapitalConfig:
    """Create and validate a CapitalConfig from user inputs.
    
    Args:
        total_capital: Total capital available (in currency units)
        reserved_cash: Amount to keep as cash reserve (default: 0)
        
    Returns:
        Validated CapitalConfig instance
        
    Raises:
        ValueError: If inputs are invalid
    """
    return CapitalConfig(total_capital=total_capital, reserved_cash=reserved_cash)


def parse_capital_from_excel(
    path: str,
    total_capital_cell: str = "B1",
    reserved_cash_cell: str = "B2",
    sheet_name: str = "Capital",
) -> Optional[CapitalConfig]:
    """Read capital configuration from an Excel file.
    
    Expects a sheet with capital values in specific cells or a simple
    two-row format with labels in column A and values in column B.
    
    Args:
        path: Path to Excel file
        total_capital_cell: Cell containing total capital (default: B1)
        reserved_cash_cell: Cell containing reserved cash (default: B2)
        sheet_name: Name of sheet to read (default: "Capital")
        
    Returns:
        CapitalConfig if sheet exists and values are valid, None otherwise
    """
    import openpyxl
    
    try:
        wb = openpyxl.load_workbook(path, data_only=True)
    except Exception:
        return None
    
    if sheet_name not in wb.sheetnames:
        # Try to read from first sheet if "Capital" sheet doesn't exist
        # Look for rows with "total_capital" or "reserved_cash" labels
        sheet = wb.active
        total_capital = None
        reserved_cash = 0.0
        
        for row in sheet.iter_rows(min_row=1, max_row=20, min_col=1, max_col=2):
            if row[0].value is None:
                continue
            label = str(row[0].value).lower().strip().replace(" ", "_")
            value = row[1].value if row[1].value is not None else 0
            
            if "total" in label and "capital" in label:
                total_capital = float(value)
            elif "reserved" in label or ("cash" in label and "reserve" in label):
                reserved_cash = float(value)
        
        if total_capital is not None:
            return CapitalConfig(total_capital=total_capital, reserved_cash=reserved_cash)
        return None
    
    sheet = wb[sheet_name]
    
    # Try to read from specified cells
    try:
        total_capital = float(sheet[total_capital_cell].value or 0)
        reserved_cash = float(sheet[reserved_cash_cell].value or 0)
        return CapitalConfig(total_capital=total_capital, reserved_cash=reserved_cash)
    except (ValueError, TypeError):
        return None


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
    parser.add_argument("--total-capital", type=float, default=None,
                        help="Total capital available for investment")
    parser.add_argument("--reserved-cash", type=float, default=0.0,
                        help="Amount to keep as cash reserve")
    args = parser.parse_args()
    
    # Parse portfolio holdings
    portfolio = parse_excel(args.path)
    print("Portfolio Holdings:")
    print(portfolio)
    
    # Parse capital config if provided
    if args.total_capital is not None:
        capital = parse_capital_config(args.total_capital, args.reserved_cash)
        print(f"\nCapital Configuration:")
        print(f"  Total Capital:      ${capital.total_capital:,.2f}")
        print(f"  Reserved Cash:      ${capital.reserved_cash:,.2f}")
        print(f"  Investable Capital: ${capital.investable_capital:,.2f}")
        print(f"  Reserve Ratio:      {capital.reserve_ratio:.1%}")
    else:
        # Try to read capital from Excel file
        capital = parse_capital_from_excel(args.path)
        if capital:
            print(f"\nCapital Configuration (from Excel):")
            print(f"  Total Capital:      ${capital.total_capital:,.2f}")
            print(f"  Reserved Cash:      ${capital.reserved_cash:,.2f}")
            print(f"  Investable Capital: ${capital.investable_capital:,.2f}")

