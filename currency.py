"""Currency helper for formatting amounts and symbol.

Central place to set display currency (e.g. INR ₹). Other modules
should import `SYMBOL` or use `format_amount()` to render values.
"""
from typing import Union

# Change this to the desired currency symbol
SYMBOL = "₹"


def format_amount(x: Union[float, int], fmt: str = ",.2f") -> str:
    """Return a formatted currency string with symbol.

    Examples:
        format_amount(12345.6) -> '₹12,345.60'
    """
    try:
        return f"{SYMBOL}{x:{fmt}}"
    except Exception:
        return f"{SYMBOL}{x}"
