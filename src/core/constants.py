"""
Domain constants for the quant trading framework.

Centralizes magic numbers used across the codebase.
"""

from __future__ import annotations

TRADING_DAYS_PER_YEAR: int = 252
TRADING_WEEKS_PER_YEAR: int = 52
US_TRADING_MINUTES_PER_DAY: int = 390
US_TRADING_SECONDS_PER_DAY: int = 23_400
US_TRADING_HOURS_PER_DAY: float = 6.5

MAX_LEVERAGE: float = 3.0
MIN_POSITION: float = -1.0
MAX_POSITION: float = 3.0

DEFAULT_REALIZED_VOL_WINDOW: int = 20

OHLCV_COLUMNS: tuple[str, str, str, str, str] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PAIRS_LEG_SUFFIXES: tuple[str, str] = ("_a", "_b")

NYSE_CALENDAR_NAME: str = "NYSE"
