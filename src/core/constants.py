"""Domain constants for the quant trading framework.

Centralizes magic numbers used across the codebase.
"""

from __future__ import annotations

# ── Market calendar ──
TRADING_DAYS_PER_YEAR: int = 252
TRADING_WEEKS_PER_YEAR: int = 52
US_TRADING_MINUTES_PER_DAY: int = 390  # 6.5 hours
US_TRADING_SECONDS_PER_DAY: int = 23_400  # 390 * 60
US_TRADING_HOURS_PER_DAY: float = 6.5

# ── Position limits ──
MAX_LEVERAGE: float = 3.0
MIN_POSITION: float = -1.0  # Full short
MAX_POSITION: float = 3.0  # 3x leveraged long
