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

# ── Realized-volatility estimator ──
# Window (in bars) for the Garman-Klass realized-vol target. Shared by
# ``VolatilityTargetingStrategy`` (ctor default) and the standalone
# training dispatcher so the two callsites cannot silently drift apart.
DEFAULT_REALIZED_VOL_WINDOW: int = 20

# ── Bar shape ──
# Canonical OHLCV column ordering. The tuple form is load-bearing in the
# engine adapter (`_bars_to_arrays` indexes by position); set/dict consumers
# can wrap in `set(...)` at the call site.
OHLCV_COLUMNS: tuple[str, str, str, str, str] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)
