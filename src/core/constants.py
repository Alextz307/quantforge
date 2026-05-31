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

ROC_QUARTER_PERIOD: int = 63
GARMAN_KLASS_WINDOW: int = 20
BOLLINGER_PERIOD: int = 20
BOLLINGER_NUM_STD: float = 2.0
ADX_PERIOD: int = 14
VOLUME_ZSCORE_WINDOW: int = 20
OBV_ZSCORE_WINDOW: int = 20

FEATURE_IMPORTANCE_RNG_SEED: int = 23
FEATURE_IMPORTANCE_N_REPEATS: int = 10

HPO_IMPORTANCE_FANOVA_SEED: int = 23

# The on-demand importance backfill re-runs a finished run's frozen config and
# compares aggregated metrics against the original to decide whether to attach
# importance in place (reproduced) or save it as a separate run (diverged). The
# tolerance separates two regimes. Benign run-to-run noise that does NOT mean
# different models - multithreaded-BLAS summation order, and the GARCH/ARMA MLE
# in the hybrids converging within its own optimizer tolerance - perturbs the
# metrics up to ~1e-6 even on the same machine and seed. A genuinely different
# fit (a non-deterministic accelerator-trained LSTM, or a flipped discrete
# signal) moves them by ~1e-4 or more. 1e-6 sits in the gap: it absorbs the
# former so a faithful re-run still backfills, and flags the latter as a
# separate run. Tighter (1e-9) would spuriously diverge every hybrid recompute;
# looser (1e-3) would mask small genuine divergences.
IMPORTANCE_REPRODUCTION_RTOL: float = 1e-6

# Absolute floor for the same comparison: near-zero metrics (a flat strategy's
# return) make the relative tolerance collapse to ~0, so without a floor benign
# noise there would spuriously diverge. The all-keys conjunction in
# _metrics_reproduced keeps this floor from ever masking a real divergence.
IMPORTANCE_REPRODUCTION_ABS_TOL: float = 1e-6

OHLCV_COLUMNS: tuple[str, str, str, str, str] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PAIRS_LEG_SUFFIXES: tuple[str, str] = ("_a", "_b")

NYSE_CALENDAR_NAME: str = "NYSE"
