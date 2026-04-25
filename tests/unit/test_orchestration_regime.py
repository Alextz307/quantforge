"""Unit tests for the three regime detectors + regime_registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.orchestration.regime import (
    UNCLASSIFIED_LABEL,
    PeriodRegimeDetector,
    TrendRegimeDetector,
    VolatilityRegimeDetector,
    regime_registry,
)
from src.orchestration.types import RegimeKind, RegimeSlice

# Synthetic-data parameters shared across detector tests.
N_BARS = 600
START_DATE = "2019-06-01"
RNG_SEED = 7
TREND_WINDOW = 50
VOL_WINDOW = 20


def _bars_with_break(n: int, *, break_at: int) -> pd.DataFrame:
    """OHLCV with a clear regime break: rising then falling close.

    ``break_at`` is the bar index where the trend reverses. The first
    half drifts up at +0.4% per bar, the second half drifts down at
    -0.4% per bar — well over the threshold for the 50-bar MA flip.
    """
    rng = np.random.default_rng(RNG_SEED)
    dates = pd.date_range(start=START_DATE, periods=n, freq="B")
    drift = np.where(np.arange(n) < break_at, 0.004, -0.004)
    noise = rng.normal(0.0, 0.001, size=n)
    log_returns = drift + noise
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * 1.005
    low = close * 0.995
    open_ = close * (1.0 + rng.normal(0.0, 0.001, size=n))
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _bars_with_vol_step(n: int, *, step_at: int) -> pd.DataFrame:
    """OHLCV whose volatility steps up at ``step_at``.

    Low-vol regime first (sigma=0.001), high-vol regime second
    (sigma=0.02). The rolling-std detector should bucket the two halves
    into different quintiles.
    """
    rng = np.random.default_rng(RNG_SEED)
    dates = pd.date_range(start=START_DATE, periods=n, freq="B")
    sigmas = np.where(np.arange(n) < step_at, 0.001, 0.02)
    log_returns = rng.normal(0.0, sigmas, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = high = low = close
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# --- PeriodRegimeDetector ---------------------------------------------------


def test_period_detector_tags_each_bar_in_its_boundary() -> None:
    bars = _bars_with_break(N_BARS, break_at=N_BARS // 2)
    midpoint = bars.index[N_BARS // 2]
    detector = PeriodRegimeDetector(
        boundaries=[
            {"label": "first", "start": str(bars.index[0]), "end": str(midpoint)},
            {
                "label": "second",
                "start": str(midpoint),
                "end": str(bars.index[-1] + pd.Timedelta(days=1)),
            },
        ]
    )
    assert detector.kind is RegimeKind.PERIOD

    tagged = detector.tag(bars)
    assert tagged.iloc[0] == "first"
    assert tagged.iloc[N_BARS - 1] == "second"
    # Boundary day belongs to the second bucket (start inclusive, end exclusive)
    assert tagged.loc[midpoint] == "second"


def test_period_detector_unclassified_outside_boundaries() -> None:
    bars = _bars_with_break(N_BARS, break_at=N_BARS // 2)
    middle_start = bars.index[100]
    middle_end = bars.index[200]
    detector = PeriodRegimeDetector(
        boundaries=[
            {"label": "middle", "start": str(middle_start), "end": str(middle_end)},
        ]
    )
    tagged = detector.tag(bars)
    assert tagged.iloc[0] == UNCLASSIFIED_LABEL
    assert tagged.iloc[150] == "middle"
    assert tagged.iloc[N_BARS - 1] == UNCLASSIFIED_LABEL


def test_period_detector_rejects_overlapping_boundaries() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PeriodRegimeDetector(
            boundaries=[
                {"label": "a", "start": "2020-01-01", "end": "2021-01-01"},
                {"label": "b", "start": "2020-06-01", "end": "2022-01-01"},
            ]
        )


def test_period_detector_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="start >= end"):
        PeriodRegimeDetector(
            boundaries=[{"label": "x", "start": "2021-01-01", "end": "2020-01-01"}]
        )


def test_period_detector_slices_runlength_encoded() -> None:
    bars = _bars_with_break(N_BARS, break_at=N_BARS // 2)
    midpoint = bars.index[N_BARS // 2]
    detector = PeriodRegimeDetector(
        boundaries=[
            {"label": "first", "start": str(bars.index[0]), "end": str(midpoint)},
            {
                "label": "second",
                "start": str(midpoint),
                "end": str(bars.index[-1] + pd.Timedelta(days=1)),
            },
        ]
    )
    slices = detector.slices(bars)
    # Two contiguous regions → exactly two slices in chronological order
    assert [s.label for s in slices] == ["first", "second"]
    assert all(isinstance(s, RegimeSlice) for s in slices)
    assert slices[0].start <= slices[0].end
    assert slices[1].start <= slices[1].end


# --- TrendRegimeDetector ----------------------------------------------------


def test_trend_detector_warmup_unclassified() -> None:
    bars = _bars_with_break(N_BARS, break_at=N_BARS // 2)
    detector = TrendRegimeDetector(window=TREND_WINDOW)
    assert detector.kind is RegimeKind.TREND

    tagged = detector.tag(bars)
    # First (window-1) bars are unclassified; window-th onward have a label
    assert (tagged.iloc[: TREND_WINDOW - 1] == UNCLASSIFIED_LABEL).all()
    assert tagged.iloc[TREND_WINDOW - 1] in {"bull", "bear"}


def test_trend_detector_flips_after_break() -> None:
    bars = _bars_with_break(N_BARS, break_at=N_BARS // 2)
    detector = TrendRegimeDetector(window=TREND_WINDOW)
    tagged = detector.tag(bars)
    # Right before the break, close is well above MA → bull
    pre_break = tagged.iloc[(N_BARS // 2) - 1]
    # Well after the break, close drops below MA → bear (allow window for MA lag)
    post_break = tagged.iloc[(N_BARS // 2) + 2 * TREND_WINDOW]
    assert pre_break == "bull"
    assert post_break == "bear"


def test_trend_detector_rejects_short_window() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        TrendRegimeDetector(window=1)


def test_trend_detector_rejects_no_close_column() -> None:
    bars = pd.DataFrame({"open": [1.0]}, index=pd.date_range("2020-01-01", periods=1))
    detector = TrendRegimeDetector(window=TREND_WINDOW)
    with pytest.raises(ValueError, match="'close' column"):
        detector.tag(bars)


# --- VolatilityRegimeDetector -----------------------------------------------


def test_volatility_detector_separates_low_and_high_vol() -> None:
    bars = _bars_with_vol_step(N_BARS, step_at=N_BARS // 2)
    detector = VolatilityRegimeDetector(window=VOL_WINDOW, n_quantiles=5)
    assert detector.kind is RegimeKind.VOLATILITY

    tagged = detector.tag(bars)
    # First (VOL_WINDOW) bars are warmup (rolling std needs window bars + 1
    # for the diff). Skip the warmup, then count Q1/Q5 between halves.
    classified = tagged[tagged != UNCLASSIFIED_LABEL]
    first_half = classified.iloc[: len(classified) // 2]
    second_half = classified.iloc[len(classified) // 2 :]
    # Low-vol half should be dominated by Q1 / Q2; high-vol by Q4 / Q5.
    low_buckets = first_half.value_counts()
    high_buckets = second_half.value_counts()
    assert (low_buckets.get("Q1", 0) + low_buckets.get("Q2", 0)) > len(first_half) // 2
    assert (high_buckets.get("Q5", 0) + high_buckets.get("Q4", 0)) > len(second_half) // 2


def test_volatility_detector_rejects_short_window() -> None:
    with pytest.raises(ValueError, match=">= 2 bars"):
        VolatilityRegimeDetector(window=1)


def test_volatility_detector_rejects_few_quantiles() -> None:
    with pytest.raises(ValueError, match="n_quantiles"):
        VolatilityRegimeDetector(n_quantiles=1)


def test_volatility_detector_short_series_all_unclassified() -> None:
    # Series shorter than the warmup window — every bar unclassified, no
    # qcut crash.
    detector = VolatilityRegimeDetector(window=VOL_WINDOW)
    short = _bars_with_break(VOL_WINDOW - 1, break_at=1)
    tagged = detector.tag(short)
    assert (tagged == UNCLASSIFIED_LABEL).all()


# --- Registry ---------------------------------------------------------------


def test_regime_registry_contains_three_detectors() -> None:
    names = set(regime_registry.list_all())
    assert {"period", "trend", "volatility"} <= names


def test_regime_registry_creates_via_create_from_config() -> None:
    from src.core.config import ComponentConfig

    cfg = ComponentConfig(name="trend", params={"window": 100})
    detector = regime_registry.create_from_config(cfg)
    assert isinstance(detector, TrendRegimeDetector)
    assert detector.window == 100
