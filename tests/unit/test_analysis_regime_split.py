"""Unit tests for ``split_folds_by_regime``.

Tests cover the four real cases:
* clean fold (one regime covers ≥60%)
* straddling fold (boundary split below threshold → mixed)
* all-mixed run (every fold is on a boundary)
* unclassified-warmup fold (trend / vol detector emits ``unclassified``
  for early bars)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.regime_split import split_folds_by_regime
from src.orchestration.regime import (
    PeriodRegimeDetector,
    TrendRegimeDetector,
)
from src.orchestration.types import FoldRecord

N_BARS = 600
START_DATE = "2019-06-01"
WARMUP_TREND_WINDOW = 50


def _bars(n: int = N_BARS) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range(start=START_DATE, periods=n, freq="B")
    drift = np.where(np.arange(n) < n // 2, 0.004, -0.004)
    noise = rng.normal(0.0, 0.001, size=n)
    log_returns = drift + noise
    close = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


def _make_fold(
    fold_index: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> FoldRecord:
    """Minimal fold record with the only fields the splitter cares about."""
    return FoldRecord(
        fold_index=fold_index,
        train_start=test_start - pd.Timedelta(days=200),
        train_end=test_start,
        test_start=test_start,
        test_end=test_end,
        total_return=0.01,
        annualized_return=0.05,
        annualized_volatility=0.15,
        sharpe_ratio=0.5,
        sortino_ratio=0.6,
        calmar_ratio=0.4,
        max_drawdown=-0.05,
        win_rate=0.55,
        trade_count=10,
        equity_curve=(1.0, 1.01),
    )


def _two_period_detector(bars: pd.DataFrame) -> PeriodRegimeDetector:
    midpoint = bars.index[len(bars) // 2]
    end = bars.index[-1] + pd.Timedelta(days=1)
    return PeriodRegimeDetector(
        boundaries=[
            {"label": "early", "start": str(bars.index[0]), "end": str(midpoint)},
            {"label": "late", "start": str(midpoint), "end": str(end)},
        ]
    )


def test_clean_fold_assigned_to_dominant_regime() -> None:
    bars = _bars()
    detector = _two_period_detector(bars)
    midpoint = bars.index[N_BARS // 2]
    # Test window entirely inside the "early" period.
    fold = _make_fold(
        fold_index=0,
        test_start=bars.index[10],
        test_end=midpoint - pd.Timedelta(days=1),
    )
    result = split_folds_by_regime((fold,), detector, bars)
    assert result.per_regime["early"] == (fold,)
    assert result.mixed == ()


def test_straddling_fold_below_threshold_goes_to_mixed() -> None:
    bars = _bars()
    detector = _two_period_detector(bars)
    midpoint = bars.index[N_BARS // 2]
    # Test window centered on the midpoint, ~50/50 split → below 60%.
    span = pd.Timedelta(days=40)
    fold = _make_fold(
        fold_index=0,
        test_start=midpoint - span,
        test_end=midpoint + span,
    )
    result = split_folds_by_regime((fold,), detector, bars)
    assert fold not in sum(result.per_regime.values(), ())
    assert result.mixed == (fold,)


def test_dominant_above_threshold_assigns_majority() -> None:
    bars = _bars()
    detector = _two_period_detector(bars)
    midpoint = bars.index[N_BARS // 2]
    # 80% of test window in "early", 20% in "late" → above 60% → "early".
    fold = _make_fold(
        fold_index=0,
        test_start=midpoint - pd.Timedelta(days=80),
        test_end=midpoint + pd.Timedelta(days=20),
    )
    result = split_folds_by_regime((fold,), detector, bars)
    assert fold in result.per_regime["early"]
    assert result.mixed == ()


def test_unclassified_warmup_excluded_from_majority_count() -> None:
    """A fold whose test window opens during trend detector's warmup (no MA)
    should still get classified by the bars that ARE classified.
    """
    bars = _bars()
    detector = TrendRegimeDetector(window=WARMUP_TREND_WINDOW)
    # Test window starts inside the warmup region but extends past it; the
    # classified bars all sit in the "bull" rising-drift region.
    fold = _make_fold(
        fold_index=0,
        test_start=bars.index[WARMUP_TREND_WINDOW - 10],
        test_end=bars.index[WARMUP_TREND_WINDOW + 80],
    )
    result = split_folds_by_regime((fold,), detector, bars)
    assert "bull" in result.per_regime
    assert fold in result.per_regime["bull"]


def test_all_unclassified_window_lands_in_mixed() -> None:
    """A fold whose test window is entirely inside the warmup region
    cannot be classified at all — it goes to mixed.
    """
    bars = _bars()
    detector = TrendRegimeDetector(window=WARMUP_TREND_WINDOW)
    fold = _make_fold(
        fold_index=0,
        test_start=bars.index[0],
        test_end=bars.index[WARMUP_TREND_WINDOW - 5],
    )
    result = split_folds_by_regime((fold,), detector, bars)
    assert result.mixed == (fold,)
    assert all(fold not in folds for folds in result.per_regime.values())


def test_empty_window_raises_value_error() -> None:
    bars = _bars()
    detector = _two_period_detector(bars)
    # Test window outside the bars range entirely.
    fold = _make_fold(
        fold_index=0,
        test_start=bars.index[-1] + pd.Timedelta(days=10),
        test_end=bars.index[-1] + pd.Timedelta(days=20),
    )
    with pytest.raises(ValueError, match="zero bars"):
        split_folds_by_regime((fold,), detector, bars)


def test_threshold_validates_above_half() -> None:
    bars = _bars()
    detector = _two_period_detector(bars)
    fold = _make_fold(
        fold_index=0,
        test_start=bars.index[0],
        test_end=bars.index[100],
    )
    with pytest.raises(ValueError, match="majority_threshold"):
        split_folds_by_regime((fold,), detector, bars, majority_threshold=0.4)


def test_split_result_per_regime_keys_only_for_seen_regimes() -> None:
    bars = _bars()
    detector = _two_period_detector(bars)
    # Fold entirely inside "early" → only "early" key in per_regime.
    fold = _make_fold(
        fold_index=0,
        test_start=bars.index[10],
        test_end=bars.index[100],
    )
    result = split_folds_by_regime((fold,), detector, bars)
    assert set(result.per_regime.keys()) == {"early"}
