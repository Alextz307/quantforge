"""
Tests for ``IStrategy.get_all_training_metadata()`` + walk-forward deep check.

Covers the key invariants of the deep leakage tripwire:

* Each strategy's override returns the expected origin set + populated
  metadata after ``train()``.
* The walk-forward orchestrator translates a drift in any leaf's metadata
  into a ``LeakageError`` naming both the strategy class AND the failing
  origin (the reason ``TrackedMetadata`` carries the ``origin`` tag).
* Absent metadata on one component is logged + skipped rather than
  collapsing the whole check.

No composite ``train()`` is run with real ML leaves here — those paths are
covered by the existing composite test suites. This file targets the deep
check itself, using the cheap strategies (AdaptiveBollinger has GARCH but
is fast) plus targeted monkey-patching for the drift scenarios.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.temporal import TrackedMetadata, TrainingMetadata
from src.core.types import Interval
from src.engine.walk_forward import validate_deep_metadata
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from tests.conftest import make_synthetic_close_df

_COMPACT_GARCH_P = 2
_COMPACT_GARCH_Q = 2
_BOLLINGER_WINDOW = 20
_BOLLINGER_TREND = 50
_EVAL_N = 60
_EVAL_START = "2021-01-04"
_EVAL_SEED = 99


@pytest.fixture
def train_df() -> pd.DataFrame:
    return make_synthetic_close_df()


@pytest.fixture
def eval_df() -> pd.DataFrame:
    return make_synthetic_close_df(n_rows=_EVAL_N, start=_EVAL_START, seed=_EVAL_SEED)


class TestDefaultOverrideForSimpleStrategies:
    """
    PairsTradingStrategy inherits the default — metadata tuple length 1.
    """

    def test_pairs_trading_uses_default(self) -> None:
        s = PairsTradingStrategy()
        tracked = s.get_all_training_metadata()
        assert len(tracked) == 1
        assert tracked[0].origin == "strategy"
        assert tracked[0].metadata is None


class TestAdaptiveBollingerOverride:
    def test_exposes_strategy_and_garch(self, train_df: pd.DataFrame) -> None:
        s = AdaptiveBollingerStrategy(
            window=_BOLLINGER_WINDOW,
            trend_window=_BOLLINGER_TREND,
            garch_p_max=_COMPACT_GARCH_P,
            garch_q_max=_COMPACT_GARCH_Q,
        )
        s.train(train_df)
        tracked = s.get_all_training_metadata()
        origins = [t.origin for t in tracked]
        assert origins == ["strategy", "garch"]
        for t in tracked:
            assert isinstance(t.metadata, TrainingMetadata)
            assert t.metadata.train_end == pd.Timestamp(train_df.index[-1])


class TestWalkForwardDeepCheck:
    """
    validate_deep_metadata is the shared codepath invoked inside the fold loop.
    """

    def test_passes_when_eval_is_after_training(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> None:
        s = AdaptiveBollingerStrategy(
            window=_BOLLINGER_WINDOW,
            trend_window=_BOLLINGER_TREND,
            garch_p_max=_COMPACT_GARCH_P,
            garch_q_max=_COMPACT_GARCH_Q,
        )
        s.train(train_df)
        validate_deep_metadata(s, test_data=eval_df)

    def test_leaf_drift_raises_with_origin_in_message(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> None:
        """
        Simulate GARCH metadata drift: leaf's train_end lies AFTER eval.start.

        This models the failure mode the deep check exists to catch —
        ``strategy.training_metadata`` looks fine, but a wrapped model's
        metadata drifted forward so its training region overlaps the
        incoming fold's test window.
        """

        s = AdaptiveBollingerStrategy(
            window=_BOLLINGER_WINDOW,
            trend_window=_BOLLINGER_TREND,
            garch_p_max=_COMPACT_GARCH_P,
            garch_q_max=_COMPACT_GARCH_Q,
        )
        s.train(train_df)
        drifted = TrainingMetadata(
            train_start=pd.Timestamp(train_df.index[0]),
            train_end=pd.Timestamp(eval_df.index[5]),
            n_train_samples=len(train_df),
            fit_timestamp=pd.Timestamp("2025-01-01"),
            interval=Interval.DAILY,
            feature_columns=("close",),
        )
        s._garch._training_metadata = drifted

        with pytest.raises(LeakageError, match="AdaptiveBollingerStrategy.garch:"):
            validate_deep_metadata(s, test_data=eval_df)

    def test_none_component_is_logged_and_skipped(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        If one tracked component has None metadata but another has valid
        metadata, the check logs a warning for the None and validates the
        rest instead of collapsing the whole call.
        """

        class _StubStrategy:
            def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
                return (
                    TrackedMetadata(origin="strategy", metadata=None),
                    TrackedMetadata(
                        origin="garch",
                        metadata=TrainingMetadata(
                            train_start=pd.Timestamp(train_df.index[0]),
                            train_end=pd.Timestamp(train_df.index[-1]),
                            n_train_samples=len(train_df),
                            fit_timestamp=pd.Timestamp("2025-01-01"),
                            interval=Interval.DAILY,
                            feature_columns=("close",),
                        ),
                    ),
                )

        stub = _StubStrategy()
        with caplog.at_level(logging.WARNING):
            validate_deep_metadata(stub, test_data=eval_df)  # type: ignore[arg-type]
        assert any("strategy" in r.message for r in caplog.records)

    def test_all_none_raises_runtime_error(
        self,
        eval_df: pd.DataFrame,
    ) -> None:
        """
        Every tracked component being None is a contract violation —
        no part of the strategy has completed fit()."""

        class _StubStrategy:
            def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
                return (TrackedMetadata(origin="strategy", metadata=None),)

        stub = _StubStrategy()
        with pytest.raises(RuntimeError, match="returned no populated metadata"):
            validate_deep_metadata(stub, test_data=eval_df)  # type: ignore[arg-type]
