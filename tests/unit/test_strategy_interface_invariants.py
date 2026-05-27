"""Invariants for :class:`IStrategy`'s atomic fitted-with-metadata pair.

The pair (:meth:`_set_fitted_with_metadata` / :meth:`_assert_fitted_with_metadata`)
is the only legal commit/observe path for the ``_fitted`` + ``_training_metadata``
state. These tests pin the contract directly on the abstract base so a
regression in any concrete strategy can't quietly hide behind that strategy's
own ``train()`` body.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.strategies.interface import IStrategy

_FIT_FEATURE_COLUMNS: tuple[str, ...] = ("close",)
_FIT_BAR_COUNT = 5
_FIT_START_DATE = "2024-01-02"


class _BareStrategy(IStrategy):
    """Minimal concrete subclass so we can instantiate the abstract base."""

    def train(self, train_data, *, checkpoint_path=None, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def generate_signals(self, data):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "bare"

    @property
    def required_warmup_bars(self) -> int:
        return 0

    @staticmethod
    def suggest_params(trial):  # type: ignore[no-untyped-def]
        return {}


def _make_metadata() -> TrainingMetadata:
    idx = pd.date_range(_FIT_START_DATE, periods=_FIT_BAR_COUNT, freq="B")
    df = pd.DataFrame({"close": range(_FIT_BAR_COUNT)}, index=idx)
    return TrainingMetadata.from_fit(df, Interval.DAILY, _FIT_FEATURE_COLUMNS)


def test_set_fitted_with_metadata_populates_metadata_slot() -> None:
    s = _BareStrategy()
    assert s.training_metadata is None

    meta = _make_metadata()
    s._set_fitted_with_metadata(meta)

    assert s.training_metadata is meta
    assert s._assert_fitted_with_metadata(caller="test") is meta


def test_set_fitted_with_metadata_rejects_none() -> None:
    s = _BareStrategy()
    with pytest.raises(ValueError, match="non-None TrainingMetadata"):
        s._set_fitted_with_metadata(None)  # type: ignore[arg-type]
    # The defensive raise must leave the slot untouched — a partial state would
    # let a downstream reader see a fitted-looking object with no metadata.
    assert s.training_metadata is None
