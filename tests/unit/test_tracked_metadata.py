"""
Tests for :class:`TrackedMetadata` and :func:`collect_metadata`.

These two primitives power the deep leakage check wired into
``evaluate_walk_forward``. They are tiny, but the ``None``-passthrough
contract is load-bearing — a regression here would silently let missing-fit
incidents escape the deep-check WARN log.
"""

from __future__ import annotations

import pandas as pd

from src.core.temporal import TrackedMetadata, TrainingMetadata, collect_metadata
from src.core.types import Interval

_TRAIN_START = "2020-01-02T00:00:00"
_TRAIN_END = "2022-12-30T00:00:00"
_FIT_TS = "2023-01-03T00:00:00"
_N_SAMPLES = 100


def _make_meta() -> TrainingMetadata:
    return TrainingMetadata(
        train_start=pd.Timestamp(_TRAIN_START),
        train_end=pd.Timestamp(_TRAIN_END),
        n_train_samples=_N_SAMPLES,
        fit_timestamp=pd.Timestamp(_FIT_TS),
        interval=Interval.DAILY,
        feature_columns=("close",),
    )


class TestTrackedMetadata:
    def test_is_frozen(self) -> None:
        tm = TrackedMetadata(origin="strategy", metadata=_make_meta())
        try:
            tm.origin = "other"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("TrackedMetadata must be frozen")

    def test_accepts_none_metadata(self) -> None:
        tm = TrackedMetadata(origin="lstm", metadata=None)
        assert tm.origin == "lstm"
        assert tm.metadata is None


class TestCollectMetadata:
    def test_bundles_pairs_in_order(self) -> None:
        meta_a = _make_meta()
        meta_b = _make_meta()
        out = collect_metadata(("strategy", meta_a), ("garch", meta_b))
        assert len(out) == 2
        assert out[0] == TrackedMetadata(origin="strategy", metadata=meta_a)
        assert out[1] == TrackedMetadata(origin="garch", metadata=meta_b)

    def test_preserves_none_metadata(self) -> None:
        out = collect_metadata(("strategy", _make_meta()), ("garch", None))
        assert len(out) == 2
        assert out[1].metadata is None

    def test_empty_call_returns_empty_tuple(self) -> None:
        assert collect_metadata() == ()
