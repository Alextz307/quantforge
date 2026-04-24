"""Tests for :func:`validate_pretrained_leaf` + :func:`normalize_pretrained_leaves`.

Pure-logic tests — no model fitting, no disk I/O. Each test builds a
tiny fake leaf object (duck-typed ``.training_metadata``) and exercises
one rejection class. Matches the rejection classes documented in
:func:`src.orchestration.pretrained_leaves.validate_pretrained_leaf`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.orchestration.pretrained_leaves import (
    normalize_pretrained_leaves,
    validate_pretrained_leaf,
)

_FEATURES: tuple[str, ...] = ("f0", "f1", "f2")
_INTERVAL = Interval.DAILY
_LSTM_LOOKBACK = 30
_MISMATCH_LOOKBACK = 45
_FAKE_N_TRAIN_SAMPLES = 1000


def _make_metadata(
    *,
    interval: Interval = _INTERVAL,
    feature_columns: tuple[str, ...] = _FEATURES,
) -> TrainingMetadata:
    return TrainingMetadata(
        train_start=pd.Timestamp("2020-01-02"),
        train_end=pd.Timestamp("2023-12-29"),
        n_train_samples=_FAKE_N_TRAIN_SAMPLES,
        fit_timestamp=pd.Timestamp("2024-01-01"),
        interval=interval,
        feature_columns=feature_columns,
    )


@dataclass
class _FakeLeaf:
    """Minimal duck-typed stand-in for an IPredictor / IClassifier leaf."""

    training_metadata: TrainingMetadata | None
    _lookback: int | None = None


@dataclass
class _FakeHybrid:
    """Composite stand-in exposing ``_lstm._lookback`` like ``HybridReturnModel``."""

    training_metadata: TrainingMetadata | None
    _lstm: _FakeLeaf


class TestValidatePretrainedLeaf:
    def test_happy_path_no_lookback_constraint(self) -> None:
        leaf = _FakeLeaf(training_metadata=_make_metadata())
        validate_pretrained_leaf(leaf, interval=_INTERVAL, feature_columns=_FEATURES)

    def test_happy_path_with_direct_lookback(self) -> None:
        leaf = _FakeLeaf(training_metadata=_make_metadata(), _lookback=_LSTM_LOOKBACK)
        validate_pretrained_leaf(
            leaf,
            interval=_INTERVAL,
            feature_columns=_FEATURES,
            lstm_lookback=_LSTM_LOOKBACK,
        )

    def test_happy_path_with_hybrid_lookback(self) -> None:
        """Hybrids expose their inner LSTM's lookback via ``_lstm._lookback``."""
        hybrid = _FakeHybrid(
            training_metadata=_make_metadata(),
            _lstm=_FakeLeaf(training_metadata=None, _lookback=_LSTM_LOOKBACK),
        )
        validate_pretrained_leaf(
            hybrid,
            interval=_INTERVAL,
            feature_columns=_FEATURES,
            lstm_lookback=_LSTM_LOOKBACK,
        )

    def test_missing_training_metadata_raises(self) -> None:
        leaf = _FakeLeaf(training_metadata=None)
        with pytest.raises(ValueError, match="has no training_metadata"):
            validate_pretrained_leaf(leaf, interval=_INTERVAL, feature_columns=_FEATURES)

    def test_interval_mismatch_raises(self) -> None:
        leaf = _FakeLeaf(training_metadata=_make_metadata(interval=Interval.HOUR))
        with pytest.raises(ValueError, match="interval mismatch"):
            validate_pretrained_leaf(leaf, interval=Interval.DAILY, feature_columns=_FEATURES)

    def test_feature_columns_mismatch_raises(self) -> None:
        leaf = _FakeLeaf(training_metadata=_make_metadata(feature_columns=("f0", "f1")))
        with pytest.raises(ValueError, match="feature_columns mismatch"):
            validate_pretrained_leaf(leaf, interval=_INTERVAL, feature_columns=_FEATURES)

    def test_feature_columns_reorder_mismatch_raises(self) -> None:
        """Tuple comparison catches reordering — order matters for the scaler."""
        leaf = _FakeLeaf(training_metadata=_make_metadata(feature_columns=("f2", "f0", "f1")))
        with pytest.raises(ValueError, match="feature_columns mismatch"):
            validate_pretrained_leaf(leaf, interval=_INTERVAL, feature_columns=_FEATURES)

    def test_lookback_mismatch_raises(self) -> None:
        leaf = _FakeLeaf(training_metadata=_make_metadata(), _lookback=_MISMATCH_LOOKBACK)
        with pytest.raises(ValueError, match="lstm_lookback mismatch"):
            validate_pretrained_leaf(
                leaf,
                interval=_INTERVAL,
                feature_columns=_FEATURES,
                lstm_lookback=_LSTM_LOOKBACK,
            )

    def test_non_lstm_leaf_skips_lookback_check(self) -> None:
        """Leaves without ``_lookback`` / ``_lstm`` silently pass even when
        the caller supplies ``lstm_lookback`` — ARMA and GARCH have no
        lookback contract to verify.
        """
        leaf = _FakeLeaf(training_metadata=_make_metadata(), _lookback=None)
        validate_pretrained_leaf(
            leaf,
            interval=_INTERVAL,
            feature_columns=_FEATURES,
            lstm_lookback=_LSTM_LOOKBACK,
        )


class TestNormalizePretrainedLeaves:
    _SUPPORTED = frozenset({"return_model", "vol_model"})

    def test_none_returns_empty_dict(self) -> None:
        assert normalize_pretrained_leaves(None, self._SUPPORTED, "Cls") == {}

    def test_empty_mapping_returns_empty_dict(self) -> None:
        assert normalize_pretrained_leaves({}, self._SUPPORTED, "Cls") == {}

    def test_valid_keys_round_trip_as_fresh_dict(self) -> None:
        source: dict[str, object] = {"return_model": object()}
        out = normalize_pretrained_leaves(source, self._SUPPORTED, "Cls")
        assert out == source
        source["vol_model"] = "extra"
        assert "vol_model" not in out  # defensive copy

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="does not own pretrained leaf"):
            normalize_pretrained_leaves({"classifier": object()}, self._SUPPORTED, "Cls")

    def test_empty_supported_rejects_any_key_with_targeted_message(self) -> None:
        with pytest.raises(ValueError, match="owns no ML leaves"):
            normalize_pretrained_leaves({"return_model": object()}, frozenset(), "NoopStrategy")
