"""
Unit tests for the feature-importance subsystem.

Covers the scoring primitives (``directional_accuracy``, ``negative_qlike``),
the permutation driver + XGBoost-gain wrapper, cross-fold aggregation, JSON
round-trips, and the ``compute_fold_importance`` orchestration via a stub
strategy that exposes the importance hooks without any heavy ML training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from quant_engine import SlippageConfig, SlippageModel
from src.analysis.feature_importance import (
    AggregatedImportance,
    FeatureImportance,
    FoldImportance,
    ImportanceMethod,
    aggregate_fold_importance,
    build_importance_artifact,
    compute_fold_importance,
    permutation_importance,
    read_aggregated_importance,
    xgb_gain_importance,
)
from src.core.temporal import WalkForwardValidator
from src.core.types import Interval
from src.core.utils import (
    compute_log_returns,
    directional_accuracy,
    negative_log_loss,
    negative_qlike,
    negative_return_mse,
)
from src.engine.cpp_engine import CppBacktestEngine
from src.engine.walk_forward import evaluate_walk_forward
from src.strategies.interface import IStrategy
from tests.conftest import make_synthetic_ohlcv_df

if TYPE_CHECKING:
    import optuna

_WF_N_SPLITS = 2
_WF_TEST_SIZE = 40
_WF_GAP = 1
_WF_FEATURE = "ret_feat"

_N_ROWS = 240
_START = "2020-01-02"
_SEED = 23
_N_REPEATS = 10
_SIGNAL_COL = "signal_feat"
_NOISE_COL = "noise_feat"
_INFORMATIVE_MIN_IMPORTANCE = 0.20
_NOISE_MAX_ABS_IMPORTANCE = 0.05
_PERFECT_VOL = 0.2
_WORSE_VOL = 0.6
_INTERIOR_NAN_ROW = 100


def _direction_frame(seed: int = _SEED) -> pd.DataFrame:
    """
    Build a frame whose ``signal_feat`` perfectly predicts the next-bar move.

    ``close`` is a random walk; ``signal_feat`` is ``+1`` when the next bar
    rises and ``-1`` otherwise (a deterministic informative feature for the
    test); ``noise_feat`` is independent noise. Permuting ``signal_feat`` must
    destroy directional accuracy; permuting ``noise_feat`` must not.
    """

    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=_START, periods=_N_ROWS, freq="B")
    returns = rng.normal(0.0005, 0.01, _N_ROWS)
    close = 100.0 * np.cumprod(1.0 + returns)
    next_up = np.empty(_N_ROWS)
    next_up[:-1] = np.where(close[1:] > close[:-1], 1.0, -1.0)
    next_up[-1] = 0.0  # no t+1 bar; dropped by next_bar_direction alignment
    return pd.DataFrame(
        {_SIGNAL_COL: next_up, _NOISE_COL: rng.normal(size=_N_ROWS), "close": close},
        index=idx,
    )


class _StubImportanceStrategy(IStrategy):
    """
    Minimal strategy exposing the importance hooks with no ML training.

    ``score`` is the directional accuracy of the ``signal_feat`` column
    against the realised next-bar move, so the importance driver can verify
    that permuting an informative column drops the score.
    """

    def __init__(
        self,
        columns: tuple[str, ...],
        *,
        gain: dict[str, float] | None = None,
    ) -> None:
        self._columns = columns
        self._gain = gain

    def train(
        self, train_data: pd.DataFrame, *, checkpoint_path: Path | None = None, **kwargs: object
    ) -> None:
        from src.core.temporal import TrainingMetadata

        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(train_data, Interval.DAILY, self._columns or (_SIGNAL_COL,))
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=data.index)

    @property
    def name(self) -> str:
        return "Stub"

    @property
    def required_warmup_bars(self) -> int:
        return 0

    @staticmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
        return {}

    def feature_columns(self) -> tuple[str, ...]:
        return self._columns

    def feature_importance_frame(self, data: pd.DataFrame) -> pd.DataFrame | None:
        if not self._columns:
            return None
        return data

    def feature_importance_score(self, frame: pd.DataFrame) -> float | None:
        if not self._columns:
            return None
        return directional_accuracy(frame[_SIGNAL_COL], frame["close"])

    def feature_gain(self) -> dict[str, float] | None:
        return self._gain


class _NoFrameStrategy(_StubImportanceStrategy):
    """
    Declares feature columns but never materialises a frame (unfit model).
    """

    def feature_importance_frame(self, data: pd.DataFrame) -> pd.DataFrame | None:
        return None


class _NanScoreStrategy(_StubImportanceStrategy):
    """
    Materialises a frame but yields a non-finite baseline score (degenerate fold).
    """

    def feature_importance_score(self, frame: pd.DataFrame) -> float | None:
        return float("nan")


class _RecordingStrategy(_StubImportanceStrategy):
    """
    Records the index of every frame it is asked to score, for contiguity checks.
    """

    def __init__(self, columns: tuple[str, ...]) -> None:
        super().__init__(columns)
        self.scored_indices: list[pd.Index] = []

    def feature_importance_score(self, frame: pd.DataFrame) -> float | None:
        self.scored_indices.append(frame.index)
        return super().feature_importance_score(frame)


def test_directional_accuracy_perfect_and_anti() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([1.0, 2.0, 1.5, 2.5, 2.0, 3.0], index=idx, dtype=float)
    # next_bar_direction: up, down, up, down, up (last row dropped)
    perfect = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, 0.0], index=idx, dtype=float)
    anti = -perfect
    assert directional_accuracy(perfect, close) == 1.0
    assert directional_accuracy(anti, close) == 0.0


def test_directional_accuracy_drops_warmup_nan() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([1.0, 2.0, 1.5, 2.5, 2.0, 3.0], index=idx, dtype=float)
    # Leading NaN must be dropped, not coerced to a "down" call at 0.
    pred = pd.Series([float("nan"), -1.0, 1.0, -1.0, 1.0, 0.0], index=idx, dtype=float)
    # remaining rows (idx 1..4) all correct -> accuracy 1.0
    assert directional_accuracy(pred, close) == 1.0


def test_directional_accuracy_empty_is_nan() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    close = pd.Series([1.0, 2.0], index=idx, dtype=float)
    pred = pd.Series([float("nan"), float("nan")], index=idx, dtype=float)
    assert math.isnan(directional_accuracy(pred, close))


def test_negative_qlike_perfect_beats_worse() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    realised = pd.Series(_PERFECT_VOL, index=idx)
    perfect = pd.Series(_PERFECT_VOL, index=idx)
    worse = pd.Series(_WORSE_VOL, index=idx)
    score_perfect = negative_qlike(perfect, realised)
    score_worse = negative_qlike(worse, realised)
    assert math.isfinite(score_perfect)
    assert score_perfect > score_worse


def test_negative_qlike_all_nan_is_nan() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    nan_series = pd.Series([float("nan")] * 3, index=idx, dtype=float)
    assert math.isnan(negative_qlike(nan_series, nan_series))


def test_negative_return_mse_perfect_beats_biased() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([1.0, 1.1, 1.0, 1.2, 1.1, 1.3], index=idx, dtype=float)
    realised_next = compute_log_returns(close).shift(-1).reindex(idx)
    score_perfect = negative_return_mse(realised_next, close)
    score_biased = negative_return_mse(realised_next + 0.05, close)
    assert score_perfect == 0.0
    assert score_perfect > score_biased


def test_negative_return_mse_empty_is_nan() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    close = pd.Series([1.0, 2.0], index=idx, dtype=float)
    forecast = pd.Series([float("nan"), float("nan")], index=idx, dtype=float)
    assert math.isnan(negative_return_mse(forecast, close))


def test_negative_return_mse_rejects_stray_label() -> None:
    close = pd.Series(
        [1.0, 1.1, 1.2], index=pd.date_range("2020-01-01", periods=3, freq="D"), dtype=float
    )
    forecast = pd.Series([0.0], index=pd.date_range("2030-01-01", periods=1, freq="D"), dtype=float)
    with pytest.raises(ValueError, match="absent from close"):
        negative_return_mse(forecast, close)


def test_negative_log_loss_confident_correct_beats_wrong() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([1.0, 2.0, 1.5, 2.5, 2.0, 3.0], index=idx, dtype=float)
    # next_bar_direction: up, down, up, down, up (last row dropped).
    confident_right = pd.Series([0.99, 0.01, 0.99, 0.01, 0.99, 0.5], index=idx, dtype=float)
    confident_wrong = pd.Series([0.01, 0.99, 0.01, 0.99, 0.01, 0.5], index=idx, dtype=float)
    score_right = negative_log_loss(confident_right, close)
    score_wrong = negative_log_loss(confident_wrong, close)
    assert math.isfinite(score_right)
    assert score_right > score_wrong


def test_negative_log_loss_clips_certain_miss_to_finite() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close = pd.Series([1.0, 2.0, 1.5], index=idx, dtype=float)
    # next_bar_direction at t=0,1 is up,down; a p=1.0 call on the down bar would
    # be log(0) without the clip.
    prob = pd.Series([1.0, 1.0, 0.5], index=idx, dtype=float)
    assert math.isfinite(negative_log_loss(prob, close))


def test_negative_log_loss_empty_is_nan() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    close = pd.Series([1.0, 2.0], index=idx, dtype=float)
    prob = pd.Series([float("nan"), float("nan")], index=idx, dtype=float)
    assert math.isnan(negative_log_loss(prob, close))


def test_permutation_importance_separates_signal_from_noise() -> None:
    frame = _direction_frame()
    strategy = _StubImportanceStrategy((_SIGNAL_COL, _NOISE_COL))

    def scorer(f: pd.DataFrame) -> float:
        value = strategy.feature_importance_score(f)
        assert value is not None
        return value

    results = permutation_importance(
        scorer,
        frame,
        (_SIGNAL_COL, _NOISE_COL),
        n_repeats=_N_REPEATS,
        rng=np.random.default_rng(_SEED),
    )
    by_feature = {r.feature: r for r in results}
    assert by_feature[_SIGNAL_COL].importance > _INFORMATIVE_MIN_IMPORTANCE
    assert abs(by_feature[_NOISE_COL].importance) < _NOISE_MAX_ABS_IMPORTANCE
    assert all(r.method is ImportanceMethod.PERMUTATION for r in results)


def test_permutation_importance_is_seed_reproducible() -> None:
    frame = _direction_frame()
    strategy = _StubImportanceStrategy((_SIGNAL_COL, _NOISE_COL))

    def scorer(f: pd.DataFrame) -> float:
        value = strategy.feature_importance_score(f)
        assert value is not None
        return value

    first = permutation_importance(
        scorer, frame, (_SIGNAL_COL, _NOISE_COL), n_repeats=_N_REPEATS, rng=np.random.default_rng(7)
    )
    second = permutation_importance(
        scorer, frame, (_SIGNAL_COL, _NOISE_COL), n_repeats=_N_REPEATS, rng=np.random.default_rng(7)
    )
    assert [r.importance for r in first] == [r.importance for r in second]


def test_permutation_importance_does_not_mutate_input() -> None:
    frame = _direction_frame()
    before = frame.copy(deep=True)
    strategy = _StubImportanceStrategy((_SIGNAL_COL, _NOISE_COL))

    def scorer(f: pd.DataFrame) -> float:
        value = strategy.feature_importance_score(f)
        assert value is not None
        return value

    permutation_importance(
        scorer, frame, (_SIGNAL_COL, _NOISE_COL), n_repeats=_N_REPEATS, rng=np.random.default_rng(1)
    )
    pd.testing.assert_frame_equal(frame, before)


def test_xgb_gain_importance_fills_unsplit_columns() -> None:
    results = xgb_gain_importance({_SIGNAL_COL: 7.5}, (_SIGNAL_COL, _NOISE_COL))
    by_feature = {r.feature: r for r in results}
    assert by_feature[_SIGNAL_COL].importance == 7.5
    assert by_feature[_NOISE_COL].importance == 0.0
    assert all(r.method is ImportanceMethod.XGB_GAIN for r in results)
    assert len(results) == 2


def test_compute_fold_importance_runs_permutation_and_gain() -> None:
    frame = _direction_frame()
    strategy = _StubImportanceStrategy((_SIGNAL_COL, _NOISE_COL), gain={_SIGNAL_COL: 3.0})
    fold = compute_fold_importance(
        strategy, frame, fold_index=0, n_repeats=_N_REPEATS, rng=np.random.default_rng(_SEED)
    )
    assert fold is not None
    methods = {s.method for s in fold.scores}
    assert methods == {ImportanceMethod.PERMUTATION, ImportanceMethod.XGB_GAIN}
    perm = {
        s.feature: s.importance for s in fold.scores if s.method is ImportanceMethod.PERMUTATION
    }
    assert perm[_SIGNAL_COL] > _INFORMATIVE_MIN_IMPORTANCE


def test_compute_fold_importance_skips_rule_based() -> None:
    frame = _direction_frame()
    strategy = _StubImportanceStrategy(())
    fold = compute_fold_importance(
        strategy, frame, fold_index=0, n_repeats=_N_REPEATS, rng=np.random.default_rng(_SEED)
    )
    assert fold is None


def test_aggregate_fold_importance_means_across_folds() -> None:
    scores_a = (
        FeatureImportance(_SIGNAL_COL, 0.2, 0.01, ImportanceMethod.PERMUTATION),
        FeatureImportance(_NOISE_COL, 0.0, 0.01, ImportanceMethod.PERMUTATION),
    )
    scores_b = (
        FeatureImportance(_SIGNAL_COL, 0.4, 0.01, ImportanceMethod.PERMUTATION),
        FeatureImportance(_NOISE_COL, 0.0, 0.01, ImportanceMethod.PERMUTATION),
    )
    aggregated = aggregate_fold_importance(
        [FoldImportance(0, scores_a), FoldImportance(1, scores_b)]
    )
    by_feature = {a.feature: a for a in aggregated}
    assert by_feature[_SIGNAL_COL].importance == pytest.approx(0.3)
    assert by_feature[_SIGNAL_COL].n_folds == 2
    assert by_feature[_SIGNAL_COL].std > 0.0
    # sorted by descending importance within method
    assert aggregated[0].importance >= aggregated[-1].importance


def test_feature_importance_dataclass_round_trip() -> None:
    fi = FeatureImportance(_SIGNAL_COL, 0.42, 0.03, ImportanceMethod.PERMUTATION)
    assert FeatureImportance.from_dict(fi.to_dict()) == fi

    fold = FoldImportance(
        2, (fi, FeatureImportance(_NOISE_COL, 0.0, 0.0, ImportanceMethod.XGB_GAIN))
    )
    assert FoldImportance.from_dict(fold.to_dict()) == fold

    agg = AggregatedImportance(_SIGNAL_COL, 0.3, 0.05, 4, ImportanceMethod.PERMUTATION)
    assert AggregatedImportance.from_dict(agg.to_dict()) == agg


def test_build_and_read_importance_artifact_round_trip() -> None:
    scores = (
        FeatureImportance(_SIGNAL_COL, 0.25, 0.02, ImportanceMethod.PERMUTATION),
        FeatureImportance(_NOISE_COL, 0.01, 0.02, ImportanceMethod.PERMUTATION),
    )
    artifact = build_importance_artifact([FoldImportance(0, scores), FoldImportance(1, scores)])
    assert artifact["n_folds"] == 2
    aggregated = read_aggregated_importance(artifact)
    by_feature = {a.feature: a for a in aggregated}
    assert by_feature[_SIGNAL_COL].importance == 0.25
    assert by_feature[_SIGNAL_COL].n_folds == 2


class _OHLCVImportanceStrategy(_StubImportanceStrategy):
    """
    Walk-forward stub: builds one feature from raw OHLCV ``close`` internally.

    Mirrors the classifier-strategy plumbing where the importance frame is
    materialised from raw bars rather than read straight from the input.
    """

    def __init__(self) -> None:
        super().__init__((_WF_FEATURE,))

    def feature_importance_frame(self, data: pd.DataFrame) -> pd.DataFrame | None:
        frame = pd.DataFrame(index=data.index)
        frame[_WF_FEATURE] = data["close"].pct_change()
        frame["close"] = data["close"]
        return frame

    def feature_importance_score(self, frame: pd.DataFrame) -> float | None:
        return directional_accuracy(frame[_WF_FEATURE], frame["close"])


def _zero_slippage() -> SlippageConfig:
    return SlippageConfig(model=SlippageModel.NoSlippage, base_bps=0.0, volume_impact_coeff=0.0)


def test_walk_forward_attaches_importance_when_enabled() -> None:
    bars = make_synthetic_ohlcv_df()
    validator = WalkForwardValidator(n_splits=_WF_N_SPLITS, test_size=_WF_TEST_SIZE, gap=_WF_GAP)
    results = evaluate_walk_forward(
        strategy=_OHLCVImportanceStrategy(),
        bars=bars,
        validator=validator,
        engine=CppBacktestEngine(),
        slippage=_zero_slippage(),
        interval=Interval.DAILY,
        compute_feature_importance=True,
    )
    assert len(results) == _WF_N_SPLITS
    for fold in results:
        assert fold.feature_importance is not None
        assert {s.feature for s in fold.feature_importance.scores} == {_WF_FEATURE}


def test_walk_forward_omits_importance_by_default() -> None:
    bars = make_synthetic_ohlcv_df()
    validator = WalkForwardValidator(n_splits=_WF_N_SPLITS, test_size=_WF_TEST_SIZE, gap=_WF_GAP)
    results = evaluate_walk_forward(
        strategy=_OHLCVImportanceStrategy(),
        bars=bars,
        validator=validator,
        engine=CppBacktestEngine(),
        slippage=_zero_slippage(),
        interval=Interval.DAILY,
    )
    assert all(fold.feature_importance is None for fold in results)


def test_permutation_importance_raises_on_nonpositive_repeats() -> None:
    frame = _direction_frame()
    strategy = _StubImportanceStrategy((_SIGNAL_COL, _NOISE_COL))

    def scorer(f: pd.DataFrame) -> float:
        value = strategy.feature_importance_score(f)
        assert value is not None
        return value

    with pytest.raises(ValueError, match="n_repeats"):
        permutation_importance(
            scorer, frame, (_SIGNAL_COL,), n_repeats=0, rng=np.random.default_rng(_SEED)
        )


def test_compute_fold_importance_skips_when_frame_none() -> None:
    strategy = _NoFrameStrategy((_SIGNAL_COL,))
    fold = compute_fold_importance(
        strategy,
        _direction_frame(),
        fold_index=0,
        n_repeats=_N_REPEATS,
        rng=np.random.default_rng(_SEED),
    )
    assert fold is None


def test_compute_fold_importance_skips_when_no_valid_rows() -> None:
    frame = _direction_frame()
    frame[_SIGNAL_COL] = np.nan
    strategy = _StubImportanceStrategy((_SIGNAL_COL,))
    fold = compute_fold_importance(
        strategy, frame, fold_index=0, n_repeats=_N_REPEATS, rng=np.random.default_rng(_SEED)
    )
    assert fold is None


def test_compute_fold_importance_skips_when_baseline_nan() -> None:
    strategy = _NanScoreStrategy((_SIGNAL_COL,))
    fold = compute_fold_importance(
        strategy,
        _direction_frame(),
        fold_index=0,
        n_repeats=_N_REPEATS,
        rng=np.random.default_rng(_SEED),
    )
    assert fold is None


def test_compute_fold_importance_uses_contiguous_tail_with_interior_nan() -> None:
    frame = _direction_frame()
    # Interior NaN must leave a CONTIGUOUS tail, not a hole: a boolean mask would
    # drop this row and mis-pair the close-derived target's shift(-1) at the gap.
    frame.loc[frame.index[_INTERIOR_NAN_ROW], _NOISE_COL] = np.nan
    strategy = _RecordingStrategy((_SIGNAL_COL, _NOISE_COL))
    fold = compute_fold_importance(
        strategy, frame, fold_index=0, n_repeats=_N_REPEATS, rng=np.random.default_rng(_SEED)
    )
    assert fold is not None
    # The first scored frame (the baseline) spans the full contiguous index,
    # interior-NaN row retained - not the holey subset a boolean mask yields.
    assert strategy.scored_indices[0].equals(frame.index)
    perm = {
        s.feature: s.importance for s in fold.scores if s.method is ImportanceMethod.PERMUTATION
    }
    assert perm[_SIGNAL_COL] > _INFORMATIVE_MIN_IMPORTANCE


def test_score_nan_serializes_as_null_and_round_trips() -> None:
    fi = FeatureImportance(_SIGNAL_COL, float("nan"), float("nan"), ImportanceMethod.PERMUTATION)
    payload = fi.to_dict()
    assert payload["importance"] is None
    assert payload["std"] is None
    assert "NaN" not in json.dumps(payload)
    restored = FeatureImportance.from_dict(payload)
    assert math.isnan(restored.importance)
    assert math.isnan(restored.std)


def test_score_inf_serializes_as_null() -> None:
    fi = FeatureImportance(_SIGNAL_COL, float("inf"), float("-inf"), ImportanceMethod.PERMUTATION)
    payload = fi.to_dict()
    assert payload["importance"] is None
    assert payload["std"] is None
    assert "Infinity" not in json.dumps(payload)
    restored = FeatureImportance.from_dict(payload)
    assert math.isnan(restored.importance)
    assert math.isnan(restored.std)
