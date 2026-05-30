"""
Model-agnostic feature-importance for feature-consuming strategies.

Two complementary methods:

* **Permutation importance** (out-of-sample) - shuffle one feature column
  at a time on the OOS test frame and measure how far a strategy-supplied,
  higher-is-better score drops. Model-agnostic: it only needs the strategy
  to re-score a frame, never reaches into model internals, so frozen GARCH /
  ARMA params and fit-once scalers keep their anti-leakage guarantees.
* **XGBoost native gain** - the booster's average loss reduction per feature,
  for the directional-classifier strategies only.

Anti-leakage contract:

* Importance is computed on the OOS test frame ONLY, with an already-fitted
  (frozen) model. The caller (walk-forward loop) is responsible for passing
  the test frame, never the train frame.
* Permutation shuffles values WITHIN the frame - no value from outside the
  frame is introduced, and only declared feature columns are touched. The
  realised target each score derives from ``close`` is therefore invariant to
  any permutation (``close`` is never a feature column).
* The driver slices to the CONTIGUOUS tail starting at the first row where
  every feature column is non-NaN - dropping the leading warmup block but
  preserving calendar order, so a close-derived target's ``shift(-1)`` /
  rolling window stays aligned. Permutation then shuffles only the finite
  entries of each column in place, leaving any interior NaN where it is, so
  the set of rows actually scored is identical between the baseline and every
  permuted frame (a shuffle can neither relocate a NaN nor change the sample).

Collinearity caveat: permutation importance under-attributes correlated
features (permuting one of two collinear features leaves the signal in the
other). The curated, low-within-family-redundancy feature set is the primary
mitigation; this is noted in the thesis rather than handled with heavier
conditional-permutation machinery.

Residual-dominance caveat (the two ARMA/GARCH+LSTM hybrids): a hybrid's final
forecast is its linear backbone (ARMA conditional mean / GARCH conditional
variance, both derived from ``close``) plus a small LSTM residual correction
that is the only part the engineered features feed. Permuting a feature
therefore moves the forecast only through that residual, so hybrid permutation
importances are compressed toward zero and separate signal from noise far less
sharply than the directional classifiers, whose every prediction flows through
the permuted features. Near-zero hybrid importances mean "the feature barely
moved the residual-corrected forecast", NOT "the feature is worthless"; the
thesis reads the hybrid bars with this in mind rather than against the
classifier bars on the same axis.

Determinism: the permutation RNG is seeded explicitly by the caller
(``FEATURE_IMPORTANCE_RNG_SEED`` offset per fold), so importances reproduce
across invocations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core import json_io
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.strategies.interface import IStrategy

_logger = get_logger(__name__)


class ImportanceMethod(StrEnum):
    """
    How a feature-importance score was produced.
    """

    PERMUTATION = "permutation"
    XGB_GAIN = "xgb_gain"


def _score_to_json(value: float) -> float | None:
    """
    Map a score field to a JSON-safe value: ``null`` for non-finite, the float otherwise.

    ``json.dump`` (default ``allow_nan=True``) would emit the bare ``NaN`` /
    ``Infinity`` tokens, which strict non-Python parsers (the webapp frontend)
    reject; encoding non-finite as ``null`` keeps ``feature_importance.json`` valid.
    """

    return None if not math.isfinite(value) else value


def _score_from_json(payload: dict[str, object], key: str) -> float:
    """
    Read a score field written by :func:`_score_to_json` back to a float (``null`` -> NaN).
    """

    value = json_io.get_optional_float(payload, key)
    return float("nan") if value is None else value


@dataclass(frozen=True)
class FeatureImportance:
    """
    Importance of one feature under one method, for one fold.

    ``importance`` is the mean score-drop across permutation repeats
    (``PERMUTATION``) or the native booster gain (``XGB_GAIN``). ``std`` is
    the across-repeats standard deviation for permutation and ``0.0`` for
    gain (a single deterministic value).
    """

    feature: str
    importance: float
    std: float
    method: ImportanceMethod

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "importance": _score_to_json(self.importance),
            "std": _score_to_json(self.std),
            "method": self.method.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> FeatureImportance:
        return cls(
            feature=json_io.get_str(d, "feature"),
            importance=_score_from_json(d, "importance"),
            std=_score_from_json(d, "std"),
            method=ImportanceMethod(json_io.get_str(d, "method")),
        )


@dataclass(frozen=True)
class FoldImportance:
    """
    All per-feature importances for one fold of one strategy.

    ``scores`` carries one :class:`FeatureImportance` per (feature, method)
    pair - so a classifier strategy contributes both a permutation entry and
    a gain entry per feature, while a hybrid contributes only permutation.
    """

    fold_index: int
    scores: tuple[FeatureImportance, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "scores": [s.to_dict() for s in self.scores],
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> FoldImportance:
        return cls(
            fold_index=json_io.get_int(d, "fold_index"),
            scores=tuple(
                FeatureImportance.from_dict(e) for e in json_io.get_list_of_dicts(d, "scores")
            ),
        )


@dataclass(frozen=True)
class AggregatedImportance:
    """
    A feature's importance aggregated across folds, for one method.

    ``importance`` is the mean of the per-fold importances and ``std`` their
    across-fold standard deviation - the fold-to-fold variability that the
    study-report error bars display (distinct from the per-fold across-repeats
    ``std`` carried by :class:`FeatureImportance`).
    """

    feature: str
    importance: float
    std: float
    n_folds: int
    method: ImportanceMethod

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "importance": _score_to_json(self.importance),
            "std": _score_to_json(self.std),
            "n_folds": self.n_folds,
            "method": self.method.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> AggregatedImportance:
        return cls(
            feature=json_io.get_str(d, "feature"),
            importance=_score_from_json(d, "importance"),
            std=_score_from_json(d, "std"),
            n_folds=json_io.get_int(d, "n_folds"),
            method=ImportanceMethod(json_io.get_str(d, "method")),
        )


def permutation_importance(
    score_fn: Callable[[pd.DataFrame], float],
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    n_repeats: int,
    rng: np.random.Generator,
    baseline: float | None = None,
) -> tuple[FeatureImportance, ...]:
    """
    Out-of-sample permutation importance for each column in ``feature_columns``.

    For each column, ``n_repeats`` independent shuffles of that column's
    values are scored; the importance is the mean of ``baseline - score`` and
    ``std`` the across-repeats SAMPLE standard deviation (ddof=1; ``0.0`` for a
    single repeat). ``baseline`` defaults to ``score_fn(features)`` when not
    supplied. Higher importance = the feature matters more (permuting it costs
    more score).

    Only the FINITE entries of each column are shuffled, in place, with any NaN
    left at its original position. This keeps the NaN pattern - and therefore
    the set of rows the score function actually scores - identical between the
    baseline and every permuted frame, so the score drop measures the feature
    rather than a shifted sample. It also lets the caller pass a contiguous
    frame (interior NaNs intact) so any close-derived realised target stays in
    calendar order.

    ``features`` is NEVER mutated: a working copy is taken per column, and only
    that one column is overwritten with shuffled values, leaving every other
    column at its original value (single-feature permutation).
    """

    if n_repeats < 1:
        raise ValueError(
            f"n_repeats must be >= 1, got {n_repeats}; fix by passing a positive "
            f"repeat count (FEATURE_IMPORTANCE_N_REPEATS is the project default)."
        )
    base = score_fn(features) if baseline is None else baseline
    results: list[FeatureImportance] = []
    for column in feature_columns:
        original = np.asarray(features[column], dtype=np.float64)
        finite_positions = np.flatnonzero(~np.isnan(original))
        working = features.copy()
        drops = np.empty(n_repeats, dtype=np.float64)
        for repeat in range(n_repeats):
            shuffled = original.copy()
            shuffled[finite_positions] = rng.permutation(original[finite_positions])
            working[column] = shuffled
            drops[repeat] = base - score_fn(working)
        results.append(
            FeatureImportance(
                feature=column,
                importance=float(np.mean(drops)),
                std=float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
                method=ImportanceMethod.PERMUTATION,
            )
        )
    return tuple(results)


def xgb_gain_importance(
    gain: Mapping[str, float],
    feature_columns: Sequence[str],
) -> tuple[FeatureImportance, ...]:
    """
    Wrap an XGBoost gain map into one :class:`FeatureImportance` per column.

    Features the booster never split on are filled with ``0.0`` so the result
    always has exactly one entry per training feature column (a stable schema
    regardless of which features the booster used).
    """

    return tuple(
        FeatureImportance(
            feature=column,
            importance=float(gain.get(column, 0.0)),
            std=0.0,
            method=ImportanceMethod.XGB_GAIN,
        )
        for column in feature_columns
    )


def compute_fold_importance(
    strategy: IStrategy,
    test_frame: pd.DataFrame,
    fold_index: int,
    *,
    n_repeats: int,
    rng: np.random.Generator,
) -> FoldImportance | None:
    """
    Compute one fold's feature importance for a feature-consuming strategy.

    Returns ``None`` (skip) when the strategy declares no feature columns
    (rule-based strategies), exposes no importance frame, or cannot produce a
    finite baseline score (frame too short after warmup). Otherwise runs
    permutation importance for every feature column and, when the strategy
    exposes a booster gain map, appends the native-gain entries.

    ``test_frame`` is the OOS fold frame of an already-fitted model, exactly
    as the strategy was evaluated on. The strategy maps it to its permutable
    importance frame (``feature_importance_frame``), which is then sliced to the
    CONTIGUOUS tail starting at the first row where every feature column is
    non-NaN. Slicing a contiguous block (rather than boolean-masking out every
    NaN row) drops only the leading warmup region while preserving calendar
    order, so a close-derived realised target's ``shift(-1)`` / rolling window
    stays correctly aligned; the permutation driver then shuffles only finite
    values in place, leaving any interior NaN where it is.

    Returns ``None`` (and logs) on each skip path so a feature-consuming
    strategy silently dropping out of the artifact is visible in the run log.
    """

    strategy_name = type(strategy).__name__
    columns = tuple(strategy.feature_columns())
    if not columns:
        return None
    frame = strategy.feature_importance_frame(test_frame)
    if frame is None:
        _logger.debug(
            "%s fold %d: no feature-importance frame (model unfit) - skipping",
            strategy_name,
            fold_index,
        )
        return None
    valid_mask = frame.loc[:, list(columns)].notna().all(axis=1)
    valid_positions = np.flatnonzero(np.asarray(valid_mask))
    if valid_positions.size == 0:
        _logger.warning(
            "%s fold %d: no rows with every feature column present - skipping feature importance",
            strategy_name,
            fold_index,
        )
        return None
    clean = frame.iloc[int(valid_positions[0]) :]

    baseline = strategy.feature_importance_score(clean)
    if baseline is None or math.isnan(baseline):
        _logger.warning(
            "%s fold %d: non-finite baseline score (scored region too short "
            "after warmup) - skipping feature importance",
            strategy_name,
            fold_index,
        )
        return None

    def scorer(frame: pd.DataFrame) -> float:
        value = strategy.feature_importance_score(frame)
        if value is None:
            raise ValueError(
                f"{type(strategy).__name__} exposes feature_columns but "
                f"feature_importance_score returned None during permutation; a "
                f"feature-consuming strategy must return a float score."
            )
        return value

    scores: list[FeatureImportance] = list(
        permutation_importance(
            scorer, clean, columns, n_repeats=n_repeats, rng=rng, baseline=baseline
        )
    )

    gain = strategy.feature_gain()
    if gain is not None:
        scores.extend(xgb_gain_importance(gain, columns))

    return FoldImportance(fold_index=fold_index, scores=tuple(scores))


def aggregate_fold_importance(
    folds: Sequence[FoldImportance],
) -> tuple[AggregatedImportance, ...]:
    """
    Aggregate per-fold importances into mean +/- across-fold std per (method, feature).

    ``std`` is the across-fold SAMPLE standard deviation (ddof=1), or ``0.0``
    for a single fold (where a population std would be a misleading zero and a
    sample std undefined). Sorted by method then descending mean importance so
    the artifact and any downstream table read top-down.
    """

    grouped: dict[tuple[ImportanceMethod, str], list[float]] = defaultdict(list)
    for fold in folds:
        for score in fold.scores:
            grouped[(score.method, score.feature)].append(score.importance)

    aggregated: list[AggregatedImportance] = []
    for (method, feature), values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        aggregated.append(
            AggregatedImportance(
                feature=feature,
                importance=float(arr.mean()),
                std=float(arr.std(ddof=1)) if len(values) > 1 else 0.0,
                n_folds=len(values),
                method=method,
            )
        )
    aggregated.sort(key=lambda a: (a.method.value, -a.importance))
    return tuple(aggregated)


def build_importance_artifact(folds: Sequence[FoldImportance]) -> dict[str, object]:
    """
    Assemble the ``feature_importance.json`` payload: per-fold + cross-fold aggregate.
    """

    return {
        "n_folds": len(folds),
        "per_fold": [fold.to_dict() for fold in folds],
        "aggregated": [agg.to_dict() for agg in aggregate_fold_importance(folds)],
    }


def read_aggregated_importance(payload: dict[str, object]) -> tuple[AggregatedImportance, ...]:
    """
    Read the cross-fold aggregate back out of a ``feature_importance.json`` payload.
    """

    return tuple(
        AggregatedImportance.from_dict(e) for e in json_io.get_list_of_dicts(payload, "aggregated")
    )
