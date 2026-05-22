"""Serializable value types for the orchestration layer.

Each record round-trips through ``to_dict`` / ``from_dict`` so experiment
output can be written as JSONL (one line per fold, diffable and appendable)
without dragging numpy / C++ binding classes into the persistence layer.

``FoldRecord`` is the serialization mirror of :class:`FoldResult`; the live
C++-owned ``BacktestResult`` and ``PerformanceMetrics`` collapse to plain
floats + a tuple equity curve before anything touches disk.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

import pandas as pd

from src.analysis.metrics_aggregator import AggregateStats
from src.core import json_io
from src.engine.walk_forward import FoldResult
from src.orchestration.manifest import Manifest


@dataclass(frozen=True)
class FoldRecord:
    """Per-fold metrics snapshot, JSON-serializable.

    Run-wide context (slippage scenario, seed, git sha) lives on
    :class:`ExperimentResult.manifest`, not here — a fold record carries
    only what differs across folds.
    """

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    equity_curve: tuple[float, ...]
    # Strategy-emitted scalars (e.g. floor_bind_fraction). Persisted as a
    # plain JSON object alongside the headline metrics so post-hoc analyses
    # can read per-fold diagnostics without rerunning the strategy.
    strategy_diagnostics: Mapping[str, float] = MappingProxyType({})

    @classmethod
    def from_fold_result(cls, fr: FoldResult) -> FoldRecord:
        """Flatten a :class:`FoldResult` into scalars + an equity tuple."""
        return cls(
            fold_index=fr.fold_index,
            train_start=pd.Timestamp(fr.train_start),
            train_end=pd.Timestamp(fr.train_end),
            test_start=pd.Timestamp(fr.test_start),
            test_end=pd.Timestamp(fr.test_end),
            total_return=fr.backtest.total_return,
            annualized_return=fr.metrics.annualized_return,
            annualized_volatility=fr.metrics.annualized_volatility,
            sharpe_ratio=fr.metrics.sharpe_ratio,
            sortino_ratio=fr.metrics.sortino_ratio,
            calmar_ratio=fr.metrics.calmar_ratio,
            max_drawdown=fr.metrics.max_drawdown,
            win_rate=fr.metrics.win_rate,
            trade_count=fr.backtest.trade_count,
            equity_curve=tuple(fr.backtest.equity_curve.tolist()),
            strategy_diagnostics=MappingProxyType(dict(fr.strategy_diagnostics)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "trade_count": self.trade_count,
            "equity_curve": list(self.equity_curve),
            "strategy_diagnostics": dict(self.strategy_diagnostics),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> FoldRecord:
        diagnostics_raw = d.get("strategy_diagnostics", {})
        if not isinstance(diagnostics_raw, dict):
            raise TypeError(
                f"strategy_diagnostics must be a dict, got {type(diagnostics_raw).__name__}"
            )
        diagnostics: dict[str, float] = {str(k): float(v) for k, v in diagnostics_raw.items()}
        return cls(
            fold_index=json_io.get_int(d, "fold_index"),
            train_start=json_io.get_timestamp(d, "train_start"),
            train_end=json_io.get_timestamp(d, "train_end"),
            test_start=json_io.get_timestamp(d, "test_start"),
            test_end=json_io.get_timestamp(d, "test_end"),
            total_return=json_io.get_float(d, "total_return"),
            annualized_return=json_io.get_float(d, "annualized_return"),
            annualized_volatility=json_io.get_float(d, "annualized_volatility"),
            sharpe_ratio=json_io.get_float(d, "sharpe_ratio"),
            sortino_ratio=json_io.get_float(d, "sortino_ratio"),
            calmar_ratio=json_io.get_float(d, "calmar_ratio"),
            max_drawdown=json_io.get_float(d, "max_drawdown"),
            win_rate=json_io.get_float(d, "win_rate"),
            trade_count=json_io.get_int(d, "trade_count"),
            equity_curve=tuple(json_io.get_float_list(d, "equity_curve")),
            strategy_diagnostics=MappingProxyType(diagnostics),
        )


@dataclass(frozen=True)
class ExperimentResult:
    """Root output of ``Experiment.run()``: manifest + per-fold records.

    ``experiment_id`` is exposed as a top-level field for convenience but is
    ALSO carried inside ``manifest``; the two are always equal (the manifest
    is the source of truth on disk, the top-level field is an in-memory
    shortcut). Round-trip tests verify the equality.
    """

    experiment_id: str
    folds: tuple[FoldRecord, ...]
    manifest: Manifest

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "folds": [fold.to_dict() for fold in self.folds],
            "manifest": self.manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ExperimentResult:
        raw_folds = json_io.get_list_of_dicts(d, "folds")
        raw_manifest = json_io.get_dict(d, "manifest")
        return cls(
            experiment_id=json_io.get_str(d, "experiment_id"),
            folds=tuple(FoldRecord.from_dict(f) for f in raw_folds),
            manifest=Manifest.from_dict(raw_manifest),
        )


@dataclass(frozen=True)
class PairwiseSignificance:
    """One pairwise Sharpe-differential bootstrap result.

    The differential is ``sharpe(name_a) - sharpe(name_b)`` computed on
    the concatenated per-fold returns of the two strategies. ``significant``
    is the observable: ``True`` when the confidence interval excludes zero,
    which is what the reporter's LaTeX cell displays.
    """

    name_a: str
    name_b: str
    point_differential: float
    lower: float
    upper: float
    confidence: float
    significant: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "point_differential": self.point_differential,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "significant": self.significant,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> PairwiseSignificance:
        return cls(
            name_a=json_io.get_str(d, "name_a"),
            name_b=json_io.get_str(d, "name_b"),
            point_differential=json_io.get_float(d, "point_differential"),
            lower=json_io.get_float(d, "lower"),
            upper=json_io.get_float(d, "upper"),
            confidence=json_io.get_float(d, "confidence"),
            significant=json_io.get_bool(d, "significant"),
        )


@dataclass(frozen=True)
class StrategyComparisonReport:
    """Aggregate output of :func:`run_comparison`.

    Value object — not persisted whole. The comparison reporter writes
    ``ranking.tex`` from ``ranking``, ``pairwise_significance.tex`` from
    ``pairwise``, and a JSON manifest from the scalar identity fields;
    ``per_strategy_stats`` is embedded in the JSON manifest as a
    per-strategy ``to_dict()`` payload so the report directory is
    self-contained.

    ``per_strategy_experiment_id`` maps each strategy name back to the
    run directory under ``experiment_results/runs/`` so a user can
    drill into a specific strategy's fold records from the report.

    ``per_strategy_per_regime_stats`` is populated when
    :func:`run_comparison` is called with a regime config. The outer key
    is the strategy name; the inner key is a regime label (incl.
    :data:`MIXED_REGIME_LABEL` for folds without a dominant regime).
    ``None`` indicates regime overlay was not requested for this run.

    Distinct from :class:`src.benchmarking.types.ComparisonReport` (which
    compares perf benchmark runs); the qualified name keeps the two from
    colliding when both packages are imported in the same module.
    """

    out_name: str
    created_at: datetime
    git_sha: str
    per_strategy_experiment_id: Mapping[str, str]
    per_strategy_stats: Mapping[str, AggregateStats]
    ranking: pd.DataFrame
    pairwise: tuple[PairwiseSignificance, ...]
    per_strategy_per_regime_stats: Mapping[str, Mapping[str, AggregateStats]] | None = None

    def __post_init__(self) -> None:
        # Frozen dataclass freezes the bindings, not the dict objects
        # themselves. Wrap in MappingProxyType so callers can't mutate
        # the maps in place after the report is constructed.
        object.__setattr__(
            self,
            "per_strategy_experiment_id",
            MappingProxyType(dict(self.per_strategy_experiment_id)),
        )
        object.__setattr__(
            self,
            "per_strategy_stats",
            MappingProxyType(dict(self.per_strategy_stats)),
        )
        if self.per_strategy_per_regime_stats is not None:
            object.__setattr__(
                self,
                "per_strategy_per_regime_stats",
                MappingProxyType(
                    {
                        name: MappingProxyType(dict(per_regime))
                        for name, per_regime in self.per_strategy_per_regime_stats.items()
                    }
                ),
            )


class RegimeKind(StrEnum):
    """Detector family — encodes the *type* of split, not the labels.

    Surfaced on :class:`RegimeReport` so a downstream reader knows which
    detector produced the labels without having to parse the YAML config
    again. The labels themselves are arbitrary strings supplied by the
    detector (e.g. ``"bull"`` / ``"bear"`` for trend, ``"Q1"``..``"Q5"``
    for volatility quintiles).
    """

    PERIOD = "period"
    TREND = "trend"
    VOLATILITY = "volatility"


# Reserved label for folds whose test window straddles regime boundaries
# without a dominant regime — see :func:`split_folds_by_regime` and the
# ``majority_threshold`` knob. Surfaced as a separate row in regime
# reports rather than silently dropped.
MIXED_REGIME_LABEL = "mixed"

# Reserved label for bars the detector cannot classify (trend / volatility
# warmup, period-detector gap days). Distinct from :data:`MIXED_REGIME_LABEL`
# (which is a fold-level concept). Excluded from per-regime stats and from
# the bar-count majority math in :func:`split_folds_by_regime`.
UNCLASSIFIED_LABEL = "unclassified"


@dataclass(frozen=True)
class RegimeSlice:
    """Contiguous time range tagged with a single regime label.

    ``start`` is inclusive, ``end`` is exclusive — matches ``df.loc[start:end]``
    semantics when the index is a ``DatetimeIndex``. A detector emits a
    list of slices via run-length encoding of its per-bar tag series; the
    same label may appear in multiple slices when the regime is non-contiguous
    (e.g., a "bear" period that recurs years later).
    """

    label: str
    start: pd.Timestamp
    end: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> RegimeSlice:
        return cls(
            label=json_io.get_str(d, "label"),
            start=json_io.get_timestamp(d, "start"),
            end=json_io.get_timestamp(d, "end"),
        )


@dataclass(frozen=True)
class RegimeReport:
    """Per-regime aggregate stats for a single experiment.

    Maps regime label → :class:`AggregateStats` over the folds whose test
    windows landed primarily in that regime. ``per_regime_fold_indices``
    records which fold indices contributed to which regime so a reader can
    drill back into ``fold_results.jsonl``. Folds without a dominant regime
    (majority below threshold) are collected in ``mixed_fold_indices`` and
    aggregated under :data:`MIXED_REGIME_LABEL` in ``per_regime_stats`` —
    they are not silently dropped.

    The detector that produced the labels is referenced via ``kind`` (the
    family) plus ``detector_name`` (the registry key) so a future reader can
    rebuild the same split by loading the matching regime YAML; the actual
    label semantics live in the detector itself.
    """

    out_name: str
    experiment_id: str
    kind: RegimeKind
    detector_name: str
    created_at: datetime
    git_sha: str
    per_regime_stats: Mapping[str, AggregateStats]
    per_regime_fold_indices: Mapping[str, tuple[int, ...]]
    mixed_fold_indices: tuple[int, ...]
    slices: tuple[RegimeSlice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "per_regime_stats",
            MappingProxyType(dict(self.per_regime_stats)),
        )
        object.__setattr__(
            self,
            "per_regime_fold_indices",
            MappingProxyType(dict(self.per_regime_fold_indices)),
        )
