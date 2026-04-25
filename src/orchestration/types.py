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
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> FoldRecord:
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
