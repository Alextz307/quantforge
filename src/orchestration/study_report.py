"""
Consolidate a study's per-leg artifacts into a single cross-leg report.

The empirical-study orchestrator (:func:`src.orchestration.study.run_study`)
writes per-leg artifact directories under ``<study_dir>/``:

* ``runs/<run_experiment_id>/``           one per leg (best-config materialised run)
* ``holdout_evals/<leg_id>/``             one per leg, only when validation reserved a holdout
* ``comparisons/<universe>/``             one per universe with >=2 strategies

This module walks that tree and builds a single
:class:`ConsolidatedStudyReport` value object covering every completed
leg. It is **read-only** with respect to the study tree: the
consolidator does not refit, retrain, or recompute anything - every
scalar comes from a JSON / JSONL artifact already on disk.

The downstream :class:`src.visualization.study_report_reporter.StudyReportReporter`
consumes this value object and emits the consolidated tables + plots
under ``<study_dir>/{tables,plots,manifest.json}``.

Incomplete legs are skipped with a WARN log line and surfaced via
:attr:`ConsolidatedStudyReport.incomplete_leg_ids`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from src.analysis.baselines import BaselineResult
from src.analysis.metrics_aggregator import AggregateStats, aggregate_folds
from src.analysis.significance import BootstrapCI, DeflatedSharpe
from src.core import json_io
from src.core.logging import get_logger
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    DSR_JSON_FILENAME,
    HOLDOUT_EVAL_JSON,
    HOLDOUT_EVALS_SUBDIR,
    HPO_SUBDIR,
)
from src.orchestration.git_info import read_git_sha
from src.orchestration.run_loader import load_experiment_result, resolve_run_dir
from src.orchestration.study import STUDY_STATE_FILENAME
from src.orchestration.study_state import read_study_state
from src.orchestration.types import PairwiseSignificance
from src.strategies.volatility_targeting import FLOOR_BIND_DIAGNOSTIC_KEY
from src.visualization.plots import MANIFEST_FILENAME

_logger = get_logger(__name__)


@dataclass(frozen=True)
class FloorBindStats:
    """
    Aggregate sigma_min saturation across a leg's folds.

    ``mean`` / ``max`` / ``min`` are taken over the folds that emitted a
    ``floor_bind_fraction`` diagnostic - typically every fold for a
    VolatilityTargeting leg, none for any other strategy. ``n_folds`` is
    the count of contributing folds; when zero the leg never produced
    the diagnostic and consumers should treat the row as absent.
    """

    mean: float
    max: float
    min: float
    n_folds: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "max": self.max,
            "min": self.min,
            "n_folds": self.n_folds,
        }


def aggregate_floor_bind_across_folds(
    diagnostics: Sequence[Mapping[str, float]],
) -> FloorBindStats | None:
    """
    Return ``FloorBindStats`` across folds that emitted the diagnostic.

    ``None`` indicates none of the folds carried ``floor_bind_fraction``
    (e.g. a non-VolatilityTargeting leg); callers should omit the leg
    from the floor-saturation table rather than inserting a sentinel.
    """

    values = [
        float(d[FLOOR_BIND_DIAGNOSTIC_KEY])
        for d in diagnostics
        if FLOOR_BIND_DIAGNOSTIC_KEY in d and math.isfinite(d[FLOOR_BIND_DIAGNOSTIC_KEY])
    ]
    if not values:
        return None
    return FloorBindStats(
        mean=sum(values) / len(values),
        max=max(values),
        min=min(values),
        n_folds=len(values),
    )


@dataclass(frozen=True)
class HoldoutSnapshot:
    """
    Scalar metrics from a per-leg ``holdout_eval.json``.

    Equity curves and other per-leg artifacts stay under
    ``holdout_evals/<leg_id>/``; the consolidator reads only the
    scalars it needs for cross-leg tables and the dev-vs-holdout scatter.

    The bootstrap Sharpe CI is carried verbatim from the holdout-eval
    payload. The buy-and-hold baseline is read in full so the consolidated
    ranking table can show excess-over-baseline columns without re-running
    the long-only backtest.
    """

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    annualized_return: float
    annualized_volatility: float
    total_return: float
    win_rate: float
    trade_count: int
    holdout_start: pd.Timestamp
    n_dev_bars: int
    n_holdout_bars: int
    sharpe_ci: BootstrapCI
    buy_and_hold: BaselineResult

    @classmethod
    def from_holdout_json(cls, path: Path) -> HoldoutSnapshot:
        """
        Read ``holdout_eval.json`` and pull the scalar metrics block.

        Validates the ``is_holdout_eval: True`` discriminator so a
        mis-pointed path against a regular run manifest fails loud
        rather than silently zero-filling the table.
        """

        d = json_io.read_dict(path)
        is_holdout = d.get("is_holdout_eval")
        if is_holdout is not True:
            raise ValueError(
                f"file {path} is not a holdout-eval payload "
                f"(is_holdout_eval={is_holdout!r}); the consolidator was pointed "
                f"at a non-holdout JSON file."
            )
        metrics = json_io.get_dict(d, "metrics")
        sharpe_ci = BootstrapCI.from_dict(json_io.get_dict(metrics, "sharpe_ci"))
        bah = BaselineResult.from_dict(json_io.get_dict(d, "buy_and_hold"))

        return cls(
            sharpe_ratio=json_io.get_float(metrics, "sharpe_ratio"),
            sortino_ratio=json_io.get_float(metrics, "sortino_ratio"),
            calmar_ratio=json_io.get_float(metrics, "calmar_ratio"),
            max_drawdown=json_io.get_float(metrics, "max_drawdown"),
            annualized_return=json_io.get_float(metrics, "annualized_return"),
            annualized_volatility=json_io.get_float(metrics, "annualized_volatility"),
            total_return=json_io.get_float(metrics, "total_return"),
            win_rate=json_io.get_float(metrics, "win_rate"),
            trade_count=json_io.get_int(metrics, "trade_count"),
            holdout_start=json_io.get_timestamp(d, "holdout_start"),
            n_dev_bars=json_io.get_int(d, "n_dev_bars"),
            n_holdout_bars=json_io.get_int(d, "n_holdout_bars"),
            sharpe_ci=sharpe_ci,
            buy_and_hold=bah,
        )


@dataclass(frozen=True)
class ConsolidatedStudyReport:
    """
    Cross-leg view of a completed study, suitable for the writeup tables.

    All per-leg maps are keyed by ``(strategy, universe)`` tuples. Per-universe
    maps are keyed by universe name. Maps may be empty when the underlying
    artifacts weren't produced (universes with ``holdout_pct=0``,
    single-strategy universes for pairwise); consumers must check membership
    before reading.
    """

    study_name: str
    study_dir: Path
    created_at: datetime
    git_sha: str
    per_leg_aggregate: Mapping[tuple[str, str], AggregateStats]
    per_leg_run_id: Mapping[tuple[str, str], str]
    per_leg_holdout: Mapping[tuple[str, str], HoldoutSnapshot]
    per_leg_dsr: Mapping[tuple[str, str], DeflatedSharpe]
    per_leg_floor_bind: Mapping[tuple[str, str], FloorBindStats]
    per_universe_pairwise: Mapping[str, tuple[PairwiseSignificance, ...]]
    incomplete_leg_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        # Wrap mapping fields so callers cannot mutate the maps in place
        # after construction - frozen=True freezes the bindings, not the
        # dict contents themselves.
        object.__setattr__(
            self, "per_leg_aggregate", MappingProxyType(dict(self.per_leg_aggregate))
        )
        object.__setattr__(self, "per_leg_run_id", MappingProxyType(dict(self.per_leg_run_id)))
        object.__setattr__(self, "per_leg_holdout", MappingProxyType(dict(self.per_leg_holdout)))
        object.__setattr__(self, "per_leg_dsr", MappingProxyType(dict(self.per_leg_dsr)))
        object.__setattr__(
            self, "per_leg_floor_bind", MappingProxyType(dict(self.per_leg_floor_bind))
        )
        object.__setattr__(
            self, "per_universe_pairwise", MappingProxyType(dict(self.per_universe_pairwise))
        )

    @property
    def strategies(self) -> tuple[str, ...]:
        """
        Sorted tuple of every strategy seen across completed legs.
        """

        return tuple(sorted({s for (s, _) in self.per_leg_aggregate}))

    @property
    def universes(self) -> tuple[str, ...]:
        """
        Sorted tuple of every universe seen across completed legs.
        """

        return tuple(sorted({u for (_, u) in self.per_leg_aggregate}))


def consolidate_study(study_dir: Path) -> ConsolidatedStudyReport:
    """
    Walk ``study_dir`` and assemble the cross-leg consolidated view.

    Reads ``study_state.json`` for the leg roster, then for each completed
    leg loads ``runs/<run_id>/{manifest.json,fold_results.jsonl}`` (via
    :func:`load_experiment_result`) plus optional
    ``holdout_evals/<leg_id>/holdout_eval.json``. Per-universe pairwise
    data comes from ``comparisons/<universe>/manifest.json``.

    Raises ``FileNotFoundError`` only when ``study_state.json`` itself is
    missing - every other artifact is treated as best-effort. Incomplete
    legs are surfaced via :attr:`ConsolidatedStudyReport.incomplete_leg_ids`
    so the reporter can flag them in the consolidated manifest.
    """

    state_path = study_dir / STUDY_STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(
            f"study state not found at {state_path}; the orchestrator writes this "
            f"file at study start, so its absence means {study_dir} is not a "
            f"completed study directory. Pass --study-dir against the path "
            f"returned by `experiment study run`."
        )
    state = read_study_state(state_path)

    per_leg_aggregate: dict[tuple[str, str], AggregateStats] = {}
    per_leg_run_id: dict[tuple[str, str], str] = {}
    per_leg_holdout: dict[tuple[str, str], HoldoutSnapshot] = {}
    per_leg_dsr: dict[tuple[str, str], DeflatedSharpe] = {}
    per_leg_floor_bind: dict[tuple[str, str], FloorBindStats] = {}
    incomplete: list[str] = []

    for leg in state.legs:
        if not leg.is_complete or leg.run_experiment_id is None:
            incomplete.append(leg.leg_id)
            continue

        key = (leg.strategy, leg.universe)
        run_dir = resolve_run_dir(study_dir, leg.run_experiment_id)
        result = load_experiment_result(run_dir)
        per_leg_aggregate[key] = aggregate_folds(result.folds)
        per_leg_run_id[key] = leg.run_experiment_id

        floor_stats = aggregate_floor_bind_across_folds(
            [fold.strategy_diagnostics for fold in result.folds]
        )
        if floor_stats is not None:
            per_leg_floor_bind[key] = floor_stats

        holdout_path = study_dir / HOLDOUT_EVALS_SUBDIR / leg.leg_id / HOLDOUT_EVAL_JSON
        try:
            per_leg_holdout[key] = HoldoutSnapshot.from_holdout_json(holdout_path)
        except FileNotFoundError:
            pass

        dsr_path = study_dir / HPO_SUBDIR / leg.leg_id / DSR_JSON_FILENAME
        try:
            per_leg_dsr[key] = DeflatedSharpe.from_dict(json_io.read_dict(dsr_path))
        except FileNotFoundError:
            pass

    per_universe_pairwise: dict[str, tuple[PairwiseSignificance, ...]] = {}
    for universe in sorted({u for (_, u) in per_leg_aggregate}):
        pairwise = _try_read_comparison_pairwise(study_dir, universe)
        if pairwise is not None:
            per_universe_pairwise[universe] = pairwise

    if incomplete:
        _logger.warning(
            "skipping %d incomplete leg(s) in consolidation: %s",
            len(incomplete),
            ", ".join(incomplete),
        )

    return ConsolidatedStudyReport(
        study_name=state.spec_name,
        study_dir=study_dir,
        created_at=datetime.now(UTC),
        git_sha=read_git_sha(),
        per_leg_aggregate=per_leg_aggregate,
        per_leg_run_id=per_leg_run_id,
        per_leg_holdout=per_leg_holdout,
        per_leg_dsr=per_leg_dsr,
        per_leg_floor_bind=per_leg_floor_bind,
        per_universe_pairwise=per_universe_pairwise,
        incomplete_leg_ids=tuple(incomplete),
    )


def _try_read_comparison_pairwise(
    study_dir: Path, universe: str
) -> tuple[PairwiseSignificance, ...] | None:
    """
    Load ``comparisons/<universe>/manifest.json`` and parse the pairwise list.

    Returns ``None`` when the manifest is absent (single-strategy universes
    that had no compare run, or studies that were launched with
    ``--skip-compares``).
    """

    manifest_path = study_dir / COMPARISONS_SUBDIR / universe / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    payload = json_io.read_dict(manifest_path)
    raw_pairwise = payload.get("pairwise", [])
    if not isinstance(raw_pairwise, list):
        raise ValueError(
            f"comparison manifest at {manifest_path} has 'pairwise' field of "
            f"unexpected type {type(raw_pairwise).__name__}; expected list."
        )
    parsed: list[PairwiseSignificance] = []
    for raw in raw_pairwise:
        if not isinstance(raw, dict):
            raise ValueError(
                f"pairwise entries in {manifest_path} must be dicts, got {type(raw).__name__}"
            )
        parsed.append(PairwiseSignificance.from_dict(raw))
    return tuple(parsed)


__all__ = [
    "ConsolidatedStudyReport",
    "FloorBindStats",
    "HoldoutSnapshot",
    "aggregate_floor_bind_across_folds",
    "consolidate_study",
]
