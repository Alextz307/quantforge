"""Multi-strategy comparison orchestrator.

Composes multiple :class:`ExperimentConfig` runs into a single
:class:`StrategyComparisonReport`: ranked per-strategy stats, pairwise Sharpe
significance, concatenated-equity view. Parallelism is opt-in via
``n_jobs`` — ``n_jobs=1`` runs in-process (simplest, no pickle pain),
``n_jobs>1`` fans out via :class:`ProcessPoolExecutor` so each experiment
gets a fresh process (fresh GPU / torch state) at the cost of per-worker
Python startup.

Execution paths
---------------
We deliberately branch on ``n_jobs`` instead of unifying the two:

* In-process (``n_jobs == 1``): cheapest single-run cost — no worker
  startup, exceptions surface with full tracebacks, breakpoints work.
  The vast majority of comparisons (3-5 strategies, overnight run)
  use this path.
* Multi-process (``n_jobs > 1``): :func:`_run_one_experiment` at module
  scope so it pickles cleanly for :class:`ProcessPoolExecutor`. Each
  worker re-imports the world (~1-3s cold) — acceptable when the
  per-experiment compute is in the minutes.

Fold alignment for pairwise bootstrap
-------------------------------------
Pairwise Sharpe significance is a paired stationary bootstrap on
bar-level log-returns derived from each strategy's concatenated
per-fold equity curves. For the pairing to be valid, the strategies
being compared MUST have run on the same data / same validator /
same holdout boundary so their folds line up bar-for-bar. We enforce
this by verifying every :class:`ExperimentResult` has an identical
``fold_count`` and identical per-fold ``equity_curve`` length; a
mismatch raises :class:`ValueError` before any bootstrap work begins.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import combinations
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.analysis.metrics_aggregator import AggregateStats, aggregate_folds
from src.analysis.ranking import rank_strategies
from src.analysis.regime_split import aggregate_split, split_folds_by_tags
from src.analysis.significance import paired_bootstrap_sharpe_differential
from src.core.config import DataConfig, ExperimentConfig
from src.core.logging import get_logger
from src.core.persistence import COMPARISONS_SUBDIR
from src.core.regime_config import RegimeConfig
from src.core.registry import data_source_registry
from src.data.fingerprint import assert_data_hash_matches, fingerprint_bars
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import RunOptions
from src.orchestration.git_info import read_git_sha
from src.orchestration.regime import regime_registry
from src.orchestration.types import (
    ExperimentResult,
    FoldRecord,
    PairwiseSignificance,
    StrategyComparisonReport,
)

_logger = get_logger(__name__)

_DEFAULT_STORE_ROOT = Path("experiment_results")

# Pairwise bootstrap default. 5k resamples keeps the wall clock tolerable
# for a 10-strategy comparison (45 pairs) while still landing 2-decimal-stable
# percentiles. Callers can override via ``n_resamples`` for tighter precision.
_DEFAULT_PAIRWISE_N_RESAMPLES = 5_000


class SignificanceTest(StrEnum):
    """Pairwise Sharpe-differential test selector for :func:`run_comparison`."""

    BOOTSTRAP = "bootstrap"
    NONE = "none"


@dataclass(frozen=True)
class _ComparisonInputs:
    """Internal bundle: configs + their resolved strategy names.

    Two strategies in one comparison with identical names would collide
    in ``per_strategy_stats`` — we surface that as a :class:`ValueError`
    early instead of silently overwriting.
    """

    configs: tuple[ExperimentConfig, ...]
    strategy_names: tuple[str, ...]


def run_comparison(
    configs: Sequence[ExperimentConfig],
    *,
    out_name: str,
    store_root: Path | None = None,
    n_jobs: int = 1,
    significance_test: SignificanceTest = SignificanceTest.BOOTSTRAP,
    n_resamples: int = _DEFAULT_PAIRWISE_N_RESAMPLES,
    regime_config: RegimeConfig | None = None,
    reused_results: Sequence[ExperimentResult] | None = None,
    reused_data_cfg: DataConfig | None = None,
) -> tuple[StrategyComparisonReport, dict[str, tuple[FoldRecord, ...]]]:
    """Run every config, aggregate, rank, optionally pairwise-test.

    Returns ``(report, folds_by_strategy)`` — the in-memory
    :class:`StrategyComparisonReport` plus a per-strategy mapping of fold
    records. The reporter's equity-overlay plot consumes the folds; the
    comparison bundle also stores per-fold data under each strategy's
    ``runs/<experiment_id>/fold_results.jsonl`` so callers that have already
    persisted the report can rebuild the dict from disk if they discard it.

    When ``regime_config`` is supplied, every config must declare an
    identical ``data`` block; bars are refetched once and fingerprint-
    checked against each strategy's ``manifest.data_hash``. For pairs
    configs the regime is tagged from ``tickers[0]`` — descriptive only,
    never feeds back into training.

    When ``reused_results`` is supplied (one :class:`ExperimentResult`
    per config, in matching order), the per-strategy walk-forward step
    is skipped entirely — ranking, pairwise bootstrap, and (optional)
    regime overlay run against the prior results. Every reused result's
    ``manifest.data_hash`` must match for the bootstrap pairing to be
    valid. ``reused_data_cfg`` is required when ``regime_config`` is
    also set (the data block needed for bar refetch is recovered from
    one of the reused runs' frozen ``config.yaml`` by the caller).
    """
    inputs = _validate_inputs(configs)
    if reused_results is not None:
        _validate_reused_results_alignment(inputs, reused_results)
        if regime_config is not None and reused_data_cfg is None:
            raise ValueError(
                "regime overlay with reused_results requires reused_data_cfg "
                "(read from one of the reused runs' frozen config.yaml); "
                "fix by passing the data block alongside reused_results."
            )
    elif regime_config is not None:
        _validate_uniform_data(inputs.configs)
    store = store_root if store_root is not None else _DEFAULT_STORE_ROOT
    cmp_dir = Path(store) / COMPARISONS_SUBDIR / out_name
    cmp_dir.mkdir(parents=True, exist_ok=True)

    _logger.info(
        "running comparison '%s' with %d configs (n_jobs=%d, significance=%s, regime=%s, reuse=%s)",
        out_name,
        len(inputs.configs),
        n_jobs,
        significance_test,
        regime_config.detector.name if regime_config is not None else "none",
        "yes" if reused_results is not None else "no",
    )

    results: list[ExperimentResult]
    if reused_results is not None:
        results = list(reused_results)
    elif n_jobs == 1:
        results = _run_sequential(inputs.configs, cmp_dir)
    elif n_jobs > 1:
        results = _run_parallel(inputs.configs, cmp_dir, n_jobs)
    else:
        raise ValueError(
            f"n_jobs must be >= 1 (got {n_jobs}); use 1 for in-process, >1 for "
            f"ProcessPoolExecutor parallelism."
        )

    per_strategy_stats = {
        name: aggregate_folds(result.folds)
        for name, result in zip(inputs.strategy_names, results, strict=True)
    }
    per_strategy_experiment_id = {
        name: result.experiment_id
        for name, result in zip(inputs.strategy_names, results, strict=True)
    }

    ranking = rank_strategies(per_strategy_stats)

    if significance_test is SignificanceTest.BOOTSTRAP:
        pairwise = _compute_pairwise_bootstrap(
            inputs.strategy_names, results, n_resamples=n_resamples
        )
    else:
        pairwise = ()

    per_strategy_per_regime_stats: dict[str, dict[str, AggregateStats]] | None
    if regime_config is not None:
        overlay_data_cfg = (
            reused_data_cfg if reused_data_cfg is not None else inputs.configs[0].data
        )
        per_strategy_per_regime_stats = _compute_regime_overlay(
            inputs.strategy_names,
            results,
            regime_config=regime_config,
            data_cfg=overlay_data_cfg,
        )
    else:
        per_strategy_per_regime_stats = None

    report = StrategyComparisonReport(
        out_name=out_name,
        created_at=datetime.now(UTC),
        git_sha=read_git_sha(),
        per_strategy_experiment_id=per_strategy_experiment_id,
        per_strategy_stats=per_strategy_stats,
        ranking=ranking,
        pairwise=pairwise,
        per_strategy_per_regime_stats=per_strategy_per_regime_stats,
    )
    folds_by_strategy = {
        name: result.folds for name, result in zip(inputs.strategy_names, results, strict=True)
    }
    return report, folds_by_strategy


def _run_one_experiment(cfg: ExperimentConfig, cmp_dir: Path) -> ExperimentResult:
    """Worker entry point — module-level so ProcessPoolExecutor can pickle it.

    ``write_report=False``: per-strategy reporting is redundant with the
    cross-strategy StrategyComparisonReport and triples the wall clock on
    a five-strategy comparison. Users who want per-strategy plots re-run
    the underlying config via ``experiment run``.

    ``store_root=cmp_dir`` routes per-strategy artifacts under
    ``<cmp_dir>/runs/<experiment_id>/`` (the ``runs/`` subdirectory is
    appended inside :meth:`Experiment.run`) so the comparison bundle
    is self-contained — a user can zip the comparison directory and
    everything travels together.
    """
    experiment = build_experiment(cfg)
    return experiment.run(RunOptions(store_root=cmp_dir, write_report=False))


def _validate_inputs(configs: Sequence[ExperimentConfig]) -> _ComparisonInputs:
    if len(configs) < 2:
        raise ValueError(
            f"run_comparison needs at least 2 configs to compare, got {len(configs)}; "
            f"fix by passing two or more distinct configs."
        )
    names = tuple(cfg.name for cfg in configs)
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"run_comparison requires unique config names, found duplicates: {duplicates}; "
            f"fix by editing the 'name:' field in each YAML so every strategy is distinct."
        )
    return _ComparisonInputs(configs=tuple(configs), strategy_names=names)


def _run_sequential(configs: tuple[ExperimentConfig, ...], cmp_dir: Path) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for cfg in configs:
        _logger.info("sequential: starting '%s'", cfg.name)
        out.append(_run_one_experiment(cfg, cmp_dir))
    return out


def _run_parallel(
    configs: tuple[ExperimentConfig, ...], cmp_dir: Path, n_jobs: int
) -> list[ExperimentResult]:
    _logger.info("parallel: spawning %d workers for %d configs", n_jobs, len(configs))
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(_run_one_experiment, cfg, cmp_dir) for cfg in configs]
        # as_completed would give incremental progress but breaks input→output
        # ordering; we need ordered results so the caller's strategy_names tuple
        # lines up with the returned ExperimentResults.
        results: list[ExperimentResult] = []
        try:
            for f in futures:
                results.append(f.result())
        except BaseException:
            # First failure: stop spending CPU on the rest. Without this,
            # ProcessPoolExecutor's __exit__ still waits on every submitted
            # future before re-raising, so an early crash burns the full
            # remaining wall clock for nothing.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        return results


def _compute_pairwise_bootstrap(
    strategy_names: tuple[str, ...],
    results: list[ExperimentResult],
    *,
    n_resamples: int,
) -> tuple[PairwiseSignificance, ...]:
    """Compute the upper-triangular pairwise Sharpe-differential matrix.

    Raises if fold counts or per-fold curve lengths differ between any
    two strategies — that's an alignment violation and pairing the
    bootstrap indices would produce meaningless results.
    """
    _validate_fold_alignment(strategy_names, results)
    per_strategy_returns = {
        name: _concatenated_log_returns(result.folds)
        for name, result in zip(strategy_names, results, strict=True)
    }

    pairwise: list[PairwiseSignificance] = []
    for name_a, name_b in combinations(strategy_names, 2):
        ci = paired_bootstrap_sharpe_differential(
            per_strategy_returns[name_a],
            per_strategy_returns[name_b],
            n_resamples=n_resamples,
        )
        pairwise.append(
            PairwiseSignificance(
                name_a=name_a,
                name_b=name_b,
                point_differential=ci.point_estimate,
                lower=ci.lower,
                upper=ci.upper,
                confidence=ci.confidence,
                significant=ci.excludes(0.0),
            )
        )
    return tuple(pairwise)


def _concatenated_log_returns(
    folds: tuple[FoldRecord, ...],
) -> npt.NDArray[np.float64]:
    """Flatten per-fold equity curves into one bar-level log-return series.

    Folds whose equity curves have fewer than 2 points contribute zero
    returns (a single-bar fold can't define a return); they're skipped
    so a degenerate edge fold doesn't NaN-pollute the whole series.
    """
    pieces: list[npt.NDArray[np.float64]] = []
    for fold in folds:
        curve = np.asarray(fold.equity_curve, dtype=np.float64)
        if len(curve) < 2:
            continue
        # Guard against non-positive equity (a catastrophic fold with
        # debt at start or end) — log of <=0 is undefined and would
        # propagate NaN through the bootstrap.
        if np.any(curve <= 0.0):
            raise ValueError(
                f"fold {fold.fold_index} has non-positive equity; log-return "
                f"concatenation requires strictly positive curves. Fix by "
                f"dropping the degenerate fold or reviewing the strategy for "
                f"blow-up behaviour."
            )
        pieces.append(np.log(curve[1:] / curve[:-1]))
    if not pieces:
        return np.array([], dtype=np.float64)
    return np.concatenate(pieces)


@dataclass(frozen=True)
class _DataBlockKey:
    """Hashable view of the fields that determine the bar index.

    ``cache_dir`` is intentionally excluded — different on-disk caches
    that yield identical bars are fine.
    """

    source_name: str
    source_params: tuple[tuple[str, object], ...]
    tickers: tuple[str, ...]
    start: datetime
    end: datetime
    interval: str

    @classmethod
    def from_data_config(cls, data: DataConfig) -> _DataBlockKey:
        return cls(
            source_name=data.source.name,
            source_params=tuple(sorted(data.source.params.items())),
            tickers=tuple(data.tickers),
            start=data.start,
            end=data.end,
            interval=data.interval.value,
        )


def _validate_uniform_data(configs: tuple[ExperimentConfig, ...]) -> None:
    head_key = _DataBlockKey.from_data_config(configs[0].data)
    for cfg in configs[1:]:
        other_key = _DataBlockKey.from_data_config(cfg.data)
        if other_key != head_key:
            mismatched = [
                f"{field}: {getattr(head_key, field)!r} vs {getattr(other_key, field)!r}"
                for field in ("source_name", "source_params", "tickers", "start", "end", "interval")
                if getattr(head_key, field) != getattr(other_key, field)
            ]
            raise ValueError(
                f"regime overlay requires every config to share the same data block; "
                f"config '{cfg.name}' differs from '{configs[0].name}' on: "
                f"{', '.join(mismatched)}. Fix by aligning the data: section across "
                f"all configs or omit --regime-config."
            )


def _fetch_overlay_bars(data_cfg: DataConfig) -> pd.DataFrame:
    """Fetch the bar frame the regime overlay tags against.

    Hoisted into a module-level helper so tests can monkeypatch it
    without spinning up the data-source registry — production callers
    never substitute it.
    """
    data_source = data_source_registry.create_from_config(data_cfg.source)
    return data_source.fetch(
        data_cfg.tickers[0],
        data_cfg.start,
        data_cfg.end,
        data_cfg.interval,
    )


def _compute_regime_overlay(
    strategy_names: tuple[str, ...],
    results: list[ExperimentResult],
    *,
    regime_config: RegimeConfig,
    data_cfg: DataConfig,
) -> dict[str, dict[str, AggregateStats]]:
    bars = _fetch_overlay_bars(data_cfg)
    refetched_hash = fingerprint_bars(bars)
    detector = regime_registry.create_from_config(regime_config.detector)
    # Tagging is the heavy step; reuse the same series across every
    # strategy since they all share the same bars (validated above).
    tagged = detector.tag(bars)

    overlay: dict[str, dict[str, AggregateStats]] = {}
    for name, result in zip(strategy_names, results, strict=True):
        assert_data_hash_matches(
            refetched_hash,
            result.manifest.data_hash,
            context=f"compare regime overlay for strategy '{name}'",
        )
        overlay[name] = aggregate_split(split_folds_by_tags(result.folds, tagged))
    return overlay


def _validate_reused_results_alignment(
    inputs: _ComparisonInputs,
    reused_results: Sequence[ExperimentResult],
) -> None:
    """Cross-check reused runs against the configs they're paired with.

    Three invariants:
    * Count matches: one reused result per config, no off-by-one.
    * Strategy name matches: the reused run's ``manifest.name`` was set
      from ``cfg.name`` at write time, so a mismatch means the user
      passed the wrong run dir for that config slot.
    * ``manifest.data_hash`` is uniform across reused runs: the pairwise
      bootstrap pairs bar-aligned per-fold returns; differing data hashes
      mean differing underlying bars and the pairing is meaningless.
    """
    if len(reused_results) != len(inputs.configs):
        raise ValueError(
            f"reused_results count ({len(reused_results)}) does not match "
            f"configs count ({len(inputs.configs)}); fix by passing one "
            f"--reuse-runs path per --config in matching order."
        )
    for cfg_name, result in zip(inputs.strategy_names, reused_results, strict=True):
        if result.manifest.name != cfg_name:
            raise ValueError(
                f"reused run strategy name '{result.manifest.name}' does not "
                f"match paired config name '{cfg_name}'; --reuse-runs paths "
                f"must be in the same order as --config paths."
            )
    head_hash = reused_results[0].manifest.data_hash
    for name, result in zip(inputs.strategy_names[1:], reused_results[1:], strict=True):
        if result.manifest.data_hash != head_hash:
            raise ValueError(
                f"reused run for '{name}' has data_hash {result.manifest.data_hash} "
                f"but '{inputs.strategy_names[0]}' has {head_hash}; pairwise "
                f"bootstrap requires every reused run to share the same bar "
                f"index. Fix by re-running the experiments under aligned "
                f"data.start / data.end / data.interval."
            )


def _validate_fold_alignment(
    strategy_names: tuple[str, ...],
    results: list[ExperimentResult],
) -> None:
    """Every strategy must have the same fold count + same per-fold curve lengths.

    Fold curve lengths are the per-bar granularity; aligning the bootstrap
    requires the underlying bar index to match 1-to-1 across strategies,
    which in practice means same data / same validator / same holdout.
    """
    reference_fold_count = len(results[0].folds)
    reference_curve_lengths = tuple(len(fold.equity_curve) for fold in results[0].folds)
    for name, result in zip(strategy_names[1:], results[1:], strict=True):
        if len(result.folds) != reference_fold_count:
            raise ValueError(
                f"strategy '{name}' has {len(result.folds)} folds but "
                f"'{strategy_names[0]}' has {reference_fold_count}; pairwise "
                f"bootstrap requires aligned folds. Fix by running every strategy "
                f"under the same validation config (n_splits / train_size / test_size)."
            )
        lengths = tuple(len(fold.equity_curve) for fold in result.folds)
        if lengths != reference_curve_lengths:
            raise ValueError(
                f"strategy '{name}' fold curve lengths {lengths} do not match "
                f"'{strategy_names[0]}' lengths {reference_curve_lengths}; pairwise "
                f"bootstrap requires bar-aligned fold windows. Fix by aligning "
                f"data.start / data.end / data.interval across all configs."
            )
