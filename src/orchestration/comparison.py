"""Multi-strategy comparison orchestrator.

Composes multiple :class:`ExperimentConfig` runs into a single
:class:`ComparisonReport`: ranked per-strategy stats, pairwise Sharpe
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
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from src.analysis.metrics_aggregator import aggregate_folds
from src.analysis.ranking import rank_strategies
from src.analysis.significance import paired_bootstrap_sharpe_differential
from src.core.config import ExperimentConfig
from src.core.logging import get_logger
from src.core.persistence import COMPARISONS_SUBDIR
from src.orchestration.builder import build_experiment
from src.orchestration.git_info import read_git_sha
from src.orchestration.types import (
    ComparisonReport,
    ExperimentResult,
    FoldRecord,
    PairwiseSignificance,
)

_logger = get_logger(__name__)

_DEFAULT_STORE_ROOT = Path("experiment_results")

SignificanceTest = Literal["bootstrap", "none"]

# Bootstrap used in pairwise significance. 5k resamples keeps the wall
# clock tolerable for a 10-strategy comparison (45 pairs) while still
# landing 2-decimal-stable percentiles.
_PAIRWISE_N_RESAMPLES = 5_000


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
    significance_test: SignificanceTest = "bootstrap",
) -> tuple[ComparisonReport, dict[str, tuple[FoldRecord, ...]]]:
    """Run every config, aggregate, rank, optionally pairwise-test.

    Returns ``(report, folds_by_strategy)`` — the in-memory
    :class:`ComparisonReport` plus a per-strategy mapping of fold
    records. The reporter's equity-overlay plot consumes the folds;
    the comparison bundle itself stores per-fold data under each
    strategy's ``runs/<experiment_id>/fold_results.jsonl`` so the
    folds dict is a zero-copy hand-off rather than a new persisted
    artifact. Callers that don't need the overlay can discard the
    second tuple element.
    """
    inputs = _validate_inputs(configs)
    store = store_root if store_root is not None else _DEFAULT_STORE_ROOT
    cmp_dir = Path(store) / COMPARISONS_SUBDIR / out_name
    cmp_dir.mkdir(parents=True, exist_ok=True)

    _logger.info(
        "running comparison '%s' with %d configs (n_jobs=%d, significance=%s)",
        out_name,
        len(inputs.configs),
        n_jobs,
        significance_test,
    )

    if n_jobs == 1:
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

    ranking = rank_strategies(per_strategy_stats, by="sharpe")

    if significance_test == "bootstrap":
        pairwise = _compute_pairwise_bootstrap(inputs.strategy_names, results)
    else:
        pairwise = ()

    report = ComparisonReport(
        out_name=out_name,
        created_at=datetime.now(UTC),
        git_sha=read_git_sha(),
        per_strategy_experiment_id=per_strategy_experiment_id,
        per_strategy_stats=per_strategy_stats,
        ranking=ranking,
        pairwise=pairwise,
    )
    folds_by_strategy = {
        name: result.folds for name, result in zip(inputs.strategy_names, results, strict=True)
    }
    return report, folds_by_strategy


def _run_one_experiment(cfg: ExperimentConfig, cmp_dir: Path) -> ExperimentResult:
    """Worker entry point — module-level so ProcessPoolExecutor can pickle it.

    ``write_report=False``: per-strategy reporting is redundant with the
    cross-strategy ComparisonReport and triples the wall clock on a
    five-strategy comparison. Users who want per-strategy plots re-run
    the underlying config via ``experiment run``.

    ``store_root=cmp_dir`` routes per-strategy artifacts under
    ``<cmp_dir>/runs/<experiment_id>/`` (the ``runs/`` subdirectory is
    appended inside :meth:`Experiment.run`) so the comparison bundle
    is self-contained — a user can zip the comparison directory and
    everything travels together.
    """
    experiment = build_experiment(cfg)
    return experiment.run(store_root=cmp_dir, write_report=False)


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
        return [f.result() for f in futures]


def _compute_pairwise_bootstrap(
    strategy_names: tuple[str, ...],
    results: list[ExperimentResult],
) -> tuple[PairwiseSignificance, ...]:
    """Compute the upper-triangular pairwise Sharpe-differential matrix.

    Skips silently when only one strategy is present (the validator
    already blocks that path but a one-strategy comparison has no
    pairwise work). Raises if fold counts or per-fold curve lengths
    differ between any two strategies — that's an alignment violation
    and pairing the bootstrap indices would produce meaningless results.
    """
    _validate_fold_alignment(strategy_names, results)
    per_strategy_returns = {
        name: _concatenated_log_returns(result.folds)
        for name, result in zip(strategy_names, results, strict=True)
    }

    pairwise: list[PairwiseSignificance] = []
    for i in range(len(strategy_names)):
        for j in range(i + 1, len(strategy_names)):
            name_a = strategy_names[i]
            name_b = strategy_names[j]
            ci = paired_bootstrap_sharpe_differential(
                per_strategy_returns[name_a],
                per_strategy_returns[name_b],
                n_resamples=_PAIRWISE_N_RESAMPLES,
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
        pieces.append(np.diff(np.log(curve)))
    if not pieces:
        return np.array([], dtype=np.float64)
    return np.concatenate(pieces)


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
