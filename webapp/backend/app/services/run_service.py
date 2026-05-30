"""
Read-only services for the persisted run tree.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

# Use the libyaml-backed CSafeLoader when available - PyYAML's pure-Python
# loader is ~10x slower and dominates the list-endpoint cold pass over a
# few thousand runs. Falls back transparently on installs without libyaml.
try:
    from yaml import CSafeLoader as _SafeLoader
except ImportError:  # pragma: no cover - depends on libyaml presence
    from yaml import SafeLoader as _SafeLoader  # type: ignore[assignment]

from src.analysis.feature_importance import AggregatedImportance, read_aggregated_importance
from src.core import json_io
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    EXPERIMENT_METRICS_JSON,
    FEATURE_IMPORTANCE_JSON,
    read_experiment_manifest,
)
from src.orchestration.run_loader import (
    load_experiment_config_from_run,
    load_experiment_result,
)
from webapp.backend.app.infrastructure.store import (
    RunNotFoundError,
    find_run_dir,
    iter_run_dirs,
    store_label,
)
from webapp.backend.app.schemas.runs import (
    FeatureImportanceEntry,
    FeatureImportanceResponse,
    FoldRow,
    RunDetail,
    RunSortBy,
    RunsPage,
    RunSummary,
    SortOrder,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services._dir_cache import (
    cached_artifact_index,
    warm_index,
)
from webapp.backend.app.services.ownership import (
    ArtifactAccessDeniedError,
    check_artifact_access,
    resolve_owner_usernames,
    scope_and_stamp_summaries,
)
from webapp.backend.app.services.plots import (
    PLOTS_DIRNAME,
    PlotNotFoundError,
    list_plots,
    resolve_plot_path,
)

logger = logging.getLogger(__name__)

# Cache RunSummary by (run_dir_str, manifest_mtime_ns) to skip the expensive
# per-run config.yaml + metrics.json reads on every list call. Once a manifest
# has been written it's effectively immutable, so mtime invalidation catches
# the only legitimate rewrites (holdout-eval metric back-writes).
_SUMMARY_CACHE: dict[str, tuple[int, RunSummary]] = {}

_RUN_KIND = "run"


__all__ = [
    "ArtifactAccessDeniedError",
    "PlotNotFoundError",
    "RunNotFoundError",
    "get_feature_importance",
    "get_folds",
    "get_run",
    "list_runs",
    "list_runs_page",
    "resolve_plot",
]

_FEATURE_IMPORTANCE_NOT_COMPUTED_MESSAGE = "Feature importance was not computed for this run."


# Per-run summarization is dominated by file I/O (manifest + metrics + config).
# A small thread pool overlaps the syscall-bound portions; raising the count
# beyond 4 hurts because YAML/JSON parsing itself is CPU-bound and contends
# for the GIL.
_LIST_WORKER_COUNT = 4


def list_runs(
    root: Path,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
    all_users: bool,
) -> list[RunSummary]:
    """
    List every run under ``root`` visible to ``user``, newest first.

    Runs missing ``config.yaml`` are skipped (they cannot populate the
    strategy/tickers/interval columns); runs missing ``metrics.json``
    surface with ``None`` aggregates. The walker keys on
    ``manifest.json``, so partial runs without one never appear at all.
    """

    summaries = scope_and_stamp_summaries(
        _summarize_all(root),
        key_fn=lambda s: s.experiment_id,
        conn=conn,
        user=user,
        all_users=all_users,
    )
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def _summarize_all(root: Path) -> list[RunSummary]:
    """
    All summaries, unsorted. Sort is the caller's job.

    Per-run summarization runs in a thread pool - the work is dominated by
    blocking file reads (manifest + metrics + config), which release the GIL,
    so threading gives a near-linear speedup on the cold pass over thousands
    of runs.
    """

    run_dirs, _ = cached_artifact_index(root, _RUN_KIND, iter_run_dirs)
    with ThreadPoolExecutor(max_workers=_LIST_WORKER_COUNT) as pool:
        results = pool.map(lambda d: _safe_summarize(d, root), run_dirs)
    return [s for s in results if s is not None]


def _safe_summarize(run_dir: Path, root: Path) -> RunSummary | None:
    try:
        return _cached_summarize(run_dir, root)
    except Exception as exc:  # noqa: BLE001 - one bad run must not 500 the whole listing
        logger.warning("skipping unreadable run at %s: %s", run_dir, exc)
        return None


def _lookup_run_dir(root: Path, experiment_id: str) -> Path:
    """
    Resolve ``experiment_id`` via the cached id index; fall back to a glob.

    Glob fallback handles freshly written runs that the path-cache snapshot
    pre-dates. On hit, the id-index is warmed in place so successive
    lookups within the same TTL window skip the glob.
    """

    _, id_index = cached_artifact_index(root, _RUN_KIND, iter_run_dirs)
    hit = id_index.get(experiment_id)
    if hit is not None and hit.is_dir():
        return hit
    resolved = find_run_dir(root, experiment_id)
    warm_index(root, _RUN_KIND, experiment_id, resolved)
    return resolved


def _cached_summarize(run_dir: Path, root: Path) -> RunSummary:
    """
    ``_summarize`` with manifest-mtime invalidation.

    A run dir is identified by its absolute path; once the manifest has been
    written it's effectively immutable, so any change in mtime invalidates the
    cached summary (covers re-runs that reuse a directory, or metrics being
    rewritten after a holdout eval). New runs always miss the cache on first
    visit.
    """

    manifest_path = run_dir / EXPERIMENT_MANIFEST_JSON
    key = str(run_dir)
    mtime = manifest_path.stat().st_mtime_ns
    cached = _SUMMARY_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    summary = _summarize(run_dir, root)
    _SUMMARY_CACHE[key] = (mtime, summary)
    return summary


def list_runs_page(
    root: Path,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
    all_users: bool,
    limit: int,
    offset: int,
    sort_by: RunSortBy,
    order: SortOrder,
    strategy: str | None = None,
    ticker: str | None = None,
    since: datetime | None = None,
) -> RunsPage:
    """
    Paginated + sorted + filtered run listing.

    The full run set is scanned and summarised before slicing - there is no
    cheaper way today since per-run metadata lives in two files per directory.
    Filters and sort happen in-memory over the summary list; ``limit``/``offset``
    pick the page that is returned to the client.
    """

    all_rows = scope_and_stamp_summaries(
        _summarize_all(root),
        key_fn=lambda s: s.experiment_id,
        conn=conn,
        user=user,
        all_users=all_users,
    )
    filtered = [r for r in all_rows if _matches_filters(r, strategy, ticker, since)]
    filtered.sort(key=_sort_key(sort_by), reverse=(order is SortOrder.DESC))

    page = filtered[offset : offset + limit]
    return RunsPage(items=page, total=len(filtered), limit=limit, offset=offset)


def _matches_filters(
    row: RunSummary, strategy: str | None, ticker: str | None, since: datetime | None
) -> bool:
    if strategy is not None and row.strategy != strategy:
        return False
    if ticker is not None and ticker not in row.tickers:
        return False
    if since is not None and row.created_at < since:
        return False
    return True


def _sort_key(sort_by: RunSortBy) -> Callable[[RunSummary], float]:
    # Metric-based sorts treat missing values as the worst possible so they
    # sink to the bottom under DESC (the usual "best first" ordering). The
    # created_at branch returns its POSIX timestamp so every branch has the
    # same return type and mypy can resolve the comparator unambiguously.
    if sort_by is RunSortBy.CREATED_AT:
        return lambda r: r.created_at.timestamp()
    if sort_by is RunSortBy.SHARPE_MEAN:
        return lambda r: r.sharpe_mean if r.sharpe_mean is not None else float("-inf")
    return lambda r: r.calmar_mean if r.calmar_mean is not None else float("-inf")


def _ensure_plots(run_dir: Path) -> None:
    """
    Render the canonical static plots into ``<run_dir>/plots/`` if missing.

    Idempotent: no-op if any plot file already exists, or if fold data is
    unavailable (partial/aborted runs surface via the empty PlotIndex).
    """

    plots_dir = run_dir / PLOTS_DIRNAME
    if plots_dir.is_dir() and any(plots_dir.iterdir()):
        return
    try:
        result = load_experiment_result(run_dir)
    except FileNotFoundError:
        return
    from src.visualization.strategy_reporter import StrategyReporter

    StrategyReporter().generate_full_report(result, run_dir)


def get_run(
    root: Path,
    experiment_id: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> RunDetail:
    """
    Read the full detail payload for one run.

    Plot generation is NOT triggered here - it's deferred to ``resolve_plot``,
    which renders lazily on the first plot fetch. The detail page returns
    immediately; the matplotlib startup + render cost only pays itself when
    the user actually clicks a plot link.

    In-flight HPO trial runs land ``config.yaml + manifest.json`` first and
    write ``metrics.json`` only after the walk-forward completes; surface them
    with empty ``metrics`` rather than 500-ing so the detail page agrees with
    the listing.
    """

    check_artifact_access(conn, experiment_id=experiment_id, user=user)
    run_dir = _lookup_run_dir(root, experiment_id)
    manifest = read_experiment_manifest(run_dir)
    config = load_experiment_config_from_run(run_dir)

    try:
        metrics = _read_metrics(run_dir)
    except FileNotFoundError:
        metrics = {}
    usernames = resolve_owner_usernames(conn, experiment_ids=[experiment_id])

    return RunDetail(
        experiment_id=manifest.experiment_id,
        name=manifest.name,
        strategy=config.strategy.name,
        tickers=list(config.data.tickers),
        interval=config.data.interval.value,
        store=store_label(run_dir, root),
        created_at=manifest.created_at,
        git_sha=manifest.git_sha,
        seed=manifest.seed,
        data_hash=manifest.data_hash,
        slippage_scenario=manifest.slippage_scenario,
        holdout_start=manifest.holdout_start,
        metrics=metrics,
        plots=list_plots(run_dir),
        launched_by_username=usernames.get(experiment_id),
    )


def get_folds(
    root: Path,
    experiment_id: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> list[FoldRow]:
    """
    Read per-fold metric rows for one run.
    """

    check_artifact_access(conn, experiment_id=experiment_id, user=user)
    run_dir = _lookup_run_dir(root, experiment_id)
    result = load_experiment_result(run_dir)

    return [
        FoldRow(
            fold_index=f.fold_index,
            train_start=f.train_start,
            train_end=f.train_end,
            test_start=f.test_start,
            test_end=f.test_end,
            total_return=f.total_return,
            annualized_return=f.annualized_return,
            annualized_volatility=f.annualized_volatility,
            sharpe_ratio=f.sharpe_ratio,
            sortino_ratio=f.sortino_ratio,
            calmar_ratio=f.calmar_ratio,
            max_drawdown=f.max_drawdown,
            win_rate=f.win_rate,
            trade_count=f.trade_count,
            equity_curve=list(f.equity_curve),
        )
        for f in result.folds
    ]


def get_feature_importance(
    root: Path,
    experiment_id: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> FeatureImportanceResponse:
    """
    Read cross-fold feature importance for one run.

    Returns ``entries=[]`` plus a ``message`` (200, not 404) when the run has
    no ``feature_importance.json``. That is the common case: importance is
    opt-in per run, rule-based strategies emit none, and pre-importance runs
    predate the artifact. A missing RUN (as opposed to a present run that
    simply lacks the artifact) raises ``RunNotFoundError`` from the dir
    lookup, which the route maps to 404.
    """

    check_artifact_access(conn, experiment_id=experiment_id, user=user)
    run_dir = _lookup_run_dir(root, experiment_id)
    try:
        payload = json_io.read_dict(run_dir / FEATURE_IMPORTANCE_JSON)
    except FileNotFoundError:
        return FeatureImportanceResponse(
            entries=[], message=_FEATURE_IMPORTANCE_NOT_COMPUTED_MESSAGE
        )
    entries = [_entry_from_aggregated(agg) for agg in read_aggregated_importance(payload)]
    return FeatureImportanceResponse(entries=entries)


def _entry_from_aggregated(agg: AggregatedImportance) -> FeatureImportanceEntry:
    return FeatureImportanceEntry(
        feature=agg.feature,
        importance=_json_safe_float(agg.importance),
        std=_json_safe_float(agg.std),
        n_folds=agg.n_folds,
        method=agg.method,
    )


def _json_safe_float(value: float) -> float | None:
    """
    Non-finite -> ``None`` so the value survives Starlette's ``allow_nan=False`` render.

    ``allow_nan=False`` rejects ``+/-inf`` as well as ``NaN``, and an artifact
    written before the write-time guard can still carry a raw ``Infinity`` token,
    so guard on ``isfinite`` rather than ``isnan``.
    """

    return None if not math.isfinite(value) else value


def resolve_plot(
    root: Path,
    experiment_id: str,
    plot_name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> Path:
    """
    Resolve a plot filename to an absolute path, blocking ``..`` traversal.

    Lazily renders missing plots on first access (covers direct/bookmarked
    plot URLs that bypass ``get_run``).
    """

    check_artifact_access(conn, experiment_id=experiment_id, user=user)
    run_dir = _lookup_run_dir(root, experiment_id)
    try:
        return resolve_plot_path(run_dir, plot_name)
    except PlotNotFoundError:
        _ensure_plots(run_dir)
        return resolve_plot_path(run_dir, plot_name)


def _summarize(run_dir: Path, root: Path) -> RunSummary:
    manifest = read_experiment_manifest(run_dir)
    strategy, tickers, interval = _read_config_summary(run_dir)

    try:
        metrics = _read_metrics(run_dir)
    except FileNotFoundError:
        metrics = {}

    return RunSummary(
        experiment_id=manifest.experiment_id,
        name=manifest.name,
        strategy=strategy,
        tickers=tickers,
        interval=interval,
        store=store_label(run_dir, root),
        created_at=manifest.created_at,
        sharpe_mean=metrics.get("sharpe_mean"),
        calmar_mean=metrics.get("calmar_mean"),
        has_holdout=manifest.holdout_start is not None,
        data_hash=manifest.data_hash,
    )


def _read_config_summary(run_dir: Path) -> tuple[str, list[str], str]:
    """
    Pluck strategy name, tickers, interval from config.yaml without Pydantic validation.

    The list endpoint runs this per row (thousands of rows); skipping the full
    ``ExperimentConfig.model_validate`` saves the bulk of the cold-pass latency
    because that validator instantiates every nested sub-model. Falls back to
    the validated path on any structural surprise so a single legacy or
    partially-written ``config.yaml`` cannot 500 the entire list endpoint.
    """

    try:
        with (run_dir / EXPERIMENT_CONFIG_YAML).open(encoding="utf-8") as f:
            raw = yaml.load(f, Loader=_SafeLoader)  # noqa: S506 - _SafeLoader is the safe loader
        return (
            str(raw["strategy"]["name"]),
            [str(t) for t in raw["data"]["tickers"]],
            str(raw["data"]["interval"]),
        )
    except (KeyError, TypeError, yaml.YAMLError):
        config = load_experiment_config_from_run(run_dir)
        return config.strategy.name, list(config.data.tickers), config.data.interval.value


def _read_metrics(run_dir: Path) -> dict[str, float]:
    raw = json_io.read_dict(run_dir / EXPERIMENT_METRICS_JSON)
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
