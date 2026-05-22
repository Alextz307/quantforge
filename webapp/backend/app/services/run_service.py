"""Read-only services for the persisted run tree."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core import json_io
from src.core.persistence import EXPERIMENT_METRICS_JSON, read_experiment_manifest
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
    FoldRow,
    PretrainedLeafDTO,
    RunDetail,
    RunSummary,
)
from webapp.backend.app.services.plots import (
    PLOTS_DIRNAME,
    PlotNotFoundError,
    list_plots,
    resolve_plot_path,
)

logger = logging.getLogger(__name__)


__all__ = [
    "PlotNotFoundError",
    "RunNotFoundError",
    "get_folds",
    "get_run",
    "list_runs",
    "resolve_plot",
]


def list_runs(root: Path) -> list[RunSummary]:
    """List every run under ``root``, newest first.

    Runs missing ``config.yaml`` are skipped (they cannot populate the
    strategy/tickers/interval columns); runs missing ``metrics.json``
    surface with ``None`` aggregates. The walker keys on
    ``manifest.json``, so partial runs without one never appear at all.
    """
    summaries: list[RunSummary] = []
    for run_dir in iter_run_dirs(root):
        try:
            summary = _summarize(run_dir, root)
        except Exception as exc:  # noqa: BLE001 — one bad run must not 500 the whole listing
            logger.warning("skipping unreadable run at %s: %s", run_dir, exc)
            continue
        summaries.append(summary)
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def _ensure_plots(run_dir: Path) -> None:
    """Render the canonical static plots into ``<run_dir>/plots/`` if missing.

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


def get_run(root: Path, experiment_id: str) -> RunDetail:
    """Read the full detail payload for one run."""
    run_dir = find_run_dir(root, experiment_id)
    _ensure_plots(run_dir)
    manifest = read_experiment_manifest(run_dir)
    config = load_experiment_config_from_run(run_dir)
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
        pretrained_leaves=[
            PretrainedLeafDTO.model_validate(r.to_dict()) for r in manifest.pretrained_leaves
        ],
        metrics=_read_metrics(run_dir),
        plots=list_plots(run_dir),
    )


def get_folds(root: Path, experiment_id: str) -> list[FoldRow]:
    """Read per-fold metric rows for one run."""
    run_dir = find_run_dir(root, experiment_id)
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


def resolve_plot(root: Path, experiment_id: str, plot_name: str) -> Path:
    """Resolve a plot filename to an absolute path, blocking ``..`` traversal.

    Lazily renders missing plots on first access (covers direct/bookmarked
    plot URLs that bypass ``get_run``).
    """
    run_dir = find_run_dir(root, experiment_id)
    try:
        return resolve_plot_path(run_dir, plot_name)
    except PlotNotFoundError:
        _ensure_plots(run_dir)
        return resolve_plot_path(run_dir, plot_name)


def _summarize(run_dir: Path, root: Path) -> RunSummary:
    manifest = read_experiment_manifest(run_dir)
    config = load_experiment_config_from_run(run_dir)
    try:
        metrics = _read_metrics(run_dir)
    except FileNotFoundError:
        metrics = {}
    return RunSummary(
        experiment_id=manifest.experiment_id,
        name=manifest.name,
        strategy=config.strategy.name,
        tickers=list(config.data.tickers),
        interval=config.data.interval.value,
        store=store_label(run_dir, root),
        created_at=manifest.created_at,
        sharpe_mean=metrics.get("sharpe_mean"),
        calmar_mean=metrics.get("calmar_mean"),
    )


def _read_metrics(run_dir: Path) -> dict[str, float]:
    raw = json_io.read_dict(run_dir / EXPERIMENT_METRICS_JSON)
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
