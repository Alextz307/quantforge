"""Read-only services for the persisted holdout-evaluations tree."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from src.core import json_io
from src.core.persistence import HOLDOUT_EVAL_JSON
from src.engine.scenarios import SlippageScenario
from src.orchestration.holdout_eval import SourceKind
from webapp.backend.app.infrastructure.store import (
    HoldoutEvalNotFoundError,
    find_holdout_eval_dir,
    iter_holdout_eval_dirs,
    store_label,
)
from webapp.backend.app.schemas.holdout import HoldoutEvalDetail, HoldoutEvalSummary
from webapp.backend.app.services._dir_cache import cached_artifact_dirs
from webapp.backend.app.services.plots import (
    PlotNotFoundError,
    list_plots,
    resolve_plot_path,
)

__all__ = [
    "HoldoutEvalNotFoundError",
    "PlotNotFoundError",
    "get_holdout_eval",
    "list_holdout_evals",
    "resolve_plot",
]


def _optional_metric(metrics: object, key: str) -> float | None:
    """Pull a numeric metric from a ``metrics`` block tolerant of missing/typed-wrong entries."""
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if not isinstance(value, int | float):
        return None
    return float(value)


def list_holdout_evals(root: Path) -> list[HoldoutEvalSummary]:
    """List every holdout eval under ``root``, newest first."""
    summaries: list[HoldoutEvalSummary] = []
    for eval_dir in cached_artifact_dirs(root, "holdout", iter_holdout_eval_dirs):
        payload = json_io.read_dict(eval_dir / HOLDOUT_EVAL_JSON)
        sharpe = _optional_metric(payload.get("metrics"), "sharpe_ratio")
        summaries.append(
            HoldoutEvalSummary(
                name=json_io.get_str(payload, "out_name"),
                store=store_label(eval_dir, root),
                created_at=json_io.get_timestamp(payload, "created_at"),
                source_kind=cast(SourceKind, json_io.get_str(payload, "source_kind")),
                source_id=json_io.get_str(payload, "source_id"),
                holdout_start=json_io.get_timestamp(payload, "holdout_start"),
                sharpe_ratio=sharpe,
            )
        )
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def get_holdout_eval(root: Path, name: str) -> HoldoutEvalDetail:
    """Read the full detail payload for one holdout eval."""
    eval_dir = find_holdout_eval_dir(root, name)
    payload = json_io.read_dict(eval_dir / HOLDOUT_EVAL_JSON)
    metrics = json_io.get_dict(payload, "metrics")
    return HoldoutEvalDetail(
        name=json_io.get_str(payload, "out_name"),
        store=store_label(eval_dir, root),
        created_at=json_io.get_timestamp(payload, "created_at"),
        git_sha=json_io.get_str(payload, "git_sha"),
        source_kind=cast(SourceKind, json_io.get_str(payload, "source_kind")),
        source_id=json_io.get_str(payload, "source_id"),
        source_path=json_io.get_str(payload, "source_path"),
        holdout_start=json_io.get_timestamp(payload, "holdout_start"),
        data_hash=json_io.get_str(payload, "data_hash"),
        n_dev_bars=json_io.get_int(payload, "n_dev_bars"),
        n_holdout_bars=json_io.get_int(payload, "n_holdout_bars"),
        slippage_scenario=SlippageScenario(json_io.get_str(payload, "slippage_scenario")),
        total_return=json_io.get_float(metrics, "total_return"),
        annualized_return=json_io.get_float(metrics, "annualized_return"),
        annualized_volatility=json_io.get_float(metrics, "annualized_volatility"),
        sharpe_ratio=json_io.get_float(metrics, "sharpe_ratio"),
        sortino_ratio=json_io.get_float(metrics, "sortino_ratio"),
        calmar_ratio=json_io.get_float(metrics, "calmar_ratio"),
        max_drawdown=json_io.get_float(metrics, "max_drawdown"),
        win_rate=json_io.get_float(metrics, "win_rate"),
        trade_count=json_io.get_int(metrics, "trade_count"),
        equity_curve=json_io.get_float_list(payload, "equity_curve"),
        plots=list_plots(eval_dir),
    )


def resolve_plot(root: Path, name: str, plot_name: str) -> Path:
    """Resolve a holdout-eval plot filename to an absolute path, blocking traversal."""
    return resolve_plot_path(find_holdout_eval_dir(root, name), plot_name)
