"""Read-only services for the persisted comparisons tree."""

from __future__ import annotations

from pathlib import Path

from src.core import json_io
from src.core.persistence import EXPERIMENT_MANIFEST_JSON
from webapp.backend.app.infrastructure.store import (
    ComparisonNotFoundError,
    find_comparison_dir,
    iter_comparison_dirs,
    store_label,
)
from webapp.backend.app.schemas.comparisons import (
    ComparisonDetail,
    ComparisonSummary,
    PerStrategyStatsRow,
)
from webapp.backend.app.services.plots import (
    PlotNotFoundError,
    list_plots,
    resolve_plot_path,
)

__all__ = [
    "ComparisonNotFoundError",
    "PlotNotFoundError",
    "get_comparison",
    "list_comparisons",
    "resolve_plot",
]


def list_comparisons(root: Path) -> list[ComparisonSummary]:
    """List every comparison under ``root``, newest first."""
    summaries: list[ComparisonSummary] = []
    for cmp_dir in iter_comparison_dirs(root):
        manifest = json_io.read_dict(cmp_dir / EXPERIMENT_MANIFEST_JSON)
        per_strategy = json_io.get_dict(manifest, "per_strategy_experiment_id")
        summaries.append(
            ComparisonSummary(
                name=json_io.get_str(manifest, "out_name"),
                store=store_label(cmp_dir, root),
                created_at=json_io.get_timestamp(manifest, "created_at"),
                strategies=sorted(per_strategy.keys()),
            )
        )
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def get_comparison(root: Path, name: str) -> ComparisonDetail:
    """Read the full detail payload for one comparison."""
    cmp_dir = find_comparison_dir(root, name)
    manifest = json_io.read_dict(cmp_dir / EXPERIMENT_MANIFEST_JSON)
    per_strategy_stats = json_io.get_dict(manifest, "per_strategy_stats")
    per_strategy_eid = json_io.get_dict(manifest, "per_strategy_experiment_id")
    rows = sorted(
        (
            PerStrategyStatsRow.model_validate(
                {
                    **json_io.get_dict(per_strategy_stats, strategy),
                    "strategy": strategy,
                    "experiment_id": json_io.get_str(per_strategy_eid, strategy),
                }
            )
            for strategy in per_strategy_stats
        ),
        key=lambda r: r.strategy,
    )
    return ComparisonDetail(
        name=json_io.get_str(manifest, "out_name"),
        store=store_label(cmp_dir, root),
        created_at=json_io.get_timestamp(manifest, "created_at"),
        git_sha=json_io.get_str(manifest, "git_sha"),
        per_strategy_stats=rows,
        plots=list_plots(cmp_dir),
    )


def resolve_plot(root: Path, name: str, plot_name: str) -> Path:
    """Resolve a comparison plot filename to an absolute path, blocking traversal."""
    return resolve_plot_path(find_comparison_dir(root, name), plot_name)
