"""
Read-only services for the persisted comparisons tree.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
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
    ComparisonsPage,
    ComparisonSummary,
    PerStrategyStatsRow,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services._dir_cache import cached_artifact_dirs
from webapp.backend.app.services.ownership import (
    ArtifactAccessDeniedError,
    check_artifact_access,
    resolve_owner_usernames,
    scope_and_stamp_summaries,
)
from webapp.backend.app.services.plots import (
    PlotNotFoundError,
    list_plots,
    resolve_plot_path,
)

__all__ = [
    "ArtifactAccessDeniedError",
    "ComparisonNotFoundError",
    "PlotNotFoundError",
    "get_comparison",
    "list_comparisons",
    "list_comparisons_page",
    "resolve_plot",
]


def _scoped_summaries(
    root: Path,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
    all_users: bool,
) -> list[ComparisonSummary]:
    summaries: list[ComparisonSummary] = []
    for cmp_dir in cached_artifact_dirs(root, "comparison", iter_comparison_dirs):
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
    return scope_and_stamp_summaries(
        summaries, key_fn=lambda s: s.name, conn=conn, user=user, all_users=all_users
    )


def list_comparisons(
    root: Path,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
    all_users: bool,
) -> list[ComparisonSummary]:
    """
    List every comparison under ``root`` visible to ``user``, newest first.
    """

    scoped = _scoped_summaries(root, conn=conn, user=user, all_users=all_users)
    scoped.sort(key=lambda s: s.created_at, reverse=True)
    return scoped


def list_comparisons_page(
    root: Path,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
    all_users: bool,
    limit: int,
    offset: int,
    strategy: str | None = None,
    since: datetime | None = None,
) -> ComparisonsPage:
    """
    Paginated + filtered comparison listing, newest first.

    ``strategies`` is computed over the full visible set before filtering so
    the dropdown can offer every strategy regardless of the current page.
    """

    scoped = _scoped_summaries(root, conn=conn, user=user, all_users=all_users)
    strategies = sorted({s for row in scoped for s in row.strategies})
    filtered = [row for row in scoped if _matches(row, strategy, since)]
    filtered.sort(key=lambda s: s.created_at, reverse=True)
    page = filtered[offset : offset + limit]
    return ComparisonsPage(
        items=page, total=len(filtered), limit=limit, offset=offset, strategies=strategies
    )


def _matches(row: ComparisonSummary, strategy: str | None, since: datetime | None) -> bool:
    if strategy is not None and strategy not in row.strategies:
        return False
    if since is not None and row.created_at < since:
        return False
    return True


def get_comparison(
    root: Path,
    name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> ComparisonDetail:
    """
    Read the full detail payload for one comparison.

    Raises :class:`ArtifactAccessDeniedError` when ``user`` is neither owner
    nor admin; the router maps that to 404 so the response doesn't disclose
    that the comparison exists.
    """

    check_artifact_access(conn, experiment_id=name, user=user)
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
    usernames = resolve_owner_usernames(conn, experiment_ids=[name])
    return ComparisonDetail(
        name=json_io.get_str(manifest, "out_name"),
        store=store_label(cmp_dir, root),
        created_at=json_io.get_timestamp(manifest, "created_at"),
        per_strategy_stats=rows,
        plots=list_plots(cmp_dir),
        launched_by_username=usernames.get(name),
    )


def resolve_plot(
    root: Path,
    name: str,
    plot_name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> Path:
    """
    Resolve a comparison plot filename to an absolute path, blocking traversal.
    """

    check_artifact_access(conn, experiment_id=name, user=user)
    return resolve_plot_path(find_comparison_dir(root, name), plot_name)
