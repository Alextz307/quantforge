"""
Read-only services for the persisted holdout-evaluations tree.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
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
from webapp.backend.app.schemas.holdout import (
    HoldoutEvalDetail,
    HoldoutEvalsPage,
    HoldoutEvalSummary,
    HoldoutSortBy,
)
from webapp.backend.app.schemas.pagination import SortOrder
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services._dir_cache import cached_artifact_dirs
from webapp.backend.app.services._pagination import matches_since, paginate, sort_by_optional
from webapp.backend.app.services.ownership import (
    ArtifactAccessDeniedError,
    check_artifact_access,
    resolve_owner_usernames,
    scoped_cached_summaries,
    stamp_summaries,
)
from webapp.backend.app.services.plots import (
    PlotNotFoundError,
    list_plots,
    resolve_plot_path,
)

__all__ = [
    "ArtifactAccessDeniedError",
    "HoldoutEvalNotFoundError",
    "PlotNotFoundError",
    "get_holdout_eval",
    "list_holdout_evals_page",
    "resolve_plot",
]


def _optional_metric(metrics: object, key: str) -> float | None:
    """
    Pull a numeric metric from a ``metrics`` block tolerant of missing/typed-wrong entries.
    """

    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if not isinstance(value, int | float):
        return None
    return float(value)


# Cache HoldoutEvalSummary by (eval_dir, holdout_eval.json mtime_ns) so
# successive page/sort/filter requests reparse only evals written since.
_SUMMARY_CACHE: dict[str, tuple[int, HoldoutEvalSummary | None]] = {}


def _summarize(eval_dir: Path, root: Path) -> HoldoutEvalSummary:
    payload = json_io.read_dict(eval_dir / HOLDOUT_EVAL_JSON)
    return HoldoutEvalSummary(
        name=json_io.get_str(payload, "out_name"),
        store=store_label(eval_dir, root),
        created_at=json_io.get_timestamp(payload, "created_at"),
        source_kind=cast(SourceKind, json_io.get_str(payload, "source_kind")),
        source_id=json_io.get_str(payload, "source_id"),
        holdout_start=json_io.get_timestamp(payload, "holdout_start"),
        sharpe_ratio=_optional_metric(payload.get("metrics"), "sharpe_ratio"),
    )


def list_holdout_evals_page(
    root: Path,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
    all_users: bool,
    limit: int,
    offset: int,
    sort_by: HoldoutSortBy,
    order: SortOrder,
    source_kind: SourceKind | None = None,
    since: datetime | None = None,
) -> HoldoutEvalsPage:
    """
    Paginated + sorted + filtered holdout-eval listing.

    ``source_kinds`` is computed over the full visible set before filtering so
    the dropdown can offer every kind regardless of the current page.
    """

    visible, usernames = scoped_cached_summaries(
        cached_artifact_dirs(root, "holdout", iter_holdout_eval_dirs),
        mtime_sources=(HOLDOUT_EVAL_JSON,),
        summarize=lambda d: _summarize(d, root),
        cache=_SUMMARY_CACHE,
        key_fn=lambda s: s.name,
        conn=conn,
        user=user,
        all_users=all_users,
    )

    source_kinds = sorted({s.source_kind for s in visible})
    filtered = [s for s in visible if _matches(s, source_kind, since)]
    _sort(filtered, sort_by, order)

    page, total = paginate(filtered, limit=limit, offset=offset)
    items = stamp_summaries(page, key_fn=lambda s: s.name, usernames=usernames)
    return HoldoutEvalsPage(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        source_kinds=source_kinds,
    )


def _matches(
    row: HoldoutEvalSummary, source_kind: SourceKind | None, since: datetime | None
) -> bool:
    if source_kind is not None and row.source_kind != source_kind:
        return False
    return matches_since(row.created_at, since)


def _sort(rows: list[HoldoutEvalSummary], sort_by: HoldoutSortBy, order: SortOrder) -> None:
    reverse = order is SortOrder.DESC
    if sort_by is HoldoutSortBy.CREATED_AT:
        rows.sort(key=lambda r: r.created_at, reverse=reverse)
        return
    if sort_by is HoldoutSortBy.HOLDOUT_START:
        rows.sort(key=lambda r: r.holdout_start, reverse=reverse)
        return
    # In-flight evals carry ``sharpe_ratio=None``; sink them last under BOTH
    # directions (see sort_by_optional).
    sort_by_optional(rows, lambda r: r.sharpe_ratio, order=order)


def get_holdout_eval(
    root: Path,
    name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> HoldoutEvalDetail:
    """
    Read the full detail payload for one holdout eval.

    Raises :class:`ArtifactAccessDeniedError` when ``user`` is neither owner
    nor admin; the router maps that to 404.
    """

    check_artifact_access(conn, experiment_id=name, user=user)
    eval_dir = find_holdout_eval_dir(root, name)
    payload = json_io.read_dict(eval_dir / HOLDOUT_EVAL_JSON)
    metrics = json_io.get_dict(payload, "metrics")
    usernames = resolve_owner_usernames(conn, experiment_ids=[name])

    return HoldoutEvalDetail(
        name=json_io.get_str(payload, "out_name"),
        store=store_label(eval_dir, root),
        created_at=json_io.get_timestamp(payload, "created_at"),
        source_kind=cast(SourceKind, json_io.get_str(payload, "source_kind")),
        source_id=json_io.get_str(payload, "source_id"),
        source_path=json_io.get_str(payload, "source_path"),
        holdout_start=json_io.get_timestamp(payload, "holdout_start"),
        n_dev_bars=json_io.get_int(payload, "n_dev_bars"),
        n_holdout_bars=json_io.get_int(payload, "n_holdout_bars"),
        slippage_scenario=SlippageScenario(json_io.get_str(payload, "slippage_scenario")),
        total_return=json_io.get_float(metrics, "total_return"),
        annualized_return=json_io.get_float(metrics, "annualized_return"),
        sharpe_ratio=json_io.get_float(metrics, "sharpe_ratio"),
        sortino_ratio=json_io.get_float(metrics, "sortino_ratio"),
        calmar_ratio=json_io.get_float(metrics, "calmar_ratio"),
        max_drawdown=json_io.get_float(metrics, "max_drawdown"),
        win_rate=json_io.get_float(metrics, "win_rate"),
        trade_count=json_io.get_int(metrics, "trade_count"),
        equity_curve=json_io.get_float_list(payload, "equity_curve"),
        plots=list_plots(eval_dir),
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
    Resolve a holdout-eval plot filename to an absolute path, blocking traversal.
    """

    check_artifact_access(conn, experiment_id=name, user=user)
    return resolve_plot_path(find_holdout_eval_dir(root, name), plot_name)
