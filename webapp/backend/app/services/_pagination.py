"""
Shared pagination + optional-key sorting helpers for the list services.

The list endpoints scan and summarise the full visible artifact set, then
filter/sort/slice it in memory. These helpers centralise the two pieces that
were previously copy-pasted across every ``list_*_page`` service: the
offset/limit slice (with the pre-slice total) and the null-last sort over an
optional metric.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from webapp.backend.app.schemas.pagination import SortOrder


def matches_since(timestamp: datetime, since: datetime | None) -> bool:
    """
    True unless ``since`` is set and ``timestamp`` falls before it.

    The list endpoints' ``since`` filter keeps only rows at or after the cutoff
    (created/started on the day or later); a ``None`` cutoff disables it.

    Stored timestamps are tz-aware (``datetime.now(UTC)``), but an API caller
    can pass an offset-less ``?since=2024-01-01T00:00:00`` that parses tz-naive;
    comparing the two raises ``TypeError``. A naive cutoff is read as UTC so the
    filter degrades gracefully instead of 500-ing the listing. (The UI always
    sends ``.toISOString()``, so this only bites hand-built requests.)
    """

    if since is None:
        return True
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    return timestamp >= since


def paginate[T](rows: list[T], *, limit: int, offset: int) -> tuple[list[T], int]:
    """
    Slice ``rows`` to one page and report the pre-slice total.

    Returns ``(rows[offset : offset + limit], len(rows))`` - the total is the
    full filtered count so the client can render "showing x-y of N" and decide
    whether a next page exists.
    """

    return rows[offset : offset + limit], len(rows)


def sort_by_optional[T](
    rows: list[T], key: Callable[[T], float | None], *, order: SortOrder
) -> None:
    """
    In-place sort by an optional numeric ``key``, sinking ``None`` last.

    Rows whose key is ``None`` (an in-flight metric not computed yet) sink to
    the bottom under BOTH sort directions: direction is folded into the value's
    sign rather than ``reverse=`` so the null-last primary key is never flipped.
    Ascending therefore surfaces the weakest real results first instead of a
    wall of nulls.
    """

    sign = -1.0 if order is SortOrder.DESC else 1.0

    def sort_key(row: T) -> tuple[bool, float]:
        value = key(row)
        return (value is None, sign * value if value is not None else 0.0)

    rows.sort(key=sort_key)
