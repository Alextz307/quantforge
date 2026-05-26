"""Owner-resolution + access-check helpers for artifact-read endpoints.

Artifacts (studies, runs, hpo studies, holdout-evals, comparisons) live on
disk under ``store_root``. The ``jobs`` table is the authoritative link:
every webapp-launched artifact has a row with ``experiment_id`` matching
the artifact's identifier. Owner = that row's ``user_id``.

Resolution rules:

* **Match in jobs table** -> owner is the job's user.
* **No match** (CLI-launched artifacts, legacy sweeps without a webapp
  job row, manual copies) -> "ownerless" -> visible to every logged-in
  user.

Access rules:

* Owner sees own artifact.
* Admin sees all (regardless of owner).
* Non-owner sees only ownerless artifacts.

The helpers raise ``ArtifactAccessDeniedError`` on a denied access — the
router maps that to ``HTTPException(404)`` so we don't leak the existence
of someone else's artifact.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.users import UserPublic

# Conservative cap on the IN-clause arity. SQLite's
# ``SQLITE_LIMIT_VARIABLE_NUMBER`` was 999 in pre-3.32 builds and 32766
# in newer ones; 500 stays well under both without forcing per-call
# tuning. Splitting into chunks at this boundary keeps the helpers safe
# for admin ``?all=true`` views that may produce thousands of artifact
# ids.
_MAX_IN_PARAMS = 500


def _in_placeholders(n: int) -> str:
    return ",".join("?" * n)


def _chunks(seq: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class ArtifactAccessDeniedError(LookupError):
    """Raised when a caller asks for an artifact owned by someone else.

    Routers translate this to ``HTTPException(status_code=404)`` rather than
    403 so the response doesn't disclose that the artifact exists.
    """


def resolve_artifact_owner(
    conn: sqlite3.Connection, *, experiment_id: str
) -> int | None:
    """Return the user_id that launched ``experiment_id``, or ``None`` if unknown.

    ``None`` covers two real cases: CLI-launched artifacts (no webapp job
    row) and artifacts that predate ownership tracking. Either way the
    artifact has no webapp owner and is treated as shared.
    """
    row = conn.execute(
        "SELECT user_id FROM jobs WHERE experiment_id = ? LIMIT 1",
        (experiment_id,),
    ).fetchone()
    return int(row["user_id"]) if row is not None else None


def check_artifact_access(
    conn: sqlite3.Connection, *, experiment_id: str, user: UserPublic
) -> None:
    """Raise ``ArtifactAccessDeniedError`` if ``user`` cannot read this artifact.

    Returns ``None`` on success. Admins always pass. Owners pass for
    artifacts they launched. Everyone passes on ownerless artifacts.
    """
    if user.role is Role.ADMIN:
        return
    owner_id = resolve_artifact_owner(conn, experiment_id=experiment_id)
    if owner_id is None or owner_id == user.id:
        return
    raise ArtifactAccessDeniedError(experiment_id)


def filter_visible_experiment_ids(
    conn: sqlite3.Connection,
    *,
    experiment_ids: list[str],
    user: UserPublic,
    all_users: bool,
) -> set[str]:
    """Return the subset of ``experiment_ids`` ``user`` may see.

    Drives list endpoints: scope to owned + ownerless by default; admins
    with ``all_users=True`` see everything.
    """
    if not experiment_ids:
        return set()
    if user.role is Role.ADMIN and all_users:
        return set(experiment_ids)
    owner_by_id: dict[str, int] = {}
    for chunk in _chunks(experiment_ids, _MAX_IN_PARAMS):
        rows = conn.execute(
            f"SELECT experiment_id, user_id FROM jobs "  # noqa: S608 - placeholders only
            f"WHERE experiment_id IN ({_in_placeholders(len(chunk))})",
            chunk,
        ).fetchall()
        owner_by_id.update(
            {str(row["experiment_id"]): int(row["user_id"]) for row in rows}
        )
    visible: set[str] = set()
    for eid in experiment_ids:
        owner = owner_by_id.get(eid)
        if owner is None or owner == user.id:
            visible.add(eid)
    return visible


def resolve_owner_usernames(
    conn: sqlite3.Connection, *, experiment_ids: list[str]
) -> dict[str, str]:
    """Look up ``launched_by_username`` for each experiment id with a webapp job.

    Returns ``{experiment_id: username}`` — keys are present only for
    artifacts with a matching jobs row. Caller treats missing keys as
    "ownerless / unknown" and renders accordingly.

    No access-scope filtering is applied here: this is an informational
    lookup. Callers must first restrict ``experiment_ids`` to the visible
    set via :func:`filter_visible_experiment_ids` before resolving names.
    """
    if not experiment_ids:
        return {}
    result: dict[str, str] = {}
    for chunk in _chunks(experiment_ids, _MAX_IN_PARAMS):
        rows = conn.execute(
            f"SELECT j.experiment_id AS experiment_id, u.username AS username "  # noqa: S608 - placeholders only
            f"FROM jobs j JOIN users u ON u.id = j.user_id "
            f"WHERE j.experiment_id IN ({_in_placeholders(len(chunk))})",
            chunk,
        ).fetchall()
        result.update(
            {str(row["experiment_id"]): str(row["username"]) for row in rows}
        )
    return result


__all__ = [
    "ArtifactAccessDeniedError",
    "check_artifact_access",
    "filter_visible_experiment_ids",
    "resolve_artifact_owner",
    "resolve_owner_usernames",
]
