"""Unit tests for the artifact-ownership helpers.

Covers the three resolution states (matching jobs row → owner;
no jobs row → ownerless = shared; admin override) and the helpers'
list-mode behaviour.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.job_store import NewJob, insert_job
from webapp.backend.app.schemas.jobs import JobKind, JobRow
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.ownership import (
    _MAX_IN_PARAMS,
    ArtifactAccessDeniedError,
    check_artifact_access,
    filter_visible_experiment_ids,
    resolve_artifact_owner,
    resolve_owner_usernames,
)
from webapp.backend.app.services.user_service import create_user

_USER_PASSWORD = "alice-password"
_ALICE_EID = "exp-alice"
_BOB_EID = "exp-bob"
_LEGACY_EID = "exp-legacy"


def _user(db_conn: sqlite3.Connection, username: str, role: Role = Role.USER) -> UserPublic:
    return create_user(db_conn, username=username, password=_USER_PASSWORD, role=role)


def _seed_job(
    db_conn: sqlite3.Connection,
    *,
    user: UserPublic,
    experiment_id: str,
) -> JobRow:
    """Insert a queued job row with the given experiment_id link."""

    job = insert_job(
        db_conn,
        NewJob(
            user_id=user.id,
            kind=JobKind.RUN,
            command=("experiment", "run"),
            config_path=Path(f"/tmp/{experiment_id}.yaml"),
            log_path=Path(f"/tmp/{experiment_id}.log"),
        ),
    )
    db_conn.execute(
        "UPDATE jobs SET experiment_id = ? WHERE id = ?", (experiment_id, job.id)
    )
    db_conn.commit()
    return job


def test_resolve_artifact_owner_returns_user_id_when_match(
    db_conn: sqlite3.Connection,
) -> None:
    alice = _user(db_conn, "alice")
    _seed_job(db_conn, user=alice, experiment_id=_ALICE_EID)
    assert resolve_artifact_owner(db_conn, experiment_id=_ALICE_EID) == alice.id


def test_resolve_artifact_owner_returns_none_when_no_job(
    db_conn: sqlite3.Connection,
) -> None:
    assert resolve_artifact_owner(db_conn, experiment_id=_LEGACY_EID) is None


def test_check_access_owner_passes(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    _seed_job(db_conn, user=alice, experiment_id=_ALICE_EID)
    check_artifact_access(db_conn, experiment_id=_ALICE_EID, user=alice)


def test_check_access_non_owner_raises(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    _seed_job(db_conn, user=alice, experiment_id=_ALICE_EID)
    with pytest.raises(ArtifactAccessDeniedError):
        check_artifact_access(db_conn, experiment_id=_ALICE_EID, user=bob)


def test_check_access_admin_bypasses(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    admin = _user(db_conn, "admin", role=Role.ADMIN)
    _seed_job(db_conn, user=alice, experiment_id=_ALICE_EID)
    check_artifact_access(db_conn, experiment_id=_ALICE_EID, user=admin)


def test_check_access_ownerless_is_shared(db_conn: sqlite3.Connection) -> None:
    bob = _user(db_conn, "bob")
    check_artifact_access(db_conn, experiment_id=_LEGACY_EID, user=bob)


def test_filter_visible_scopes_to_caller(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    _seed_job(db_conn, user=alice, experiment_id=_ALICE_EID)
    _seed_job(db_conn, user=bob, experiment_id=_BOB_EID)
    visible = filter_visible_experiment_ids(
        db_conn,
        experiment_ids=[_ALICE_EID, _BOB_EID, _LEGACY_EID],
        user=alice,
        all_users=False,
    )
    assert visible == {_ALICE_EID, _LEGACY_EID}


def test_filter_visible_admin_with_all_returns_everything(
    db_conn: sqlite3.Connection,
) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    admin = _user(db_conn, "admin", role=Role.ADMIN)
    _seed_job(db_conn, user=alice, experiment_id=_ALICE_EID)
    _seed_job(db_conn, user=bob, experiment_id=_BOB_EID)
    visible = filter_visible_experiment_ids(
        db_conn,
        experiment_ids=[_ALICE_EID, _BOB_EID, _LEGACY_EID],
        user=admin,
        all_users=True,
    )
    assert visible == {_ALICE_EID, _BOB_EID, _LEGACY_EID}


def test_filter_visible_empty_input_short_circuits(
    db_conn: sqlite3.Connection,
) -> None:
    alice = _user(db_conn, "alice")
    assert filter_visible_experiment_ids(
        db_conn, experiment_ids=[], user=alice, all_users=False
    ) == set()


def test_resolve_usernames_maps_matched_eids_only(
    db_conn: sqlite3.Connection,
) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    _seed_job(db_conn, user=alice, experiment_id=_ALICE_EID)
    _seed_job(db_conn, user=bob, experiment_id=_BOB_EID)
    usernames = resolve_owner_usernames(
        db_conn, experiment_ids=[_ALICE_EID, _BOB_EID, _LEGACY_EID]
    )
    assert usernames == {_ALICE_EID: "alice", _BOB_EID: "bob"}


def _seed_n_jobs(
    db_conn: sqlite3.Connection, *, user: UserPublic, count: int
) -> list[str]:
    eids = [f"chunk-eid-{i:04d}" for i in range(count)]
    for eid in eids:
        _seed_job(db_conn, user=user, experiment_id=eid)
    return eids


def test_filter_visible_chunks_above_in_clause_limit(
    db_conn: sqlite3.Connection,
) -> None:
    alice = _user(db_conn, "alice")
    eids = _seed_n_jobs(db_conn, user=alice, count=_MAX_IN_PARAMS + 50)
    visible = filter_visible_experiment_ids(
        db_conn, experiment_ids=eids, user=alice, all_users=False
    )
    assert visible == set(eids)


def test_resolve_usernames_chunks_above_in_clause_limit(
    db_conn: sqlite3.Connection,
) -> None:
    alice = _user(db_conn, "alice")
    eids = _seed_n_jobs(db_conn, user=alice, count=_MAX_IN_PARAMS + 50)
    usernames = resolve_owner_usernames(db_conn, experiment_ids=eids)
    assert usernames == dict.fromkeys(eids, "alice")
