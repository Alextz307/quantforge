"""Unit tests for ``scripts.backfill_artifact_owners``.

Covers the happy path against the canonical synthetic store, idempotency
on re-run, dry-run no-op semantics, user-not-found error, and the
empty-store-root short circuit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click
import pytest

from scripts.backfill_artifact_owners import backfill
from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.jobs import JobKind, JobStatus
from webapp.backend.app.services.user_service import create_user


_ADMIN_USERNAME = "alex"
_ADMIN_PASSWORD = "alex-password"


def _seed_admin(conn: sqlite3.Connection) -> int:
    user = create_user(
        conn, username=_ADMIN_USERNAME, password=_ADMIN_PASSWORD, role=Role.ADMIN
    )
    return int(user.id)


def test_backfill_attributes_every_artifact(
    db_conn: sqlite3.Connection, webapp_store: Path
) -> None:
    """A fresh DB + the canonical synthetic store → every artifact gets one row."""
    admin_id = _seed_admin(db_conn)
    plans = backfill(
        db_conn, username=_ADMIN_USERNAME, store_root=webapp_store, dry_run=False
    )
    # webapp_store creates: 2 runs, 1 comparison, 1 holdout, 1 study,
    # and 1 nested HPO study (which is skipped — nested HPO inherits owner).
    assert {p.kind_label for p in plans} == {
        "run",
        "comparison",
        "holdout",
        "study",
    }
    inserted = db_conn.execute(
        "SELECT kind, status, user_id, experiment_id FROM jobs"
    ).fetchall()
    assert all(row["user_id"] == admin_id for row in inserted)
    assert all(row["status"] == JobStatus.COMPLETED.value for row in inserted)
    inserted_kinds = {str(row["kind"]) for row in inserted}
    assert inserted_kinds == {
        JobKind.RUN.value,
        JobKind.COMPARE.value,
        JobKind.HOLDOUT.value,
        JobKind.STUDY.value,
    }


def test_backfill_is_idempotent(
    db_conn: sqlite3.Connection, webapp_store: Path
) -> None:
    _seed_admin(db_conn)
    first = backfill(
        db_conn, username=_ADMIN_USERNAME, store_root=webapp_store, dry_run=False
    )
    assert first  # sanity: first pass actually did work
    first_count = db_conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]

    second = backfill(
        db_conn, username=_ADMIN_USERNAME, store_root=webapp_store, dry_run=False
    )
    assert second == []
    second_count = db_conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    assert second_count == first_count


def test_dry_run_writes_nothing(
    db_conn: sqlite3.Connection, webapp_store: Path
) -> None:
    _seed_admin(db_conn)
    plans = backfill(
        db_conn, username=_ADMIN_USERNAME, store_root=webapp_store, dry_run=True
    )
    assert plans  # sanity: there's something it would have written
    count = db_conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    assert count == 0


def test_unknown_user_raises(
    db_conn: sqlite3.Connection, webapp_store: Path
) -> None:
    with pytest.raises(click.ClickException, match="not found"):
        backfill(
            db_conn, username="ghost", store_root=webapp_store, dry_run=False
        )


def test_empty_store_root_is_no_op(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    _seed_admin(db_conn)
    empty_root = tmp_path / "empty_store"
    empty_root.mkdir()
    plans = backfill(
        db_conn, username=_ADMIN_USERNAME, store_root=empty_root, dry_run=False
    )
    assert plans == []
    count = db_conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    assert count == 0


def test_existing_jobs_row_is_preserved(
    db_conn: sqlite3.Connection, webapp_store: Path
) -> None:
    """An artifact that already has an owner is not overwritten."""
    admin_id = _seed_admin(db_conn)
    bob = create_user(
        db_conn, username="bob", password="bob-password", role=Role.USER
    )
    # Pre-attribute one specific run to bob (not alex)
    bob_eid = "20260201_090000_PairsTrading_def5678_cafebabe"
    db_conn.execute(
        "INSERT INTO jobs (id, user_id, kind, command, config_path, log_path, "
        "status, experiment_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "preexisting-job-id",
            bob.id,
            JobKind.RUN.value,
            "manual",
            "",
            "",
            JobStatus.COMPLETED.value,
            bob_eid,
        ),
    )
    db_conn.commit()

    backfill(
        db_conn, username=_ADMIN_USERNAME, store_root=webapp_store, dry_run=False
    )
    owner_row = db_conn.execute(
        "SELECT user_id FROM jobs WHERE experiment_id = ?", (bob_eid,)
    ).fetchone()
    assert int(owner_row["user_id"]) == bob.id
    # Other artifacts went to admin.
    admin_rows = db_conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE user_id = ?", (admin_id,)
    ).fetchone()
    assert int(admin_rows["n"]) > 0
