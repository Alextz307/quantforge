"""Unit tests for the universe-spec uploads service.

Mirrors the study-uploads test surface: YAML-text validation
(parse + schema), create/update/soft-delete round trips, tombstone-reuse
under the partial unique index, and library-slug rejection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.spec_upload_store import LibrarySlugCollisionError
from webapp.backend.app.services.universe_spec_uploads import (
    UniverseSpecUploadInvalidError,
    UniverseSpecUploadNotFoundError,
    find_upload_path,
    get_upload,
    list_uploads,
    save_upload,
    soft_delete_upload,
    validate_universe_spec_text,
)
from webapp.backend.app.services.user_service import create_user

_USER_PASSWORD = "alice-password"

VALID_UNIVERSE_YAML = """\
data:
  source: yfinance
  tickers: [SPY]
  start: 2020-01-01
  end: 2024-12-31
  interval: daily
validation:
  holdout_pct: 0.20
"""


def _user(db_conn: sqlite3.Connection, username: str) -> UserPublic:
    return create_user(db_conn, username=username, password=_USER_PASSWORD, role=Role.USER)


def test_validate_happy_path() -> None:
    result = validate_universe_spec_text(VALID_UNIVERSE_YAML)
    assert result.valid is True
    assert result.errors == []


def test_validate_yaml_parse_error() -> None:
    result = validate_universe_spec_text("data: [unterminated")
    assert result.valid is False
    assert result.errors[0].loc == ["yaml"]
    assert "parse error" in result.errors[0].msg.lower()


def test_validate_empty_body() -> None:
    result = validate_universe_spec_text("\n")
    assert result.valid is False
    assert result.errors[0].loc == ["yaml"]
    assert "empty" in result.errors[0].msg.lower()


def test_validate_non_mapping_top_level() -> None:
    result = validate_universe_spec_text("- a\n- b\n")
    assert result.valid is False
    assert result.errors[0].loc == ["yaml"]
    assert "mapping" in result.errors[0].msg.lower()


def test_validate_missing_required_field() -> None:
    result = validate_universe_spec_text("validation: {}\n")
    assert result.valid is False
    assert any("data" in err.loc for err in result.errors)


def test_save_persists_row_and_file(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    alice = _user(db_conn, "alice")
    detail = save_upload(
        db_conn,
        user=alice,
        slug="my_universe",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    assert detail.slug == "my_universe"
    assert detail.owner_user_id == alice.id
    assert detail.owner_username == "alice"
    on_disk = find_upload_path(tmp_path, alice.id, "my_universe")
    assert on_disk is not None and on_disk.read_text() == VALID_UNIVERSE_YAML


def test_save_invalid_yaml_raises(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    alice = _user(db_conn, "alice")
    with pytest.raises(UniverseSpecUploadInvalidError):
        save_upload(
            db_conn,
            user=alice,
            slug="bad",
            yaml_text="validation: {}\n",
            uploads_root=tmp_path,
            config_root=tmp_path / "config",
        )


def test_save_library_slug_collision(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    alice = _user(db_conn, "alice")
    (tmp_path / "config" / "universes").mkdir(parents=True)
    (tmp_path / "config" / "universes" / "main.yaml").write_text("body\n")
    with pytest.raises(LibrarySlugCollisionError):
        save_upload(
            db_conn,
            user=alice,
            slug="main",
            yaml_text=VALID_UNIVERSE_YAML,
            uploads_root=tmp_path,
            config_root=tmp_path / "config",
        )


def test_list_uploads_scopes_to_caller(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    save_upload(
        db_conn,
        user=alice,
        slug="alice_universe",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    save_upload(
        db_conn,
        user=bob,
        slug="bob_universe",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    alice_view = list_uploads(db_conn, user=alice, all_users=False)
    assert [u.slug for u in alice_view] == ["alice_universe"]


def test_admin_all_users_lists_everything(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    admin = create_user(
        db_conn, username="admin", password=_USER_PASSWORD, role=Role.ADMIN
    )
    save_upload(
        db_conn,
        user=alice,
        slug="alice_universe",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    save_upload(
        db_conn,
        user=bob,
        slug="bob_universe",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    full = list_uploads(db_conn, user=admin, all_users=True)
    assert {u.slug for u in full} == {"alice_universe", "bob_universe"}


def test_non_admin_all_users_raises(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    with pytest.raises(PermissionError):
        list_uploads(db_conn, user=alice, all_users=True)


def test_get_upload_returns_own(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    alice = _user(db_conn, "alice")
    save_upload(
        db_conn,
        user=alice,
        slug="u",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    detail = get_upload(db_conn, user=alice, slug="u")
    assert detail.slug == "u"
    assert detail.yaml == VALID_UNIVERSE_YAML


def test_get_upload_other_users_returns_not_found(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    save_upload(
        db_conn,
        user=alice,
        slug="u",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    with pytest.raises(UniverseSpecUploadNotFoundError):
        get_upload(db_conn, user=bob, slug="u")


def test_soft_delete_removes_file_and_row(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    alice = _user(db_conn, "alice")
    save_upload(
        db_conn,
        user=alice,
        slug="u",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    assert find_upload_path(tmp_path, alice.id, "u") is not None
    soft_delete_upload(db_conn, user=alice, slug="u", uploads_root=tmp_path)
    assert find_upload_path(tmp_path, alice.id, "u") is None
    assert list_uploads(db_conn, user=alice) == []


def test_resave_after_soft_delete_reactivates_tombstone(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Re-saving a previously soft-deleted slug must reactivate the
    tombstoned row rather than collide on the unique index.
    """

    alice = _user(db_conn, "alice")
    save_upload(
        db_conn,
        user=alice,
        slug="u",
        yaml_text=VALID_UNIVERSE_YAML,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    soft_delete_upload(db_conn, user=alice, slug="u", uploads_root=tmp_path)
    refreshed_yaml = VALID_UNIVERSE_YAML.replace("SPY", "QQQ")
    detail = save_upload(
        db_conn,
        user=alice,
        slug="u",
        yaml_text=refreshed_yaml,
        uploads_root=tmp_path,
        config_root=tmp_path / "config",
    )
    assert detail.yaml == refreshed_yaml
    assert [u.slug for u in list_uploads(db_conn, user=alice)] == ["u"]
