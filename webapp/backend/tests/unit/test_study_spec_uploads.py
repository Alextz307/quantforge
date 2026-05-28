"""
Unit tests for the study-spec uploads service.

Covers: YAML-text validation (parse, schema, referenced-file existence),
create / update / soft-delete round trips, the tombstone-collision
[[soft-delete-schema-pattern]] regression, and library-slug rejection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.study_spec_uploads import (
    LibrarySlugCollisionError,
    StudySpecUploadInvalidError,
    StudySpecUploadNotFoundError,
    find_upload_path,
    get_upload,
    list_uploads,
    save_upload,
    soft_delete_upload,
    validate_study_spec_text,
)
from webapp.backend.app.services.user_service import create_user

USER_PASSWORD = "alice-password"


def _user(db_conn: sqlite3.Connection, username: str) -> UserPublic:
    return create_user(db_conn, username=username, password=USER_PASSWORD, role=Role.USER)


def _seed_library_files(config_root: Path) -> None:
    """
    Plant the strategy/hpo/universe files a valid 1-leg spec references.
    """

    (config_root / "strategies").mkdir(parents=True)
    (config_root / "hpo").mkdir()
    (config_root / "universes").mkdir()
    (config_root / "strategies" / "adaptive_bollinger.yaml").write_text("body\n")
    (config_root / "hpo" / "adaptive_bollinger.yaml").write_text("body\n")
    (config_root / "universes" / "spy_daily_5y.yaml").write_text("body\n")


VALID_YAML = """\
name: my_study
description: Toy 1-leg study.
seed: 42
output_dir: studies/my_study
legs:
  - strategy: AdaptiveBollinger
    strategy_config: STRAT_PATH
    hpo_config: HPO_PATH
    universes:
      - spy_daily_5y
"""


def _valid_yaml(config_root: Path) -> str:
    """
    Render the canonical 1-leg spec with absolute leg paths under ``config_root``.
    """

    return VALID_YAML.replace(
        "STRAT_PATH", str(config_root / "strategies" / "adaptive_bollinger.yaml")
    ).replace("HPO_PATH", str(config_root / "hpo" / "adaptive_bollinger.yaml"))


def test_validate_happy_path(tmp_path: Path) -> None:
    _seed_library_files(tmp_path)
    result = validate_study_spec_text(_valid_yaml(tmp_path), config_root=tmp_path)
    assert result.valid is True
    assert result.errors == []


def test_validate_yaml_parse_error(tmp_path: Path) -> None:
    result = validate_study_spec_text("name: [unterminated", config_root=tmp_path)
    assert result.valid is False
    assert result.errors[0].loc == ["yaml"]
    assert "parse error" in result.errors[0].msg.lower()


def test_validate_empty_body(tmp_path: Path) -> None:
    result = validate_study_spec_text("\n", config_root=tmp_path)
    assert result.valid is False
    assert result.errors[0].loc == ["yaml"]
    assert result.errors[0].msg == "empty YAML"


def test_validate_non_mapping_body(tmp_path: Path) -> None:
    result = validate_study_spec_text("- 1\n- 2\n", config_root=tmp_path)
    assert result.valid is False
    assert result.errors[0].loc == ["yaml"]
    assert "mapping" in result.errors[0].msg


def test_validate_missing_required_field(tmp_path: Path) -> None:
    _seed_library_files(tmp_path)
    bad = _valid_yaml(tmp_path).replace("output_dir: studies/my_study\n", "")
    result = validate_study_spec_text(bad, config_root=tmp_path)
    assert result.valid is False
    assert any("output_dir" in err.loc for err in result.errors)


def test_validate_missing_strategy_config(tmp_path: Path) -> None:
    _seed_library_files(tmp_path)
    yaml_text = _valid_yaml(tmp_path).replace(
        "adaptive_bollinger.yaml", "missing.yaml", 1
    )
    result = validate_study_spec_text(yaml_text, config_root=tmp_path)
    assert result.valid is False
    assert any(
        err.loc == ["legs", "0", "strategy_config"] and "not found" in err.msg
        for err in result.errors
    )


def test_validate_unknown_universe(tmp_path: Path) -> None:
    _seed_library_files(tmp_path)
    yaml_text = _valid_yaml(tmp_path).replace("spy_daily_5y", "ghost_universe")
    result = validate_study_spec_text(yaml_text, config_root=tmp_path)
    assert result.valid is False
    assert any(
        err.loc == ["legs", "0", "universes", "0"] for err in result.errors
    )


def test_save_creates_row_and_yaml_file(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    _seed_library_files(tmp_path)
    user = _user(db_conn, "alice")
    uploads_root = tmp_path / "uploads"

    detail = save_upload(
        db_conn,
        user=user,
        slug="my_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=uploads_root,
        config_root=tmp_path,
    )

    assert detail.slug == "my_study"
    assert detail.owner_user_id == user.id
    on_disk = uploads_root / str(user.id) / "my_study.yaml"
    assert on_disk.is_file()
    assert on_disk.read_text(encoding="utf-8") == _valid_yaml(tmp_path)


def test_save_overwrites_existing_active_row(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    _seed_library_files(tmp_path)
    user = _user(db_conn, "alice")
    uploads_root = tmp_path / "uploads"

    save_upload(
        db_conn,
        user=user,
        slug="my_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=uploads_root,
        config_root=tmp_path,
    )
    edited = _valid_yaml(tmp_path).replace("Toy 1-leg study.", "Edited summary.")
    detail = save_upload(
        db_conn,
        user=user,
        slug="my_study",
        yaml_text=edited,
        uploads_root=uploads_root,
        config_root=tmp_path,
    )

    assert "Edited summary." in detail.yaml
    rows = db_conn.execute(
        "SELECT COUNT(*) AS n FROM study_spec_uploads WHERE user_id = ? AND slug = ?",
        (user.id, "my_study"),
    ).fetchone()
    assert rows["n"] == 1


def test_save_rejects_library_slug_collision(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    _seed_library_files(tmp_path)
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "main_study.yaml").write_text("body\n")
    user = _user(db_conn, "alice")

    with pytest.raises(LibrarySlugCollisionError):
        save_upload(
            db_conn,
            user=user,
            slug="main_study",
            yaml_text=_valid_yaml(tmp_path),
            uploads_root=tmp_path / "uploads",
            config_root=tmp_path,
        )


def test_save_rejects_invalid_yaml(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    user = _user(db_conn, "alice")
    with pytest.raises(StudySpecUploadInvalidError) as exc_info:
        save_upload(
            db_conn,
            user=user,
            slug="busted",
            yaml_text="name: [bad",
            uploads_root=tmp_path / "uploads",
            config_root=tmp_path,
        )
    assert exc_info.value.errors


def test_list_scopes_to_caller_by_default(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    _seed_library_files(tmp_path)
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    save_upload(
        db_conn,
        user=alice,
        slug="a_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=tmp_path / "uploads",
        config_root=tmp_path,
    )
    save_upload(
        db_conn,
        user=bob,
        slug="b_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=tmp_path / "uploads",
        config_root=tmp_path,
    )

    alice_view = list_uploads(db_conn, user=alice)
    assert {u.slug for u in alice_view} == {"a_study"}


def test_list_all_users_admin_only(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    _seed_library_files(tmp_path)
    alice = _user(db_conn, "alice")
    bob = create_user(
        db_conn, username="bob", password=USER_PASSWORD, role=Role.ADMIN
    )
    save_upload(
        db_conn,
        user=alice,
        slug="a_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=tmp_path / "uploads",
        config_root=tmp_path,
    )

    with pytest.raises(PermissionError):
        list_uploads(db_conn, user=alice, all_users=True)
    assert {u.slug for u in list_uploads(db_conn, user=bob, all_users=True)} == {"a_study"}


def test_get_missing_raises(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    alice = _user(db_conn, "alice")
    with pytest.raises(StudySpecUploadNotFoundError):
        get_upload(db_conn, user=alice, slug="ghost")


def test_soft_delete_removes_file_and_hides_row(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    _seed_library_files(tmp_path)
    user = _user(db_conn, "alice")
    uploads_root = tmp_path / "uploads"
    save_upload(
        db_conn,
        user=user,
        slug="my_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=uploads_root,
        config_root=tmp_path,
    )

    soft_delete_upload(db_conn, user=user, slug="my_study", uploads_root=uploads_root)

    assert list_uploads(db_conn, user=user) == []
    assert not (uploads_root / str(user.id) / "my_study.yaml").exists()
    with pytest.raises(StudySpecUploadNotFoundError):
        get_upload(db_conn, user=user, slug="my_study")


def test_save_after_soft_delete_reactivates_row(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """
    Tombstone-collision regression — see [[soft-delete-schema-pattern]].

    A previously-soft-deleted (user_id, slug) row must be re-savable. The
    partial unique index plus the UPDATE-on-existing branch in save_upload
    together make this work; this test pins the contract.
    """

    _seed_library_files(tmp_path)
    user = _user(db_conn, "alice")
    uploads_root = tmp_path / "uploads"
    save_upload(
        db_conn,
        user=user,
        slug="my_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=uploads_root,
        config_root=tmp_path,
    )
    soft_delete_upload(db_conn, user=user, slug="my_study", uploads_root=uploads_root)

    detail = save_upload(
        db_conn,
        user=user,
        slug="my_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=uploads_root,
        config_root=tmp_path,
    )

    assert detail.slug == "my_study"
    rows = db_conn.execute(
        "SELECT COUNT(*) AS n FROM study_spec_uploads "
        "WHERE user_id = ? AND slug = ? AND deleted_at IS NULL",
        (user.id, "my_study"),
    ).fetchone()
    assert rows["n"] == 1


def test_find_upload_path_returns_path_when_present(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    _seed_library_files(tmp_path)
    user = _user(db_conn, "alice")
    uploads_root = tmp_path / "uploads"
    save_upload(
        db_conn,
        user=user,
        slug="my_study",
        yaml_text=_valid_yaml(tmp_path),
        uploads_root=uploads_root,
        config_root=tmp_path,
    )

    found = find_upload_path(uploads_root, user.id, "my_study")
    assert found is not None
    assert found.is_file()


def test_find_upload_path_returns_none_for_missing(tmp_path: Path) -> None:
    assert find_upload_path(tmp_path / "uploads", user_id=42, slug="ghost") is None
