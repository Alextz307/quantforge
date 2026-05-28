"""
Drift guard for the Pydantic ↔ zod form-validation mirror.

The frontend mirrors a small set of Pydantic write-DTOs as zod schemas under
``webapp/frontend/src/lib/schemas/`` so forms validate before hitting the
backend. The two sources are hand-kept-in-sync. This script extracts each
mirrored Pydantic model's field shape (name, type, ``min``/``max`` constraints)
and writes a canonical JSON snapshot at
``webapp/frontend/schema-mirror.snapshot.json``. CI runs ``--check`` to fail on
drift; a vitest test on the frontend asserts the zod schema agrees with the
same snapshot.

Add a new mirrored model by appending to ``MIRRORED_MODELS`` here AND its zod
counterpart AND wiring the vitest assertion. Keep field names in lockstep.

Usage::

    python -m scripts.check_webapp_schema_mirror              # exit 1 on drift
    python -m scripts.check_webapp_schema_mirror --write      # regenerate snapshot
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import click

from src.core import json_io

DEFAULT_SNAPSHOT_PATH = Path("webapp/frontend/schema-mirror.snapshot.json")
DUMMY_SECRET = "x" * 64

# Webapp settings refuse a short secret; seed a dummy before any webapp import.
os.environ.setdefault("WEBAPP_SECRET_KEY", DUMMY_SECRET)

_FIX_COMMAND = "python -m scripts.check_webapp_schema_mirror --write"


def _build_mirror_shape() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Extract the canonical field shape for every mirrored Pydantic model.

    Lazy webapp import keeps this script importable in environments without
    fastapi installed (e.g. the unit-test path that exercises the diff function
    directly with a fixture model).
    """

    from webapp.backend.app.schemas.auth import LoginRequest
    from webapp.backend.app.schemas.users import UserCreate

    mirrored: dict[str, type] = {"login": LoginRequest, "userCreate": UserCreate}
    return {name: extract_field_shape(model) for name, model in mirrored.items()}


def extract_field_shape(model: type) -> dict[str, dict[str, Any]]:
    """
    Return ``{field_name: {type, min, max, default, enum?}}`` for a Pydantic model.

    Constraints are introspected from the field's ``metadata`` (pydantic v2
    moves ``min_length``/``max_length`` etc into annotated metadata).
    """

    fields_attr = getattr(model, "model_fields", None)
    if fields_attr is None:
        raise TypeError(f"{model!r} is not a pydantic v2 model")
    shape: dict[str, dict[str, Any]] = {}
    for field_name, info in fields_attr.items():
        shape[field_name] = _field_to_shape(info)
    return shape


def _field_to_shape(info: object) -> dict[str, Any]:
    """
    Convert a single pydantic v2 ``FieldInfo`` into a canonical dict.
    """

    annotation = getattr(info, "annotation", None)
    out: dict[str, Any] = {"type": _annotation_to_type(annotation)}
    metadata = getattr(info, "metadata", []) or []
    for entry in metadata:
        for attr, key in (("min_length", "min"), ("max_length", "max")):
            value = getattr(entry, attr, None)
            if value is not None:
                out[key] = value
    if (default := getattr(info, "default", None)) is not None and default is not Ellipsis:
        # Don't emit pydantic's PydanticUndefined sentinel.
        if default.__class__.__name__ != "PydanticUndefinedType":
            out["default"] = _serialize_default(default)
    return out


def _annotation_to_type(annotation: Any) -> str:
    """
    Render a python annotation as a stable string the frontend can match.
    """

    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if isinstance(annotation, type) and issubclass(annotation, str):
        members = getattr(annotation, "__members__", None)
        if members is not None:
            return "enum:" + "|".join(sorted(m.value for m in members.values()))
        return "string"
    return repr(annotation)


def _serialize_default(default: object) -> str | int | float | bool:
    """
    Render a default value as JSON-friendly data.

    Currently only StrEnum-like defaults appear in the mirrored set; extend
    when a new default type lands rather than silently falling back.
    """

    value: object = getattr(default, "value", None)
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported default value: {default!r} ({type(default).__name__})")


@click.command()
@click.option(
    "--snapshot",
    "snapshot_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SNAPSHOT_PATH,
)
@click.option("--write", is_flag=True, help="Regenerate the snapshot instead of checking it.")
def main(snapshot_path: Path, write: bool) -> None:
    shape = _build_mirror_shape()
    if write:
        json_io.write(snapshot_path, shape)
        click.echo(f"Wrote {snapshot_path}")
        return
    errors = json_io.diff_against_snapshot(
        shape,
        snapshot_path,
        label="Schema-mirror snapshot",
        fix_command=_FIX_COMMAND,
    )
    if errors:
        for line in errors:
            click.echo(line, err=True)
        sys.exit(1)
    click.echo(f"OK: schema-mirror snapshot at {snapshot_path} is current")


if __name__ == "__main__":
    main()
