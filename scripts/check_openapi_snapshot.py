"""Drift guard for ``webapp/frontend/openapi.snapshot.json``.

The committed snapshot is the contract between the FastAPI backend and the
TypeScript frontend (``npm run gen:api`` reads it). When backend routes or
Pydantic DTOs change, the snapshot must be regenerated and committed in the
same change. This script re-builds the OpenAPI spec from the live FastAPI app
and fails CI when it diverges from the committed snapshot.

Usage::

    python -m scripts.check_openapi_snapshot

Pass ``--snapshot <path>`` to point at a non-default location (used by the
unit test to compare against a tmp_path fixture).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from scripts.dump_openapi import DEFAULT_SNAPSHOT_PATH, build_openapi_spec
from src.core import json_io


@click.command()
@click.option(
    "--snapshot",
    "snapshot_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SNAPSHOT_PATH,
)
def main(snapshot_path: Path) -> None:
    errors = json_io.diff_against_snapshot(
        build_openapi_spec(),
        snapshot_path,
        label="OpenAPI snapshot",
        fix_command="make webapp-openapi-snapshot",
    )
    if errors:
        for line in errors:
            click.echo(line, err=True)
        sys.exit(1)
    click.echo(f"OK: OpenAPI snapshot at {snapshot_path} is current")


if __name__ == "__main__":
    main()
