"""
Dump the FastAPI OpenAPI spec to a JSON snapshot for frontend type generation.

Usage::

    python -m scripts.dump_openapi [--out <path>]

The default output path is ``webapp/frontend/openapi.snapshot.json``. The
frontend's ``npm run gen:api`` step reads this snapshot and emits
``src/api/generated/schema.ts``; the snapshot is the committed contract
between the Python backend and the TypeScript frontend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import click

from src.core import json_io

DEFAULT_SNAPSHOT_PATH = Path("webapp/frontend/openapi.snapshot.json")
DUMMY_SECRET = "x" * 64


def build_openapi_spec() -> dict[str, Any]:
    """
    Return the FastAPI OpenAPI dict — single source for dump + drift check.

    Webapp imports are lazy so importers that only need ``DEFAULT_SNAPSHOT_PATH``
    (e.g. the drift-guard unit test) don't pay for fastapi at import time.
    """

    # WebappSettings rejects secret_key shorter than 32 chars at construction time;
    # this script only needs the OpenAPI shape, so seed a dummy before the import below.
    os.environ.setdefault("WEBAPP_SECRET_KEY", DUMMY_SECRET)
    from webapp.backend.app.main import create_app

    # cast: in the lint-and-typecheck job the [mypy-webapp.*] override treats
    # webapp.* as Any, so mypy can't see that openapi() returns a dict.
    return cast(dict[str, Any], create_app().openapi())


@click.command()
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SNAPSHOT_PATH,
)
def main(out: Path) -> None:
    json_io.write(out, build_openapi_spec())
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
