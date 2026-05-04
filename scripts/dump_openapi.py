"""Dump the FastAPI OpenAPI spec to a JSON snapshot for frontend type generation.

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

import click

DEFAULT_SNAPSHOT_PATH = Path("webapp/frontend/openapi.snapshot.json")
DUMMY_SECRET = "x" * 64

# WebappSettings rejects secret_key shorter than 32 chars at construction time;
# this script only needs the OpenAPI shape, so seed a dummy before the import below.
os.environ.setdefault("WEBAPP_SECRET_KEY", DUMMY_SECRET)

from src.core import json_io  # noqa: E402
from webapp.backend.app.main import create_app  # noqa: E402


@click.command()
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SNAPSHOT_PATH,
)
def main(out: Path) -> None:
    spec = create_app().openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    json_io.write(out, spec)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
