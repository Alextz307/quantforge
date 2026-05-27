"""Create or update a webapp account in the SQLite users table.

Usage::

    python -m scripts.create_user <username> [--role admin|user]

Prompts for the password (hidden input, with confirmation), bcrypt-hashes it,
and inserts or overwrites the row keyed by ``username``. The DB path comes
from ``WEBAPP_DB_PATH`` (default: ``webapp/data/webapp.sqlite``).
"""

from __future__ import annotations

import sys

import click

from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.services.auth_service import MIN_PASSWORD_LENGTH
from webapp.backend.app.services.user_service import upsert_user


@click.command()
@click.argument("username")
@click.option("--role", type=click.Choice([r.value for r in Role]), default=Role.USER.value)
@click.password_option()
def main(username: str, role: str, password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        click.echo(f"Password must be at least {MIN_PASSWORD_LENGTH} characters", err=True)
        sys.exit(1)
    with open_db() as conn:
        bootstrap_schema(conn)
        user = upsert_user(conn, username=username, password=password, role=Role(role))
    click.echo(f"Saved user '{user.username}' ({user.role.value}) to {get_settings().db_path}")


if __name__ == "__main__":
    main()
