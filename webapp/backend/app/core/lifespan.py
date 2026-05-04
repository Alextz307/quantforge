"""FastAPI lifespan: bootstrap the SQLite schema and warn on empty users."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.services.auth_service import has_any_active_user

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    with open_db() as conn:
        bootstrap_schema(conn)
        if not has_any_active_user(conn):
            logger.warning(
                "No users found in %s. Run `python -m scripts.create_user "
                "<username> --role admin` to bootstrap an admin account.",
                get_settings().db_path,
            )
    yield
