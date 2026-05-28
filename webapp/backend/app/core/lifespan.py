"""
FastAPI lifespan: bootstrap the SQLite schema and wire the job substrate.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.infrastructure.job_store import (
    IllegalStatusTransitionError,
    mark_terminal,
)
from webapp.backend.app.infrastructure.process_manager import (
    JobEventBroker,
    ProcessManager,
)
from webapp.backend.app.schemas.jobs import JobStatus
from webapp.backend.app.services.auth_service import has_any_active_user
from webapp.backend.app.services.job_service import reconcile_orphans

logger = logging.getLogger(__name__)


async def _persist_terminal_status(
    job_id: str,
    status: JobStatus,
    exit_code: int | None,
    experiment_id: str | None,
) -> None:
    # Module-level (not nested in lifespan) so test reloads don't accumulate
    # closures over old `app` instances; opens its own connection.
    with open_db() as conn:
        try:
            mark_terminal(
                conn,
                job_id,
                status=status,
                exit_code=exit_code,
                experiment_id=experiment_id,
            )
        except IllegalStatusTransitionError:
            logger.warning("on_complete: job %s already terminal, skipping update", job_id)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    with open_db() as conn:
        bootstrap_schema(conn)
        if not has_any_active_user(conn):
            logger.warning(
                "No users found in %s. Run `python -m scripts.create_user "
                "<username> --role admin` to bootstrap an admin account.",
                settings.db_path,
            )
        reconcile_orphans(conn)

    broker = JobEventBroker()
    manager = ProcessManager(broker, on_complete=_persist_terminal_status)
    app.state.job_broker = broker
    app.state.job_manager = manager

    # Warm component registries once at startup so the torch/statsmodels/arch/xgboost
    # cost doesn't land on the first request that hits a registry-backed endpoint.
    import src.data  # noqa: F401
    import src.features  # noqa: F401
    import src.models  # noqa: F401
    import src.strategies  # noqa: F401

    try:
        yield
    finally:
        await manager.shutdown()
