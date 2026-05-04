"""FastAPI application factory for the webapp backend."""

from __future__ import annotations

from fastapi import FastAPI

from webapp.backend.app.api import (
    auth,
    comparisons,
    health,
    holdout,
    models,
    regime,
    runs,
    strategies,
    users,
)
from webapp.backend.app.core import rate_limit
from webapp.backend.app.core.lifespan import lifespan
from webapp.backend.app.core.security import SessionCookies
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.core.version import APP_TITLE, APP_VERSION

SECONDS_PER_MINUTE = 60


def create_app() -> FastAPI:
    settings = get_settings()
    sessions = SessionCookies(
        secret_key=settings.secret_key,
        max_age_seconds=settings.session_ttl_minutes * SECONDS_PER_MINUTE,
    )
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        openapi_url="/openapi.json",
        docs_url="/docs",
        lifespan=lifespan,
    )
    app.state.sessions = sessions
    rate_limit.attach(app)

    app.include_router(health.router, prefix="/api")
    app.include_router(auth.router, prefix="/api")
    app.include_router(users.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(comparisons.router, prefix="/api")
    app.include_router(regime.router, prefix="/api")
    app.include_router(holdout.router, prefix="/api")
    app.include_router(strategies.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    return app
