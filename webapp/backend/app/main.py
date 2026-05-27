"""FastAPI application factory for the webapp backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp.backend.app.api import (
    auth,
    comparisons,
    configs,
    health,
    holdout,
    hpo,
    jobs,
    models,
    runs,
    strategies,
    studies,
    users,
)
from webapp.backend.app.core import rate_limit, error_handlers
from webapp.backend.app.core.lifespan import lifespan
from webapp.backend.app.core.security import SessionCookies
from webapp.backend.app.core.settings import WebappEnv, get_settings
from webapp.backend.app.core.version import APP_TITLE, APP_VERSION

SECONDS_PER_MINUTE = 60
DEV_FRONTEND_ORIGIN = "http://localhost:5173"


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
    error_handlers.attach(app)

    if settings.env is WebappEnv.DEVELOPMENT:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[DEV_FRONTEND_ORIGIN],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router, prefix="/api")
    app.include_router(auth.router, prefix="/api")
    app.include_router(users.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(comparisons.router, prefix="/api")
    app.include_router(holdout.router, prefix="/api")
    app.include_router(studies.router, prefix="/api")
    app.include_router(hpo.router, prefix="/api")
    app.include_router(strategies.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(configs.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    return app
