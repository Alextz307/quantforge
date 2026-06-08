"""
FastAPI application factory for the webapp backend.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.status import HTTP_404_NOT_FOUND
from starlette.types import Scope

from webapp.backend.app.api import (
    auth,
    comparisons,
    configs,
    deployments,
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
from webapp.backend.app.core import error_handlers, rate_limit
from webapp.backend.app.core.lifespan import lifespan
from webapp.backend.app.core.security import SessionCookies
from webapp.backend.app.core.settings import WebappEnv, get_settings
from webapp.backend.app.core.version import APP_TITLE, APP_VERSION

SECONDS_PER_MINUTE = 60
DEV_FRONTEND_ORIGIN = "http://localhost:5173"
SPA_INDEX = "index.html"
API_PREFIX = "api"


class _SPAStaticFiles(StaticFiles):
    """
    Serve the built bundle, falling back to ``index.html`` for unmatched
    non-API paths so client-side routes (deep links, refreshes) resolve to the
    single-page-application shell instead of 404-ing. A genuinely missing asset
    and any unknown ``/api`` path keep their 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            is_api = path == API_PREFIX or path.startswith(f"{API_PREFIX}/")
            if exc.status_code == HTTP_404_NOT_FOUND and not is_api:
                return await super().get_response(SPA_INDEX, scope)
            raise


def _mount_frontend(app: FastAPI, dist_dir: Path) -> None:
    """
    Mount the built single-page application at ``/`` when a frontend bundle is
    present, so one ``uvicorn`` process exposes the API on ``/api`` and the
    application on ``/``. A no-op when the bundle is absent (development against
    the Vite dev server, or the backend test suite), keeping the API usable on
    its own. Mounted last so the ``/api`` routers and ``/openapi.json`` win.
    """

    if not dist_dir.is_dir():
        return
    app.mount("/", _SPAStaticFiles(directory=dist_dir, html=True), name="frontend")


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
    app.include_router(deployments.router, prefix="/api")
    _mount_frontend(app, settings.frontend_dist)
    return app
