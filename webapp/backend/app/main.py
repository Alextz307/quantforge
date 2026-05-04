"""FastAPI application factory for the webapp backend."""

from __future__ import annotations

from fastapi import FastAPI

from webapp.backend.app.api import health
from webapp.backend.app.core.lifespan import lifespan
from webapp.backend.app.core.version import APP_TITLE, APP_VERSION


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        openapi_url="/openapi.json",
        docs_url="/docs",
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/api")
    return app
