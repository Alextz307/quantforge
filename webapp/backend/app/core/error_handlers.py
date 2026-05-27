"""App-level exception handlers — translate service-layer errors to HTTP responses.

Registers FastAPI handlers that map the small set of cross-cutting exception
classes raised by service modules into the HTTP responses the frontend
expects. Endpoints stay free of repetitive try/except blocks for these
errors; the mapping lives here, once.

Status code mapping:

* :class:`SpecUploadNotFoundError`     -> ``404`` (uses per-kind
  ``kind_label`` class attribute for the detail message).
* :class:`LibrarySlugCollisionError`   -> ``409`` with the resolved
  library path in the detail.
* :class:`SpecUploadInvalidError`      -> ``422`` with the verbatim
  ``ValidationErrorItem`` list the editor renders inline.
* :class:`ArtifactAccessDeniedError`   -> ``404`` (not 403, so the
  response does not disclose that the artifact exists at all).
* :class:`PermissionError` raised from a list call (non-admin
  ``?all=true``)                       -> ``403``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from webapp.backend.app.services.ownership import ArtifactAccessDeniedError
from webapp.backend.app.services.spec_upload_store import (
    LibrarySlugCollisionError,
    SpecUploadInvalidError,
    SpecUploadNotFoundError,
)

ExcHandler = Callable[[Request, Exception], Awaitable[Response] | Response]


async def _handle_not_found(_req: Request, exc: Exception) -> Response:
    err = cast(SpecUploadNotFoundError, exc)
    slug = err.args[0] if err.args else ""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"{err.kind_label} spec upload not found: {slug}"},
    )


async def _handle_library_collision(_req: Request, exc: Exception) -> Response:
    err = cast(LibrarySlugCollisionError, exc)
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "detail": (
                f"slug '{err.slug}' shadows a library spec at "
                f"{err.library_path} — pick a different slug"
            )
        },
    )


async def _handle_invalid(_req: Request, exc: Exception) -> Response:
    err = cast(SpecUploadInvalidError, exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": [item.model_dump() for item in err.errors]},
    )


async def _handle_access_denied(_req: Request, exc: Exception) -> Response:
    """Map ``ArtifactAccessDeniedError`` to 404 (not 403).

    Surfaced from artifact read endpoints when the caller is neither the
    artifact's owner nor an admin. 404 (not 403) so the response does not
    disclose that the artifact exists — matching the framework-wide
    no-peek-forward policy on identity leaks.
    """
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)}
    )


async def _handle_list_permission(_req: Request, exc: Exception) -> Response:
    """Map non-admin ``?all=true`` PermissionError to 403.

    Scoped narrowly: only PermissionErrors raised from spec-upload list
    paths take this route. Other domain modules that may raise
    PermissionError in the future need their own handlers — this one is
    registered specifically for the upload list endpoints' error shape.
    """
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)}
    )


def attach(app: FastAPI) -> None:
    app.add_exception_handler(
        SpecUploadNotFoundError, cast(ExcHandler, _handle_not_found)
    )
    app.add_exception_handler(
        LibrarySlugCollisionError, cast(ExcHandler, _handle_library_collision)
    )
    app.add_exception_handler(
        SpecUploadInvalidError, cast(ExcHandler, _handle_invalid)
    )
    app.add_exception_handler(
        ArtifactAccessDeniedError, cast(ExcHandler, _handle_access_denied)
    )
    app.add_exception_handler(
        PermissionError, cast(ExcHandler, _handle_list_permission)
    )
