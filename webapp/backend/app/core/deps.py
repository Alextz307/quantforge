"""
FastAPI dependency providers for DB connections, sessions, and auth gates.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import cast

from fastapi import Depends, HTTPException, Request, status

from webapp.backend.app.core.security import SESSION_COOKIE_NAME, SessionCookies
from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.db import open_db
from webapp.backend.app.infrastructure.process_manager import ProcessManager
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.user_service import get_user


def get_db() -> Iterator[sqlite3.Connection]:
    with open_db() as conn:
        yield conn


def get_sessions(request: Request) -> SessionCookies:
    return cast(SessionCookies, request.app.state.sessions)


def get_optional_user(
    request: Request,
    sessions: SessionCookies = Depends(get_sessions),
    conn: sqlite3.Connection = Depends(get_db),
) -> UserPublic | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    user_id = sessions.decode(token)
    if user_id is None:
        return None
    return get_user(conn, user_id)


def get_current_user(user: UserPublic | None = Depends(get_optional_user)) -> UserPublic:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def require_admin(user: UserPublic = Depends(get_current_user)) -> UserPublic:
    if user.role is not Role.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


def get_job_manager(request: Request) -> ProcessManager:
    return cast(ProcessManager, request.app.state.job_manager)
