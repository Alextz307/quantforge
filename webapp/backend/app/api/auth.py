"""Login / logout / me endpoints. Login is rate-limited per-IP via slowapi."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from webapp.backend.app.core.deps import get_current_user, get_db, get_sessions
from webapp.backend.app.core.rate_limit import LOGIN_RATE_LIMIT, login_limiter
from webapp.backend.app.core.security import SessionCookies
from webapp.backend.app.schemas.auth import LoginRequest
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.auth_service import authenticate

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=UserPublic)
@login_limiter.limit(LOGIN_RATE_LIMIT)
def login(
    request: Request,
    body: LoginRequest,
    response: Response,
    conn: sqlite3.Connection = Depends(get_db),
    sessions: SessionCookies = Depends(get_sessions),
) -> UserPublic:
    authenticated = authenticate(conn, body.username, body.password)
    if authenticated is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    sessions.apply(response, authenticated.id)
    return UserPublic(id=authenticated.id, username=authenticated.username, role=authenticated.role)


@router.post("/logout")
def logout(response: Response, sessions: SessionCookies = Depends(get_sessions)) -> dict[str, str]:
    sessions.clear(response)
    return {"status": "logged_out"}


@router.get("/me", response_model=UserPublic)
def me(user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return user
