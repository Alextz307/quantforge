"""WebSocket-side cookie auth shared by job + HPO stream endpoints."""

from __future__ import annotations

from fastapi import WebSocket

from webapp.backend.app.core.security import SESSION_COOKIE_NAME, SessionCookies
from webapp.backend.app.infrastructure.db import open_db
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.user_service import get_user

# Custom close codes use the application range (4000–4999). They mirror the
# corresponding HTTP status semantics so client error handling stays uniform
# whether a request reached the HTTP or the WS layer.
WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_FORBIDDEN = 4403
WS_CLOSE_NOT_FOUND = 4404


def resolve_ws_user(websocket: WebSocket) -> UserPublic | None:
    """WebSocket equivalent of ``get_optional_user`` (no Response for cookie refresh)."""

    sessions: SessionCookies = websocket.app.state.sessions
    token = websocket.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    user_id = sessions.decode(token)
    if user_id is None:
        return None
    with open_db() as conn:
        return get_user(conn, user_id)


__all__ = [
    "WS_CLOSE_FORBIDDEN",
    "WS_CLOSE_NOT_FOUND",
    "WS_CLOSE_UNAUTHORIZED",
    "resolve_ws_user",
]
