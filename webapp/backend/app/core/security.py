"""Session-cookie signing primitives backed by itsdangerous."""

from __future__ import annotations

from fastapi import Response
from itsdangerous import BadSignature, URLSafeTimedSerializer

SESSION_COOKIE_NAME = "webapp_session"
SESSION_SALT = "webapp-session-v1"
SESSION_COOKIE_PATH = "/"
MIN_SECRET_KEY_LENGTH = 32

# secure=False because the webapp is local-only HTTP; browsers (and httpx in TestClient)
# refuse Secure cookies over plain HTTP. Revisit when an HTTPS reverse proxy lands.
SESSION_COOKIE_SECURE = False


class SessionCookies:
    def __init__(self, secret_key: str, max_age_seconds: int) -> None:
        if len(secret_key) < MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"secret_key must be at least {MIN_SECRET_KEY_LENGTH} characters; "
                f"generate one with `python -c 'import secrets; "
                f"print(secrets.token_urlsafe(32))'` and set WEBAPP_SECRET_KEY."
            )
        if max_age_seconds <= 0:
            raise ValueError(f"max_age_seconds must be positive, got {max_age_seconds}")
        self._serializer = URLSafeTimedSerializer(secret_key, salt=SESSION_SALT)
        self._max_age_seconds = max_age_seconds

    @property
    def max_age_seconds(self) -> int:
        return self._max_age_seconds

    def encode(self, user_id: int) -> str:
        return self._serializer.dumps(user_id)

    def decode(self, token: str) -> int | None:
        try:
            payload: object = self._serializer.loads(token, max_age=self._max_age_seconds)
        except BadSignature:
            return None
        if not isinstance(payload, int):
            return None
        return payload

    def apply(self, response: Response, user_id: int) -> None:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=self.encode(user_id),
            max_age=self._max_age_seconds,
            httponly=True,
            secure=SESSION_COOKIE_SECURE,
            samesite="lax",
            path=SESSION_COOKIE_PATH,
        )

    def clear(self, response: Response) -> None:
        response.delete_cookie(SESSION_COOKIE_NAME, path=SESSION_COOKIE_PATH)
