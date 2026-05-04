"""Login rate-limiter wired into the FastAPI app via slowapi."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

LOGIN_RATE_LIMIT = "5 per 15 minutes"

login_limiter = Limiter(key_func=get_remote_address)


def attach(app: FastAPI) -> None:
    app.state.limiter = login_limiter
    app.add_exception_handler(
        RateLimitExceeded,
        cast(Callable[[Request, Exception], Response], _rate_limit_exceeded_handler),
    )
