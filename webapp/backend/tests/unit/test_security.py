"""
Verify session-cookie sign/verify, secret-key validation, and tamper detection.
"""

from __future__ import annotations

import secrets

import pytest

from webapp.backend.app.core.security import MIN_SECRET_KEY_LENGTH, SessionCookies

VALID_SECRET = secrets.token_urlsafe(48)
SECONDS_PER_HOUR = 3600
TEST_USER_ID = 7


def _cookies(secret: str = VALID_SECRET, max_age: int = SECONDS_PER_HOUR) -> SessionCookies:
    return SessionCookies(secret_key=secret, max_age_seconds=max_age)


def test_encode_decode_round_trips() -> None:
    cookies = _cookies()

    token = cookies.encode(TEST_USER_ID)

    assert cookies.decode(token) == TEST_USER_ID


def test_short_secret_key_raises() -> None:
    short = "x" * (MIN_SECRET_KEY_LENGTH - 1)

    with pytest.raises(ValueError, match="at least"):
        SessionCookies(secret_key=short, max_age_seconds=SECONDS_PER_HOUR)


def test_non_positive_max_age_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        SessionCookies(secret_key=VALID_SECRET, max_age_seconds=0)


def test_tampered_token_decodes_to_none() -> None:
    cookies = _cookies()
    token = cookies.encode(TEST_USER_ID)
    tampered = token[:-2] + "AA" if token[-2:] != "AA" else token[:-2] + "BB"

    assert cookies.decode(tampered) is None


def test_token_signed_with_different_secret_decodes_to_none() -> None:
    other_secret = secrets.token_urlsafe(48)
    token = SessionCookies(other_secret, SECONDS_PER_HOUR).encode(TEST_USER_ID)

    cookies = _cookies()

    assert cookies.decode(token) is None


def test_max_age_seconds_is_exposed() -> None:
    cookies = _cookies(max_age=SECONDS_PER_HOUR)

    assert cookies.max_age_seconds == SECONDS_PER_HOUR
