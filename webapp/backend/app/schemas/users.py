"""
Wire DTOs for user resources — password hashes never appear here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from webapp.backend.app.core.types import Role


class UserPublic(BaseModel):
    id: int
    username: str
    role: Role
    # ISO-8601 timestamp set only when the account was minted by the CLI
    # ``--user`` auto-create path (typo-stub candidates). ``None`` for
    # accounts created through the admin form, ``scripts.create_user``, or
    # any other deliberate path.
    auto_created_at: str | None = None


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=256)
    role: Role = Role.USER
