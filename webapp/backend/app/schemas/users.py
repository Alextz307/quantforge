"""Wire DTOs for user resources — password hashes never appear here."""

from __future__ import annotations

from pydantic import BaseModel, Field

from webapp.backend.app.core.types import Role


class UserPublic(BaseModel):
    id: int
    username: str
    role: Role


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=256)
    role: Role = Role.USER
