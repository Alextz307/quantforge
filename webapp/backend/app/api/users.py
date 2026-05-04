"""Admin-only user CRUD endpoints."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status

from webapp.backend.app.core.deps import get_db, require_admin
from webapp.backend.app.schemas.users import UserCreate, UserPublic
from webapp.backend.app.services.user_service import (
    UsernameAlreadyExistsError,
    create_user,
    list_users,
    soft_delete_user,
)

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_admin)])


@router.get("", response_model=list[UserPublic])
def get_users(conn: sqlite3.Connection = Depends(get_db)) -> list[UserPublic]:
    return list_users(conn)


@router.post("", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def post_user(body: UserCreate, conn: sqlite3.Connection = Depends(get_db)) -> UserPublic:
    try:
        return create_user(conn, username=body.username, password=body.password, role=body.role)
    except UsernameAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, conn: sqlite3.Connection = Depends(get_db)) -> None:
    if not soft_delete_user(conn, user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
