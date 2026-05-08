"""Read-only HTTP endpoints over persisted HPO studies."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status

from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.hpo import HpoDetail, HpoSummary, TrialRow
from webapp.backend.app.services.hpo_service import (
    HpoStudyNotFoundError,
    find_live_job_for,
    get_hpo_study,
    list_hpo_studies,
    list_trials,
)

router = APIRouter(prefix="/hpo", tags=["hpo"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[HpoSummary])
def get_hpo_studies() -> list[HpoSummary]:
    return list_hpo_studies(get_settings().store_root)


@router.get("/{name}", response_model=HpoDetail)
def get_hpo_study_detail(
    name: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> HpoDetail:
    try:
        live_job_id = find_live_job_for(conn, name)
        return get_hpo_study(get_settings().store_root, name, live_job_id=live_job_id)
    except HpoStudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/trials", response_model=list[TrialRow])
def get_hpo_trials(name: str, after_trial: int | None = None) -> list[TrialRow]:
    try:
        return list_trials(get_settings().store_root, name, after_trial=after_trial)
    except HpoStudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
