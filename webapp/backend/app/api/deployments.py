"""HTTP routes for live-inference deployments."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.core.exceptions import LeakageError, WarmupInsufficientError
from src.engine.scenarios import SlippageScenario
from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.deployments import (
    DeploymentCreate,
    DeploymentDetail,
    DeploymentRename,
    DeploymentSummary,
    PredictIfStaleResponse,
    SignalEvaluationOut,
    SignalRowOut,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.deployment_service import (
    DeploymentAccessDeniedError,
    DeploymentNotFoundError,
    DeploymentSourceInvalidError,
    create_deployment,
    delete_deployment,
    evaluate_signal_log,
    get_deployment,
    list_deployments,
    predict_if_stale,
    read_signal_log,
    rename_deployment,
)

router = APIRouter(prefix="/deployments", tags=["deployments"])


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("", response_model=list[DeploymentSummary])
def get_deployments(
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[DeploymentSummary]:
    return list_deployments(conn, user=user, all_users=all_users)


@router.post(
    "",
    response_model=DeploymentDetail,
    status_code=status.HTTP_201_CREATED,
)
def post_deployment(
    body: DeploymentCreate,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> DeploymentDetail:
    try:
        return create_deployment(
            conn,
            store_root=get_settings().store_root,
            user=user,
            source_kind=body.source_kind,
            source_id=body.source_id,
            name=body.name,
            warmup_bars=body.warmup_bars,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DeploymentSourceInvalidError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get("/{deployment_id}", response_model=DeploymentDetail)
def get_deployment_detail(
    deployment_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> DeploymentDetail:
    try:
        return get_deployment(
            conn,
            store_root=get_settings().store_root,
            user=user,
            deployment_id=deployment_id,
        )
    except (DeploymentNotFoundError, DeploymentAccessDeniedError) as exc:
        raise _not_found(exc) from exc


@router.patch("/{deployment_id}", response_model=DeploymentDetail)
def patch_deployment(
    deployment_id: str,
    body: DeploymentRename,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> DeploymentDetail:
    try:
        return rename_deployment(
            conn,
            store_root=get_settings().store_root,
            user=user,
            deployment_id=deployment_id,
            new_name=body.name,
        )
    except (DeploymentNotFoundError, DeploymentAccessDeniedError) as exc:
        raise _not_found(exc) from exc


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_deployment_route(
    deployment_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> None:
    try:
        delete_deployment(
            conn,
            store_root=get_settings().store_root,
            user=user,
            deployment_id=deployment_id,
        )
    except (DeploymentNotFoundError, DeploymentAccessDeniedError) as exc:
        raise _not_found(exc) from exc


@router.get("/{deployment_id}/signals", response_model=list[SignalRowOut])
def get_signal_log(
    deployment_id: str,
    limit: int | None = Query(None, ge=1, le=10000),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[SignalRowOut]:
    try:
        return read_signal_log(
            conn,
            store_root=get_settings().store_root,
            user=user,
            deployment_id=deployment_id,
            limit=limit,
        )
    except (DeploymentNotFoundError, DeploymentAccessDeniedError) as exc:
        raise _not_found(exc) from exc


@router.get("/{deployment_id}/signal-evaluation", response_model=SignalEvaluationOut)
def get_signal_evaluation(
    deployment_id: str,
    cost: SlippageScenario = Query(SlippageScenario.NORMAL),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> SignalEvaluationOut:
    """Score the deployment's emitted signals open->open; ``cost`` sets the net friction tier."""

    try:
        return evaluate_signal_log(
            conn,
            store_root=get_settings().store_root,
            user=user,
            deployment_id=deployment_id,
            cost_scenario=cost,
        )
    except (DeploymentNotFoundError, DeploymentAccessDeniedError) as exc:
        raise _not_found(exc) from exc


@router.post("/{deployment_id}/predict-if-stale", response_model=PredictIfStaleResponse)
def post_predict_if_stale(
    deployment_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> PredictIfStaleResponse:
    """Recall today's signal or compute it now."""

    try:
        return predict_if_stale(
            conn,
            store_root=get_settings().store_root,
            user=user,
            deployment_id=deployment_id,
        )
    except (DeploymentNotFoundError, DeploymentAccessDeniedError) as exc:
        raise _not_found(exc) from exc
    except (LeakageError, WarmupInsufficientError, DeploymentSourceInvalidError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
