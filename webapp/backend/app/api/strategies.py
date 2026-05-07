"""Registry introspection: list registered strategies + per-strategy schema."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.core.registry import strategy_registry
from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.schemas.registry import RegistryEntry
from webapp.backend.app.schemas.strategies import StrategySchema
from webapp.backend.app.services.strategy_service import describe_strategy

router = APIRouter(
    prefix="/strategies", tags=["strategies"], dependencies=[Depends(get_current_user)]
)


@router.get("", response_model=list[RegistryEntry])
def get_strategies() -> list[RegistryEntry]:
    entries: list[RegistryEntry] = []
    for name in sorted(strategy_registry.list_all()):
        cls = strategy_registry.get(name)
        entries.append(RegistryEntry(name=name, qualname=f"{cls.__module__}.{cls.__qualname__}"))
    return entries


@router.get("/{name}/schema", response_model=StrategySchema)
def get_strategy_schema(name: str) -> StrategySchema:
    try:
        return describe_strategy(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown strategy: {name}",
        ) from exc
