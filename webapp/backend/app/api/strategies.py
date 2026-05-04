"""Registry introspection: list registered strategies."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.core.registry import strategy_registry
from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.schemas.registry import RegistryEntry

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
