"""Registry introspection: list registered predictors + classifiers."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.core.registry import classifier_registry, model_registry
from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.schemas.registry import ModelKind, ModelRegistryEntry

router = APIRouter(prefix="/models", tags=["models"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[ModelRegistryEntry])
def get_models() -> list[ModelRegistryEntry]:
    entries: list[ModelRegistryEntry] = []
    for name in model_registry.list_public():
        cls = model_registry.get(name)
        entries.append(
            ModelRegistryEntry(
                name=name,
                qualname=f"{cls.__module__}.{cls.__qualname__}",
                kind=ModelKind.PREDICTOR,
            )
        )
    for name in classifier_registry.list_public():
        clf = classifier_registry.get(name)
        entries.append(
            ModelRegistryEntry(
                name=name,
                qualname=f"{clf.__module__}.{clf.__qualname__}",
                kind=ModelKind.CLASSIFIER,
            )
        )
    entries.sort(key=lambda e: (e.kind.value, e.name))
    return entries
