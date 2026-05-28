"""
Wire DTOs for registry-introspection endpoints.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ModelKind(StrEnum):
    PREDICTOR = "predictor"
    CLASSIFIER = "classifier"


class RegistryEntry(BaseModel):
    name: str
    qualname: str


class ModelRegistryEntry(RegistryEntry):
    kind: ModelKind
