"""Wire DTOs for the configs validation + browsing endpoints."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ConfigKind(StrEnum):
    EXPERIMENT = "experiment"
    UNIVERSE = "universe"
    STUDY = "study"
    MODEL = "model"
    STRATEGY = "strategy"
    HPO = "hpo"
    REGIME = "regime"


class ConfigEntry(BaseModel):
    name: str


class ConfigDetail(BaseModel):
    name: str
    raw: str
    parsed: dict[str, object] | None = None
    parse_error: str | None = None


class ValidationErrorItem(BaseModel):
    loc: list[str]
    msg: str
    type: str


class ValidateRequest(BaseModel):
    kind: ConfigKind
    payload: dict[str, object]


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[ValidationErrorItem] = []
