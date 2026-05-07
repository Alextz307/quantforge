"""Wire DTOs for the strategy schema-introspection endpoint."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ParamKind(StrEnum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    ENUM = "enum"
    COMPLEX = "complex"


class StrategyParam(BaseModel):
    name: str
    kind: ParamKind
    required: bool
    default: object | None = None
    choices: list[str] | None = None


class StrategySchema(BaseModel):
    name: str
    qualname: str
    params: list[StrategyParam]
