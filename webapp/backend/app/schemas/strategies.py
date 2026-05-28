"""
Wire DTOs for the strategy schema-introspection endpoint.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ParamKind(StrEnum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    ENUM = "enum"
    # ``list[str]`` / ``tuple[str, ...]`` / ``Sequence[str]`` — rendered as a
    # comma-or-space separated text input (mirrors the data-block ``Tickers``
    # field). Far better UX than forcing the user to hand-write JSON for
    # ``feature_columns`` / ``feature_tickers``.
    STR_LIST = "str_list"
    COMPLEX = "complex"


class StrategyParam(BaseModel):
    name: str
    kind: ParamKind
    required: bool
    # ``nullable`` is True when the ctor annotation is ``Optional[T]`` /
    # ``T | None``. The form uses it to label the empty-option as
    # "— none —" instead of "— use default —" and to render
    # ``Optional[Enum]`` as a typed dropdown rather than a JSON textarea.
    nullable: bool = False
    default: object | None = None
    choices: list[str] | None = None


class StrategySchema(BaseModel):
    name: str
    qualname: str
    params: list[StrategyParam]
    # Best-effort hydration of the form from ``config/strategies/<name>.yaml``'s
    # ``strategy.params`` block — the same defaults the canonical CLI sweep
    # uses. ``None`` when no canonical YAML exists. Framework-managed params
    # (``interval``) are stripped before the dict ships.
    canonical_params: dict[str, object] | None = None
