"""
Best-effort ctor introspection for the hybrid configure form.

The frontend renders typed inputs for ``int|float|str|bool|enum`` params
and falls back to a JSON editor for ``complex``: non-Optional union types
(``int | str``), generics (``list[str]``, ``Mapping[...]``), forward refs
that fail to resolve. ``Optional[T]`` / ``T | None`` is peeled - the
inner ``T`` drives the kind, and ``nullable=True`` is set so the form
labels the empty option "- none -" and (for Enums) renders a typed
dropdown instead of a JSON textarea. Recursive type unwrapping beyond
one Optional layer is intentionally out of scope.
"""

from __future__ import annotations

import collections.abc
import enum
import inspect
import re
import types
import typing
from pathlib import Path

import yaml

from src.core.device import available_devices
from src.core.registry import strategy_registry
from src.core.types import Device
from webapp.backend.app.schemas.strategies import (
    ParamKind,
    StrategyParam,
    StrategySchema,
)

# Ctor params that are framework-injected or duplicate a top-level config
# block, so they must never surface as form fields:
#   * ``self`` - Python receiver.
#   * ``interval`` - always equal to ``data.interval``; showing it as a
#     strategy param duplicates the data-block field and lets the user
#     desync the two.
_HIDDEN_PARAMS: frozenset[str] = frozenset({"self", "interval"})


def _peel_optional(annotation: object) -> tuple[object, bool]:
    """
    Strip ``None`` from ``T | None`` / ``Optional[T]``; return ``(T, True)``.

    Multi-member unions (``int | str``) are not Optional and stay as-is.
    """

    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return annotation, False


_STR_LIST_ORIGINS: tuple[object, ...] = (
    list,
    collections.abc.Sequence,
    collections.abc.Iterable,
    collections.abc.Collection,
)


def _is_str_list_annotation(annotation: object) -> bool:
    """
    Detect ``list[str]`` / ``tuple[str, ...]`` / ``Sequence[str]`` etc.

    Used to render the comma/space-separated text input instead of forcing
    the user to hand-write a JSON list for ``feature_columns`` etc.
    """

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is tuple and len(args) == 2 and args[0] is str and args[1] is Ellipsis:
        return True
    return origin in _STR_LIST_ORIGINS and args == (str,)


def _classify(annotation: object) -> tuple[ParamKind, list[str] | None, bool]:
    inner, nullable = _peel_optional(annotation)
    if _is_str_list_annotation(inner):
        return ParamKind.STR_LIST, None, nullable
    origin = typing.get_origin(inner)
    if origin is typing.Union or origin is types.UnionType:
        return ParamKind.COMPLEX, None, nullable
    if isinstance(inner, type):
        # ``bool`` must precede ``int`` - bool is a subclass of int.
        if inner is bool:
            return ParamKind.BOOL, None, nullable
        if inner is int:
            return ParamKind.INT, None, nullable
        if inner is float:
            return ParamKind.FLOAT, None, nullable
        if inner is str:
            return ParamKind.STR, None, nullable
        if issubclass(inner, enum.Enum):
            return ParamKind.ENUM, _enum_choices(inner), nullable
    return ParamKind.COMPLEX, None, nullable


def _enum_choices(enum_cls: type[enum.Enum]) -> list[str]:
    """
    Materialise an Enum's values as wire strings, host-aware for ``Device``.
    """

    if enum_cls is Device:
        usable = {d.value for d in available_devices()}
        return [str(member.value) for member in enum_cls if member.value in usable]
    return [str(member.value) for member in enum_cls]


def _default_for_wire(value: object, kind: ParamKind) -> object | None:
    if kind is ParamKind.ENUM and isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Lists / dicts / Path / arbitrary objects - surface as a string so
    # the JSON editor has a starting hint without serializing wrong.
    return repr(value)


def _strategy_yaml_stem(name: str) -> str:
    """
    ``CrossAssetMomentum`` -> ``cross_asset_momentum``.
    """

    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def get_canonical_strategy_params(config_root: Path, name: str) -> dict[str, object] | None:
    """
    Best-effort load of ``<config_root>/strategies/<snake_case>.yaml`` ->
    ``strategy.params``. Returns ``None`` for missing/malformed YAML and
    strips ``_HIDDEN_PARAMS`` so the form never receives framework-managed
    keys it would have to filter again.
    """

    yaml_path = config_root / "strategies" / f"{_strategy_yaml_stem(name)}.yaml"
    try:
        raw = yaml_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None

    strategy = parsed.get("strategy")
    if not isinstance(strategy, dict):
        return None
    params = strategy.get("params")
    if not isinstance(params, dict):
        return None

    return {k: v for k, v in params.items() if k not in _HIDDEN_PARAMS}


def describe_strategy(name: str) -> StrategySchema:
    """
    Return the form-backing schema for ``strategy_registry.get(name)``.

    Raises ``KeyError`` (with the registry's "available: [...]" hint) if
    ``name`` is not registered.
    """

    cls = strategy_registry.get(name)
    uses_xgboost = bool(getattr(cls, "uses_xgboost", False))
    hints: dict[str, object] = {}
    try:
        hints = dict(typing.get_type_hints(cls.__init__))
    except Exception:  # noqa: BLE001
        pass
    sig = inspect.signature(cls.__init__)

    params: list[StrategyParam] = []
    for pname, parameter in sig.parameters.items():
        if pname in _HIDDEN_PARAMS:
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation: object = hints.get(pname, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            kind, choices, nullable = ParamKind.COMPLEX, None, False
        else:
            kind, choices, nullable = _classify(annotation)
            if uses_xgboost and choices is not None and Device.MPS.value in choices:
                choices = [c for c in choices if c != Device.MPS.value]
        required = parameter.default is inspect.Parameter.empty
        default = None if required else _default_for_wire(parameter.default, kind)
        params.append(
            StrategyParam(
                name=pname,
                kind=kind,
                required=required,
                nullable=nullable,
                default=default,
                choices=choices,
            )
        )

    return StrategySchema(
        name=name,
        qualname=f"{cls.__module__}.{cls.__qualname__}",
        params=params,
    )
