"""Best-effort ctor introspection for the hybrid configure form.

The frontend renders typed inputs for ``int|float|str|bool|enum`` params
and falls back to a JSON editor for ``complex``: union types
(``Device | None``), generics (``list[str]``, ``Mapping[...]``), forward
refs that fail to resolve. Recursive type unwrapping is intentionally
out of scope — the JSON editor handles every shape the typed bucket
can't.
"""

from __future__ import annotations

import enum
import inspect
import types
import typing

from src.core.registry import strategy_registry
from webapp.backend.app.schemas.strategies import (
    ParamKind,
    StrategyParam,
    StrategySchema,
)

# Composite-only kwarg the experiment builder injects from
# ``ExperimentConfig.pretrained_leaves``; never user-form input.
_HIDDEN_PARAMS: frozenset[str] = frozenset({"self", "pretrained_leaves"})


def _classify(annotation: object) -> tuple[ParamKind, list[str] | None]:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        # ``X | None`` and ``X | Y`` both fall through to the JSON editor.
        return ParamKind.COMPLEX, None
    if isinstance(annotation, type):
        # ``bool`` must be checked before ``int`` (it's a subclass of int).
        if annotation is bool:
            return ParamKind.BOOL, None
        if annotation is int:
            return ParamKind.INT, None
        if annotation is float:
            return ParamKind.FLOAT, None
        if annotation is str:
            return ParamKind.STR, None
        if issubclass(annotation, enum.Enum):
            return ParamKind.ENUM, [str(member.value) for member in annotation]
    return ParamKind.COMPLEX, None


def _default_for_wire(value: object, kind: ParamKind) -> object | None:
    if kind is ParamKind.ENUM and isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Lists / dicts / Path / arbitrary objects — surface as a string so
    # the JSON editor has a starting hint without serializing wrong.
    return repr(value)


def describe_strategy(name: str) -> StrategySchema:
    """Return the form-backing schema for ``strategy_registry.get(name)``.

    Raises ``KeyError`` (with the registry's "available: [...]" hint) if
    ``name`` is not registered.
    """
    cls = strategy_registry.get(name)
    hints: dict[str, object] = {}
    try:
        hints = dict(typing.get_type_hints(cls.__init__))
    except Exception:  # noqa: BLE001
        # Forward refs that can't be resolved (rare) — every annotation
        # falls back to ``complex`` via ``inspect.Parameter.annotation``.
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
            kind, choices = ParamKind.COMPLEX, None
        else:
            kind, choices = _classify(annotation)
        required = parameter.default is inspect.Parameter.empty
        default = None if required else _default_for_wire(parameter.default, kind)
        params.append(
            StrategyParam(
                name=pname,
                kind=kind,
                required=required,
                default=default,
                choices=choices,
            )
        )

    return StrategySchema(
        name=name,
        qualname=f"{cls.__module__}.{cls.__qualname__}",
        params=params,
    )
