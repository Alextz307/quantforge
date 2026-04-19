"""Pure JSON read/write + typed field-extraction helpers.

Intentionally free of any business-logic imports (sklearn, pandas, torch, our
own ``TrainingMetadata`` etc.): this module is a generic namespace usable
anywhere the codebase reads or writes JSON. Import it as a namespace:

    from src.core import json_io

    data = json_io.read_dict(path)
    p_max = json_io.get_int(data, "p_max")
    alpha = json_io.get_float_list(data, "alpha")

All ``get_*`` helpers raise ``KeyError`` with a named key on a missing field
and ``ValueError`` with a named key on a wrong-type field, so load paths
surface actionable errors instead of late-binding ``TypeError`` further down.
"""

from __future__ import annotations

import json
from pathlib import Path


def write(path: str | Path, obj: object) -> None:
    """Write ``obj`` as UTF-8 JSON at ``path`` with sorted keys and 2-space indent.

    Accepts ``object`` rather than a narrow union to match ``json.dump``'s own
    duck-typed contract — callers pass arbitrarily-nested dict/list/scalar
    payloads and invariance on ``dict[str, X]`` would otherwise force casts at
    every call site.
    """
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read(path: str | Path) -> object:
    """Load JSON from ``path``. Caller narrows the return type via ``isinstance``."""
    parsed: object = json.loads(Path(path).read_text(encoding="utf-8"))
    return parsed


def read_dict(path: str | Path) -> dict[str, object]:
    """Load JSON from ``path`` and require the top level to be an object."""
    raw = read(path)
    if not isinstance(raw, dict):
        raise ValueError(f"JSON at {path} must be an object, got {type(raw).__name__}")
    return raw


# --- Typed JSON field accessors --------------------------------------------
# Narrowing the values out of ``read_dict(...)`` is ceremonial (``int(str
# (raw[key]))``) when repeated at every load site. These helpers centralize
# the "read then narrow" pattern with uniform error messages, so a load() body
# reads as a flat list of field extractions rather than nested casts.


def get_int(d: dict[str, object], key: str) -> int:
    """Pull ``key`` out of ``d`` and narrow to ``int`` with a named error."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, int):
        # bool is an int subclass — reject to avoid ``True``/``False`` leaking in
        raise ValueError(f"JSON field {key!r} must be an int, got {type(value).__name__}")
    return value


def get_float(d: dict[str, object], key: str) -> float:
    """Pull ``key`` out of ``d`` and narrow to ``float`` (accepting ``int``)."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"JSON field {key!r} must be a number, got {type(value).__name__}")
    return float(value)


def get_bool(d: dict[str, object], key: str) -> bool:
    """Pull ``key`` out of ``d`` and require a ``bool``.

    Rejects ``int`` explicitly even though ``True``/``False`` are int
    subclasses — JSON ``true``/``false`` round-trip to Python ``bool``,
    and an int leaking in means someone hand-edited the file.
    """
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if not isinstance(value, bool):
        raise ValueError(f"JSON field {key!r} must be a bool, got {type(value).__name__}")
    return value


def get_str(d: dict[str, object], key: str) -> str:
    """Pull ``key`` out of ``d`` and require a ``str``."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if not isinstance(value, str):
        raise ValueError(f"JSON field {key!r} must be a string, got {type(value).__name__}")
    return value


def _get_list(d: dict[str, object], key: str) -> list[object]:
    """Module-private: pull ``key`` and require a ``list``. Callers should use
    a typed variant (``get_int_list``, ``get_float_list``, ``get_str_list``)
    — untyped element access leaves mypy unhappy."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if not isinstance(value, list):
        raise ValueError(f"JSON field {key!r} must be a list, got {type(value).__name__}")
    return value


def get_float_list(d: dict[str, object], key: str) -> list[float]:
    """Pull ``key`` out of ``d`` and require a list of numbers."""
    raw = _get_list(d, key)
    out: list[float] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"JSON field {key!r}[{i}] must be a number, got {type(item).__name__}")
        out.append(float(item))
    return out


def get_int_list(d: dict[str, object], key: str) -> list[int]:
    """Pull ``key`` out of ``d`` and require a list of integers."""
    raw = _get_list(d, key)
    out: list[int] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"JSON field {key!r}[{i}] must be an int, got {type(item).__name__}")
        out.append(item)
    return out


def get_str_list(d: dict[str, object], key: str) -> list[str]:
    """Pull ``key`` out of ``d`` and require a list of strings."""
    raw = _get_list(d, key)
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"JSON field {key!r}[{i}] must be a string, got {type(item).__name__}")
    return [str(item) for item in raw]
