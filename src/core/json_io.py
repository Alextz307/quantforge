"""Pure JSON read/write + typed field-extraction helpers.

Intentionally free of domain imports (sklearn, torch, our own
``TrainingMetadata`` etc.): this module is a generic namespace usable
anywhere the codebase reads or writes JSON. ``pandas`` is imported solely
for :func:`get_timestamp` — the JSON round-trip format for timestamps is
ISO strings, and handing callers back a ``pd.Timestamp`` is the one place
this module can't stay pandas-agnostic. Import as a namespace:

    from src.core import json_io

    data = json_io.read_dict(path)
    p_max = json_io.get_int(data, "p_max")
    alpha = json_io.get_float_list(data, "alpha")
    ts = json_io.get_timestamp(data, "train_end")

All ``get_*`` helpers raise ``KeyError`` with a named key on a missing field
and ``ValueError`` with a named key on a wrong-type field, so load paths
surface actionable errors instead of late-binding ``TypeError`` further down.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd


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


def diff_against_snapshot(
    actual: object,
    snapshot_path: str | Path,
    *,
    label: str,
    fix_command: str,
) -> list[str]:
    """Return error lines describing how the JSON file at ``snapshot_path`` drifts from ``actual``.

    Empty list means the snapshot is current. ``label`` appears verbatim in
    the human-readable messages (e.g. ``"OpenAPI snapshot"``); ``fix_command``
    is the shell command callers should run to regenerate the file.
    """
    snapshot = Path(snapshot_path)
    try:
        committed = read_dict(snapshot)
    except FileNotFoundError:
        return [
            f"{label} is missing at {snapshot}",
            f"  Run `{fix_command}` and commit the file.",
        ]
    if committed == actual:
        return []
    return [
        f"{label} at {snapshot} is stale",
        f"  Run `{fix_command}` and commit the regenerated file.",
    ]


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load a JSON-lines file (one JSON object per non-blank line).

    Each line must parse to an object; non-object lines raise
    :class:`ValueError` with the offending 1-based line number so a
    malformed record surfaces actionably instead of late-binding deep
    in caller code. Blank lines are skipped (trailing newlines are the
    common case).
    """
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"JSONL at {path} line {lineno} must be an object, got {type(parsed).__name__}"
                )
            records.append(parsed)
    return records


def write_jsonl(path: str | Path, records: Iterable[object]) -> None:
    """Write ``records`` to ``path`` as JSON-lines (one object per line).

    JSONL over one-big-JSON-array lets long-running writers be tailed
    via ``tail -f`` and the record count read via ``wc -l`` without
    parsing. ``records`` is iterated once and each element is dumped
    with sorted keys so two equivalent runs produce byte-identical
    files (load paths can hash for cache invalidation).
    """
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def append_jsonl(path: str | Path, record: object) -> None:
    """Append a single ``record`` to ``path`` as one JSON line.

    Used by streaming writers (HPO trial logs, progressive run dumps)
    where each call adds one record. ``sort_keys=True`` mirrors
    :func:`write_jsonl` so streamed and batched writers produce
    byte-identical content for the same record sequence.
    """
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")


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


def get_dict(d: dict[str, object], key: str) -> dict[str, object]:
    """Pull ``key`` out of ``d`` and require a nested object."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if not isinstance(value, dict):
        raise ValueError(f"JSON field {key!r} must be an object, got {type(value).__name__}")
    return value


def get_list_of_dicts(d: dict[str, object], key: str) -> list[dict[str, object]]:
    """Pull ``key`` out of ``d`` and require a list of nested objects.

    Each element is validated to be a dict; callers can then hand them off
    to a per-element ``from_dict`` without re-checking.
    """
    raw = _get_list(d, key)
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"JSON field {key!r}[{i}] must be an object, got {type(item).__name__}"
            )
    return [item for item in raw if isinstance(item, dict)]


def get_timestamp(d: dict[str, object], key: str) -> pd.Timestamp:
    """Pull ``key`` out of ``d`` and parse as a ``pd.Timestamp``.

    Accepts ISO strings (the canonical JSON round-trip format) and
    pre-parsed ``pd.Timestamp`` instances for callers that pass already-
    decoded dicts.
    """
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if isinstance(value, pd.Timestamp):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"JSON field {key!r} must be an ISO timestamp string, got {type(value).__name__}"
        )
    return pd.Timestamp(value)


def get_optional_str(d: dict[str, object], key: str) -> str | None:
    """Pull ``key`` if present and non-null; ``None`` otherwise."""
    raw = d.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"JSON field {key!r} must be a string or null, got {type(raw).__name__}")
    return raw


def get_optional_float(d: dict[str, object], key: str) -> float | None:
    """Pull ``key`` if present and non-null and narrow to ``float``."""
    raw = d.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"JSON field {key!r} must be a number or null, got {type(raw).__name__}")
    return float(raw)


def get_optional_iso_datetime(d: dict[str, object], key: str) -> datetime | None:
    """Pull ``key`` if present and non-null and parse as ``datetime``."""
    raw = d.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(
            f"JSON field {key!r} must be an ISO string or null, got {type(raw).__name__}"
        )
    return datetime.fromisoformat(raw)


def get_optional_timestamp(d: dict[str, object], key: str) -> pd.Timestamp | None:
    """Pull ``key`` if present and non-null and parse as ``pd.Timestamp``."""
    raw = d.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(
            f"JSON field {key!r} must be an ISO string or null, got {type(raw).__name__}"
        )
    return pd.Timestamp(raw)
