"""
Dotted-path overrides for config dicts before pydantic validation.

The empirical study drives the same strategy YAML across many universes
without copying the ``strategy:`` section into one YAML per universe.
Rather than invent an inheritance scheme on top of the existing flat
YAMLs, the CLI accepts repeated ``--override key.path=value`` flags
and applies them to the loaded ``dict`` before pydantic re-validates.

Value parsing
-------------
Values are parsed with :func:`yaml.safe_load`, so the override surface
matches the YAML files users already write: bare numbers parse as
numbers, ``[A, B]`` parses as a list, ``true`` / ``false`` as bools,
``2024-01-01`` as a date. Quoting (``--override name="123"``) keeps
strings strings. Inventing a parser whose surprises differed from YAML
would defeat the point.

Typo safety
-----------
Every intermediate key on the path MUST already exist in the loaded
dict and resolve to a sub-dict. ``--override dat.tickers=[QQQ]`` (typo
for ``data``) raises :class:`ValueError` instead of silently inserting
a stub key that pydantic's ``extra="forbid"`` mode would later reject
with a less-actionable message. The leaf segment may either replace an
existing field or add a new one — pydantic catches genuine spelling
mistakes at the leaf via the same ``extra="forbid"``.
"""

from __future__ import annotations

from collections.abc import Sequence

import yaml


def apply_overrides(payload: dict[str, object], overrides: Sequence[str]) -> dict[str, object]:
    """
    Apply each ``key.path=value`` override in order; return ``payload``.

    The dict is mutated; callers that need to keep the original
    untouched should ``copy.deepcopy`` first.

    Raises:
        ValueError: malformed override (no ``=``, empty key, missing or
            non-dict intermediate key on the path).
    """

    for raw in overrides:
        if "=" not in raw:
            raise ValueError(
                f"override '{raw}' is missing '='; format is "
                f"key.path=value (e.g. data.tickers=[QQQ])."
            )
        path, raw_value = raw.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError(
                f"override '{raw}' has an empty key; format is "
                f"key.path=value (e.g. data.tickers=[QQQ])."
            )
        segments = path.split(".")
        value = yaml.safe_load(raw_value)
        cursor: dict[str, object] = payload
        for segment in segments[:-1]:
            child = cursor.get(segment)
            if not isinstance(child, dict):
                raise ValueError(
                    f"override '{raw}': intermediate key '{segment}' is "
                    f"missing or not a dict in the config; check spelling "
                    f"or add the key to the YAML before overriding "
                    f"underneath it."
                )
            cursor = child
        cursor[segments[-1]] = value
    return payload
