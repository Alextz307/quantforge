"""Guard against drift between ``_LEAF_KEY_OWNED_PARAMS`` and strategy ``_leaf_keys``.

Two sources of truth describe the same set of pretrained-leaf keys:

* ``src/core/config.py::_LEAF_KEY_OWNED_PARAMS`` — per-strategy map of
  ``{strategy_name: {leaf_key: (owned_hyperparam, ...)}}`` used by the
  config-layer collision validator.
* Each concrete ``IStrategy`` subclass's ``_leaf_keys: ClassVar[frozenset[str]]``
  — used by ``normalize_pretrained_leaves`` at ctor time.

A rename in one place without the other silently breaks injection (the
config validator passes but the ctor rejects, or vice versa). This guard
catches that drift before the bug ships.

Run locally with ``python scripts/check_leaf_keys_consistent.py``; wired
into CI so a drift lands a failing check.
"""

from __future__ import annotations

import sys

from src.core.config import _LEAF_KEY_OWNED_PARAMS
from src.core.registry import strategy_registry


def main() -> int:
    # Populate the registry — importing ``src.strategies`` runs every
    # strategy module's ``@strategy_registry.register`` decorator.
    import src.strategies  # noqa: F401

    errors: list[str] = []
    for strategy_name, leaf_map in _LEAF_KEY_OWNED_PARAMS.items():
        if strategy_name not in strategy_registry:
            errors.append(f"_LEAF_KEY_OWNED_PARAMS references unknown strategy {strategy_name!r}")
            continue
        cls = strategy_registry.get(strategy_name)
        strategy_keys: frozenset[str] = getattr(cls, "_leaf_keys", frozenset())
        config_keys = frozenset(leaf_map)
        extra_in_config = config_keys - strategy_keys
        extra_in_strategy = strategy_keys - config_keys
        if extra_in_config:
            errors.append(
                f"{strategy_name}: _LEAF_KEY_OWNED_PARAMS has keys "
                f"{sorted(extra_in_config)!r} missing from _leaf_keys"
            )
        if extra_in_strategy:
            errors.append(
                f"{strategy_name}: _leaf_keys has keys "
                f"{sorted(extra_in_strategy)!r} missing from _LEAF_KEY_OWNED_PARAMS"
            )

    if errors:
        print("leaf-keys drift detected:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "Fix: align _LEAF_KEY_OWNED_PARAMS (src/core/config.py) with each "
            "strategy's _leaf_keys ClassVar.",
            file=sys.stderr,
        )
        return 1

    print(f"leaf-keys consistent across {len(_LEAF_KEY_OWNED_PARAMS)} strategy entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
