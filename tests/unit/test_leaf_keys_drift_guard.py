"""Tests for the leaf-keys consistency drift guard.

The guard compares ``_LEAF_KEY_OWNED_PARAMS`` (config-layer collision map)
against each strategy's ``_leaf_keys`` ClassVar. A rename in one place
without the other silently breaks injection — the config-layer validator
might pass while the strategy ctor rejects the same key, or vice versa.

This test asserts the two sources of truth currently agree. If the check
ever drifts, the failing test names the offending strategy + key.
"""

from __future__ import annotations

from src.core.config import _LEAF_KEY_OWNED_PARAMS
from src.core.registry import strategy_registry


def _populate_registry() -> None:
    import src.strategies  # noqa: F401  — registers every @strategy_registry.register


def test_leaf_keys_consistent_with_config_map() -> None:
    _populate_registry()
    for strategy_name, leaf_map in _LEAF_KEY_OWNED_PARAMS.items():
        assert strategy_name in strategy_registry, (
            f"_LEAF_KEY_OWNED_PARAMS references unregistered strategy {strategy_name!r}"
        )
        cls = strategy_registry.get(strategy_name)
        strategy_keys: frozenset[str] = getattr(cls, "_leaf_keys", frozenset())
        config_keys = frozenset(leaf_map)
        assert config_keys == strategy_keys, (
            f"leaf-key drift for {strategy_name}: "
            f"_LEAF_KEY_OWNED_PARAMS has {sorted(config_keys)!r} but "
            f"_leaf_keys has {sorted(strategy_keys)!r}"
        )


def test_non_ml_strategies_have_empty_config_map() -> None:
    """Strategies with empty ``_leaf_keys`` must not appear in the config
    map — otherwise users could pass collision hyperparameters through
    strategy.params and the validator wouldn't catch it.
    """
    _populate_registry()
    for strategy_name in strategy_registry.list_all():
        cls = strategy_registry.get(strategy_name)
        strategy_keys: frozenset[str] = getattr(cls, "_leaf_keys", frozenset())
        if not strategy_keys:
            assert strategy_name not in _LEAF_KEY_OWNED_PARAMS, (
                f"{strategy_name} has empty _leaf_keys but appears in "
                f"_LEAF_KEY_OWNED_PARAMS — remove the stale entry."
            )
