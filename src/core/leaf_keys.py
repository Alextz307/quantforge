"""Pretrained-leaf key error-text helper.

Tiny shared helper that keeps the config-layer validator and the
strategy-ctor normaliser from drifting on wording ("owns no ML leaves"
phrasing vs "supported keys: [...]"). Lives in ``src.core`` rather than
``src.orchestration`` so it can be imported by ``src.core.config``
without inverting the core→orchestration layering.
"""

from __future__ import annotations

from collections.abc import Iterable


def describe_supported_leaf_keys(supported_keys: Iterable[str], strategy_cls_name: str) -> str:
    """Render the "supported keys" hint for pretrained-leaf error messages."""
    materialised = frozenset(supported_keys)
    if materialised:
        return f"supported keys: {sorted(materialised)}"
    return f"{strategy_cls_name} owns no ML leaves — pretrained_leaves must be empty"
