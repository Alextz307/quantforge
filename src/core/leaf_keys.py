"""Pretrained-leaf key constants + error-text helper.

Owns the canonical ``"<leaf_key>"`` strings every strategy and config
layer agrees on. Strategy modules import these instead of redeclaring
the same literal twice (once on the class, once in ``config.py``'s
``_LEAF_KEY_OWNED_PARAMS`` table). Lives in ``src.core`` rather than
``src.orchestration`` so it can be imported by ``src.core.config``
without inverting the core→orchestration layering.
"""

from __future__ import annotations

from collections.abc import Iterable

LEAF_KEY_DIRECTIONAL_CLASSIFIER = "directional_classifier"
LEAF_KEY_RETURN_MODEL = "return_model"
LEAF_KEY_VOL_MODEL = "vol_model"


def describe_supported_leaf_keys(supported_keys: Iterable[str], strategy_cls_name: str) -> str:
    """Render the "supported keys" hint for pretrained-leaf error messages."""
    materialised = frozenset(supported_keys)
    if materialised:
        return f"supported keys: {sorted(materialised)}"
    return f"{strategy_cls_name} owns no ML leaves — pretrained_leaves must be empty"
