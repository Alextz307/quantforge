"""Contextual logger wrapper — prefixes every message with caller-bound context.

Pure stdlib, no structlog / loguru. ``get_logger(name, **context)`` returns a
:class:`logging.LoggerAdapter` whose ``process()`` prepends the bound key/value
pairs to every emitted message:

    logger = get_logger(__name__, experiment_id="abc", strategy_name="Strat")
    logger.info("fold 2/5 complete")
    # → INFO src.orchestration.experiment:
    #     [experiment_id=abc strategy_name=Strat] fold 2/5 complete

This approach lets the root logging format string stay simple (no custom
``%(experiment_id)s`` attributes) while still surfacing the context in every
downstream record. Modules that don't need context keep using
``logging.getLogger(__name__)`` unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import Any


# ``logging.LoggerAdapter`` became generic in Python 3.12 (parameterised over
# the logger it wraps), but typeshed doesn't yet publish the ``[LoggerT]``
# default, so parameterising would require ``logging.LoggerAdapter[Logger]``
# which isn't version-portable. The ``type-arg`` ignore is the typeshed
# round-trip cost; remove once stubs stabilise.
class _ContextAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """LoggerAdapter that formats bound context as ``[k1=v1 k2=v2] <msg>``.

    ``self.extra`` is the context dict passed to :func:`get_logger`.
    ``process()`` runs on every log call — we keep it allocation-light by
    using ``" ".join`` over a generator rather than materialising an
    intermediate list.
    """

    # ``msg``/``kwargs`` use ``Any`` because the parent ``LoggerAdapter.process``
    # signature does; narrowing would violate Liskov substitution and break
    # legitimate callers who pass LogRecord-compatible objects.
    def process(
        self,
        msg: Any,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = self.extra or {}
        if not extra:
            return msg, kwargs
        ctx = " ".join(f"{k}={v}" for k, v in extra.items())
        return f"[{ctx}] {msg}", kwargs


def get_logger(name: str, **context: object) -> logging.LoggerAdapter:  # type: ignore[type-arg]
    """Return a context-bound logger.

    ``name`` is the standard ``logging.getLogger(name)`` handle; ``context``
    becomes a persistent prefix on every message. Empty context is legal —
    the adapter degrades to a zero-cost passthrough in ``process()``.

    Return-type ``type-arg`` ignore matches the reason on ``_ContextAdapter``:
    Python 3.12 made ``LoggerAdapter`` generic but the version-portable
    parameterisation isn't stable in typeshed yet.
    """
    base = logging.getLogger(name)
    extra: Mapping[str, object] = dict(context)
    return _ContextAdapter(base, extra)
