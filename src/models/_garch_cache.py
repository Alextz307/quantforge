"""
Cross-trial AIC cache for the GARCH ``(p, q)`` grid search.

Within an Optuna study, every HPO trial runs the same walk-forward fold
loop. For each fold, ``GARCHPredictor.fit`` calls ``_grid_search`` over
``(p, q) in [1, p_max] x [1, q_max]`` - up to 25 ``arch_model.fit`` calls
per fold-trial. The AIC of fitting ``GARCH(p, q)`` on a given returns
window is independent of the trial's *strategy-level* hyperparameters,
so two trials whose ``(p_max, q_max)`` overlap on the same fold are
re-fitting identical sub-grids. This module caches the per-cell AIC +
fitted result so the second trial onwards becomes a table lookup.

Architecture
------------
The cache is *study-scoped*, not strategy-scoped: the strategy /
``GARCHPredictor`` instance is rebuilt every trial via
``build_experiment``, so attaching the cache to the predictor would
discard it across trial boundaries. Threading the cache through
``ExperimentConfig`` is also blocked - that model is Pydantic-frozen and
serialisable, so it can't carry a runtime object. We use a
``ContextVar`` set by :class:`StrategyTuner.run` for the duration of
``study.optimize``; ``GARCHPredictor._grid_search`` reads the active
cache via :func:`active_cache` and falls back to the un-cached fit path
when no cache is set (CLI ``run``, holdout-eval, tests).

Keying
------
The cache key is ``(sha256(scaled_returns_bytes), p, q)``. Different
fold training windows produce different return arrays and therefore
distinct hashes, so cross-fold cache hits are impossible by
construction - only AIC values across HPO trials sharing the same fold
data alias.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arch.univariate.base import ARCHModelResult


@dataclass
class GarchGridCache:
    """
    In-memory ``(returns_hash, p, q) -> (aic, fitted_result)`` table.

    Mutable and not thread-safe - Optuna's ``n_jobs > 1`` mode runs
    trials in *threads* (not processes) sharing one Python interpreter,
    so concurrent dict writes from two trials grid-searching the same
    fold could race. Optuna's HPO config caps ``n_jobs`` at 4 in
    practice and the dict op is a single ``__setitem__``; we accept
    last-writer-wins on collision since both writers fit the identical
    ``(returns, p, q)`` triple to identical AICs anyway.
    """

    _table: dict[tuple[bytes, int, int], tuple[float, ARCHModelResult]] = field(
        default_factory=dict
    )

    @staticmethod
    def hash_returns(scaled: np.ndarray[tuple[int], np.dtype[np.float64]]) -> bytes:
        """
        SHA-256 of the scaled-returns bytes - stable across processes.
        """

        return hashlib.sha256(scaled.tobytes()).digest()

    def lookup_or_compute(
        self,
        returns_hash: bytes,
        p_max: int,
        q_max: int,
        fit_fn: Callable[[int, int], ARCHModelResult | None],
    ) -> tuple[ARCHModelResult | None, int, int, float]:
        """
        Resolve the best ``(p, q)`` within ``[1, p_max] x [1, q_max]``.

        For each cell, returns the cached fit if present; otherwise
        invokes ``fit_fn(p, q)`` and stores the result. ``fit_fn`` must
        return ``None`` on a fit failure (matching the existing
        ``_grid_search`` ``try/except`` convention) - failed cells are
        NOT cached so a future trial with a different ``arch`` build or
        numerical seed gets a fresh attempt.

        Returns ``(best_result, best_p, best_q, best_aic)``;
        ``best_result`` is ``None`` only when every cell failed (caller
        falls back to the GARCH(1,1) hard-coded path).
        """

        best_aic = math.inf
        best_p, best_q = 1, 1
        best_result: ARCHModelResult | None = None
        for p in range(1, p_max + 1):
            for q in range(1, q_max + 1):
                key = (returns_hash, p, q)
                cached = self._table.get(key)
                if cached is not None:
                    aic, fitted = cached
                else:
                    candidate = fit_fn(p, q)
                    if candidate is None:
                        continue
                    aic = float(candidate.aic)
                    fitted = candidate
                    self._table[key] = (aic, fitted)
                if aic < best_aic:
                    best_aic = aic
                    best_p, best_q = p, q
                    best_result = fitted
        return best_result, best_p, best_q, best_aic

    def __len__(self) -> int:
        return len(self._table)


_CACHE_CTX: ContextVar[GarchGridCache | None] = ContextVar("_GARCH_CACHE_CTX", default=None)


@contextmanager
def garch_cache_context(cache: GarchGridCache) -> Iterator[GarchGridCache]:
    """
    Bind ``cache`` as the active cache for the enclosed block.

    Restoration is unconditional (try/finally) so an exception raised
    inside the block doesn't leave a stale cache visible to subsequent
    code on the same thread.
    """

    token = _CACHE_CTX.set(cache)
    try:
        yield cache
    finally:
        _CACHE_CTX.reset(token)


def active_cache() -> GarchGridCache | None:
    """
    Return the cache bound by the innermost enclosing ``garch_cache_context``.
    """

    return _CACHE_CTX.get()
