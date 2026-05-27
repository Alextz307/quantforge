"""Tests for the cross-trial GARCH grid cache (``src/models/_garch_cache.py``).

The cache exists to amortise the ``(p, q)`` AIC sweep across HPO trials
that share a fold's training window. These tests pin the contract:

* Identical-input fits return identical params on second call.
* Re-running with an overlapping ``(p_max, q_max)`` grid only fits the
  newly-introduced cells.
* Outside ``garch_cache_context``, behaviour is unchanged from the
  pre-cache code path.
* Distinct returns windows produce distinct cache keys (hash-keyed).
* Anti-leakage invariants on the strategy that owns GARCH still hold
  when the cache is active.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.utils import compute_log_returns
from src.models._garch_cache import GarchGridCache, active_cache, garch_cache_context
from src.models.garch import GARCHPredictor
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_synthetic_close_df

GRID_SMALL_P = 2
GRID_SMALL_Q = 2
GRID_LARGE_P = 3
GRID_LARGE_Q = 3
SECOND_DF_SEED = 7
SECOND_DF_ROWS = 250
SECOND_DF_START = "2022-01-03"


@pytest.fixture
def returns_a() -> pd.Series:
    df = make_synthetic_close_df()
    return compute_log_returns(df["close"]).dropna()


@pytest.fixture
def returns_b() -> pd.Series:
    df = make_synthetic_close_df(n_rows=SECOND_DF_ROWS, start=SECOND_DF_START, seed=SECOND_DF_SEED)
    return compute_log_returns(df["close"]).dropna()


def _fit_garch(returns: pd.Series, *, p_max: int, q_max: int) -> GARCHPredictor:
    g = GARCHPredictor(p_max=p_max, q_max=q_max)
    g.fit(pd.DataFrame({"r": returns}, index=returns.index), returns)
    return g


class TestGarchGridCache:
    def test_second_fit_reuses_cache(self, returns_a: pd.Series) -> None:
        cache = GarchGridCache()
        with garch_cache_context(cache):
            first = _fit_garch(returns_a, p_max=GRID_SMALL_P, q_max=GRID_SMALL_Q)
            cells_after_first = len(cache)
            second = _fit_garch(returns_a, p_max=GRID_SMALL_P, q_max=GRID_SMALL_Q)
            cells_after_second = len(cache)

        assert cells_after_first == GRID_SMALL_P * GRID_SMALL_Q
        assert cells_after_second == cells_after_first
        assert first._best_p == second._best_p
        assert first._best_q == second._best_q
        assert first._omega == pytest.approx(second._omega)

    def test_grid_extension_only_fits_missing_cells(self, returns_a: pd.Series) -> None:
        cache = GarchGridCache()
        with garch_cache_context(cache):
            _fit_garch(returns_a, p_max=GRID_SMALL_P, q_max=GRID_SMALL_Q)
            small_cells = len(cache)
            _fit_garch(returns_a, p_max=GRID_LARGE_P, q_max=GRID_LARGE_Q)
            large_cells = len(cache)

        assert small_cells == GRID_SMALL_P * GRID_SMALL_Q
        assert large_cells == GRID_LARGE_P * GRID_LARGE_Q

    def test_no_cache_when_context_unset(self, returns_a: pd.Series) -> None:
        assert active_cache() is None
        g = _fit_garch(returns_a, p_max=GRID_SMALL_P, q_max=GRID_SMALL_Q)
        assert active_cache() is None
        assert 1 <= g._best_p <= GRID_SMALL_P
        assert 1 <= g._best_q <= GRID_SMALL_Q

    def test_different_returns_dont_collide(
        self, returns_a: pd.Series, returns_b: pd.Series
    ) -> None:
        cache = GarchGridCache()
        with garch_cache_context(cache):
            _fit_garch(returns_a, p_max=GRID_SMALL_P, q_max=GRID_SMALL_Q)
            _fit_garch(returns_b, p_max=GRID_SMALL_P, q_max=GRID_SMALL_Q)

        assert len(cache) == 2 * GRID_SMALL_P * GRID_SMALL_Q

    def test_context_resets_on_exception(self, returns_a: pd.Series) -> None:
        cache = GarchGridCache()
        with pytest.raises(RuntimeError, match="boom"):
            with garch_cache_context(cache):
                raise RuntimeError("boom")
        # ContextVar must reset to None on exit so a subsequent fit doesn't
        # see a stale cache from the failed block.
        assert active_cache() is None

    def test_strategy_anti_leakage_invariant_holds_with_cache(self) -> None:
        """AdaptiveBollinger's TrainingMetadata overlap guard must still
        fire when the cache is active — the cache is pure memoisation
        and must not weaken any anti-leakage invariant.
        """

        cache = GarchGridCache()
        df = make_synthetic_close_df()
        with garch_cache_context(cache):
            strategy = AdaptiveBollingerStrategy(garch_p_max=GRID_SMALL_P, garch_q_max=GRID_SMALL_Q)
            strategy.train(df)
        meta = strategy.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(df)
