"""
Parity tests for the ``SpreadCalculator`` C++ binding.

gtest exhaustively covers the numeric engine (see
``cpp/tests/test_spread.cpp``). These tests lock in the **binding layer**:
numpy array marshalling, keyword-argument surface, NaN passthrough, and
parity with a pure-pandas reference for the rolling z-score.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import quant_engine as qe

F64Array = npt.NDArray[np.float64]

HEDGE_RATIO = 1.25
ZSCORE_WINDOW = 30
LONG_SERIES_LEN = 500
LONG_SERIES_SEED_A = 42
LONG_SERIES_SEED_B = 99
# Welford-vs-naive ordering noise in fp64 is comfortably under 1e-10 for
# 500 bars of log-normal noise.
RSTD_PARITY_RTOL = 1e-10


def _pandas_zscore_reference(spread: F64Array, window: int) -> F64Array:
    """
    pandas ``rolling(window).mean/std`` → z-score; zero-std bars → NaN.
    """

    s = pd.Series(spread)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    z = (s - mean) / std.where(std > 0.0, other=np.nan)
    return np.asarray(z, dtype=np.float64)


class TestComputeSpread:
    def test_hand_computed_parity(self) -> None:
        a = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        got = qe.SpreadCalculator.compute_spread(a=a, b=b, hedge_ratio=2.0)
        np.testing.assert_array_equal(got, np.array([8.0, 16.0, 24.0]))

    def test_length_mismatch_raises(self) -> None:
        a = np.ones(5, dtype=np.float64)
        b = np.ones(4, dtype=np.float64)
        with pytest.raises(ValueError, match="same length"):
            qe.SpreadCalculator.compute_spread(a=a, b=b, hedge_ratio=1.0)

    def test_empty_input_returns_empty(self) -> None:
        empty = np.array([], dtype=np.float64)
        got = qe.SpreadCalculator.compute_spread(a=empty, b=empty, hedge_ratio=1.0)
        assert got.shape == (0,)


class TestComputeZScore:
    def test_leading_window_is_nan(self) -> None:
        spread = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        got = qe.SpreadCalculator.compute_zscore(spread=spread, window=3)
        assert np.isnan(got[:2]).all()
        assert not np.isnan(got[2:]).any()

    def test_constant_series_is_nan(self) -> None:
        spread = np.full(10, 42.0, dtype=np.float64)
        got = qe.SpreadCalculator.compute_zscore(spread=spread, window=3)
        assert np.isnan(got).all()

    def test_matches_pandas_reference_on_random_walk(self) -> None:
        rng = np.random.default_rng(LONG_SERIES_SEED_A)
        spread = np.cumsum(rng.normal(0.0, 1.0, LONG_SERIES_LEN))
        got = qe.SpreadCalculator.compute_zscore(spread=spread, window=ZSCORE_WINDOW)
        ref = _pandas_zscore_reference(spread, ZSCORE_WINDOW)

        # NaN positions must match exactly.
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))

        mask = ~np.isnan(got)
        np.testing.assert_allclose(got[mask], ref[mask], rtol=RSTD_PARITY_RTOL)

    def test_window_below_two_raises(self) -> None:
        spread = np.arange(10, dtype=np.float64)
        with pytest.raises(ValueError, match=">= 2"):
            qe.SpreadCalculator.compute_zscore(spread=spread, window=1)


class TestPairsTradingStrategyBinding:
    def test_config_has_expected_defaults(self) -> None:
        cfg = qe.PairsTradingStrategy.Config()
        assert cfg.entry_zscore == 2.0
        assert cfg.exit_zscore == 0.5
        assert cfg.stop_loss_zscore == 4.0
        assert cfg.zscore_lookback == 60

    def test_cpp_name_matches_python_wrapper(self) -> None:
        """
        Drift guard: C++ ``name()`` must equal the Python wrapper's name.
        """

        from src.strategies.pairs_trading import PairsTradingStrategy

        cpp = qe.PairsTradingStrategy(qe.PairsTradingStrategy.Config())
        py = PairsTradingStrategy()
        assert cpp.name == py.name

    def test_generate_signals_matches_composed_primitives(self) -> None:
        rng_a = np.random.default_rng(LONG_SERIES_SEED_A)
        rng_b = np.random.default_rng(LONG_SERIES_SEED_B)
        b = 100.0 + np.cumsum(rng_b.normal(0.0, 0.5, LONG_SERIES_LEN))
        a = b * HEDGE_RATIO + rng_a.normal(0.0, 2.0, LONG_SERIES_LEN)

        cfg = qe.PairsTradingStrategy.Config(
            entry_zscore=2.0, exit_zscore=0.5, stop_loss_zscore=4.0, zscore_lookback=ZSCORE_WINDOW
        )
        coint = qe.CointegrationParams(hedge_ratio=HEDGE_RATIO, spread_mean=0.0, spread_std=2.0)
        strategy_out = qe.PairsTradingStrategy(cfg).generate_signals(
            prices_a=a, prices_b=b, coint=coint
        )

        spread = qe.SpreadCalculator.compute_spread(a=a, b=b, hedge_ratio=HEDGE_RATIO)
        zscore = qe.SpreadCalculator.compute_zscore(spread=spread, window=ZSCORE_WINDOW)
        manual_out = qe.run_pairs_state_machine(
            zscore=zscore,
            entry_zscore=cfg.entry_zscore,
            exit_zscore=cfg.exit_zscore,
            stop_loss_zscore=cfg.stop_loss_zscore,
        )
        np.testing.assert_array_equal(strategy_out, manual_out)


class TestAdaptiveBollingerStrategyBinding:
    def test_config_has_expected_defaults(self) -> None:
        cfg = qe.AdaptiveBollingerStrategy.Config()
        assert cfg.band_window == 20
        assert cfg.k == 2.0
        assert cfg.trend_window == 100

    def test_cpp_name_matches_python_wrapper(self) -> None:
        """
        Drift guard: C++ ``name()`` must equal the Python wrapper's name.
        """

        from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy

        cpp = qe.AdaptiveBollingerStrategy(qe.AdaptiveBollingerStrategy.Config())
        py = AdaptiveBollingerStrategy()
        assert cpp.name == py.name

    def test_generate_signals_matches_composed_primitives(self) -> None:
        rng = np.random.default_rng(LONG_SERIES_SEED_A)
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, LONG_SERIES_LEN))
        cond_vol = np.full(LONG_SERIES_LEN, 1.5, dtype=np.float64)

        cfg = qe.AdaptiveBollingerStrategy.Config(band_window=20, k=2.0, trend_window=50)
        strategy_out = qe.AdaptiveBollingerStrategy(cfg).generate_signals(
            close=close, cond_vol=cond_vol
        )

        close_series = pd.Series(close)
        mid = np.asarray(close_series.rolling(cfg.band_window).mean(), dtype=np.float64)
        trend_ma = np.asarray(close_series.rolling(cfg.trend_window).mean(), dtype=np.float64)
        upper = mid + cfg.k * cond_vol
        lower = mid - cfg.k * cond_vol
        manual_out = qe.run_mean_reversion_state_machine(
            close=close, mid=mid, upper=upper, lower=lower, trend_ma=trend_ma
        )
        np.testing.assert_array_equal(strategy_out, manual_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
