"""Tests for PairsTradingStrategy (cointegration + z-score mean reversion)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.registry import strategy_registry
from src.core.types import Interval
from src.strategies.pairs_trading import PairsTradingStrategy
from tests.conftest import (
    GLOBAL_NUMPY_SEED,
    SYNTH_RETURN_MEAN,
    SYNTH_RETURN_STD,
    make_pair_close_df,
)

# Strategy defaults tuned for short synthetic fixtures
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
LOOKBACK = 30

# Non-cointegrated pair fixture — two independent random walks
RW_ROW_COUNT = 300
RW_START_DATE = "2020-01-02"
RW_SEED_A = 11
RW_SEED_B = 17
RW_BASE_PRICE_A = 100.0
RW_BASE_PRICE_B = 100.0

# Valid discrete signal values
VALID_SIGNALS = {-1.0, 0.0, 1.0}


@pytest.fixture
def pair_df() -> pd.DataFrame:
    return make_pair_close_df()


@pytest.fixture
def fitted_strategy(pair_df: pd.DataFrame) -> PairsTradingStrategy:
    s = PairsTradingStrategy(
        entry_zscore=ENTRY_Z,
        exit_zscore=EXIT_Z,
        stop_loss_zscore=STOP_Z,
        zscore_lookback=LOOKBACK,
    )
    s.train(pair_df)
    return s


def _make_independent_random_walks() -> pd.DataFrame:
    """Two independent random walks — not cointegrated."""
    idx = pd.bdate_range(start=RW_START_DATE, periods=RW_ROW_COUNT, freq="B")
    rng_a = np.random.default_rng(RW_SEED_A)
    rng_b = np.random.default_rng(RW_SEED_B)
    returns_a = rng_a.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, RW_ROW_COUNT)
    returns_b = rng_b.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, RW_ROW_COUNT)
    close_a = RW_BASE_PRICE_A * np.cumprod(1 + returns_a)
    close_b = RW_BASE_PRICE_B * np.cumprod(1 + returns_b)
    return pd.DataFrame({"close_a": close_a, "close_b": close_b}, index=idx)


class TestPairsTradingStrategy:
    def test_generate_signals_before_train_raises(self, pair_df: pd.DataFrame) -> None:
        s = PairsTradingStrategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.generate_signals(pair_df)

    def test_hedge_ratio_before_train_raises(self) -> None:
        s = PairsTradingStrategy()
        with pytest.raises(RuntimeError, match="before train"):
            _ = s.hedge_ratio

    def test_train_generate_basic(
        self, fitted_strategy: PairsTradingStrategy, pair_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(pair_df)
        assert isinstance(signals, pd.Series)
        assert signals.index.equals(pair_df.index)
        assert signals.name == "pairs_signal"

    def test_signals_in_discrete_set(
        self, fitted_strategy: PairsTradingStrategy, pair_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(pair_df)
        non_nan = signals.dropna()
        assert set(non_nan.unique()).issubset(VALID_SIGNALS)

    def test_leading_lookback_is_nan(
        self, fitted_strategy: PairsTradingStrategy, pair_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(pair_df)
        # The rolling std needs LOOKBACK bars, so positions[0 : LOOKBACK - 1] are NaN
        assert signals.iloc[: LOOKBACK - 1].isna().all()

    def test_hedge_ratio_exposed_after_train(self, fitted_strategy: PairsTradingStrategy) -> None:
        hr = fitted_strategy.hedge_ratio
        assert isinstance(hr, float)
        assert hr != 0.0

    def test_non_cointegrated_raises(self) -> None:
        df = _make_independent_random_walks()
        s = PairsTradingStrategy(zscore_lookback=LOOKBACK)
        with pytest.raises(ValueError, match="not cointegrated"):
            s.train(df)

    def test_missing_columns_raises(self, pair_df: pd.DataFrame) -> None:
        s = PairsTradingStrategy()
        bad = pair_df.rename(columns={"close_a": "wrong_name"})
        with pytest.raises(ValueError, match="close_a"):
            s.train(bad)

    def test_training_metadata_populated(
        self, fitted_strategy: PairsTradingStrategy, pair_df: pd.DataFrame
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(pair_df)
        assert meta.interval == Interval.DAILY
        assert meta.feature_columns == ("close_a", "close_b")

    def test_training_metadata_validates_overlap(
        self, fitted_strategy: PairsTradingStrategy, pair_df: pd.DataFrame
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(pair_df)

    def test_registry_registration(self) -> None:
        assert "PairsTrading" in strategy_registry

    def test_required_warmup_bars(self) -> None:
        s = PairsTradingStrategy(zscore_lookback=LOOKBACK)
        assert s.required_warmup_bars == LOOKBACK

    def test_suggest_params_keys(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = PairsTradingStrategy.suggest_params(trial)
        assert set(params.keys()) == {
            "entry_zscore",
            "exit_zscore",
            "stop_loss_zscore",
            "zscore_lookback",
        }

    def test_invalid_zscore_ordering_raises(self) -> None:
        with pytest.raises(ValueError, match="exit_zscore"):
            PairsTradingStrategy(entry_zscore=2.0, exit_zscore=2.5)
        with pytest.raises(ValueError, match="stop_loss_zscore"):
            PairsTradingStrategy(entry_zscore=2.0, stop_loss_zscore=1.5)

    def test_deterministic_signals(
        self, fitted_strategy: PairsTradingStrategy, pair_df: pd.DataFrame
    ) -> None:
        """Signals are a pure function of data once trained."""
        np.random.seed(GLOBAL_NUMPY_SEED)
        s1 = fitted_strategy.generate_signals(pair_df)
        s2 = fitted_strategy.generate_signals(pair_df)
        pd.testing.assert_series_equal(s1, s2)

    def test_update_extends_training_window(self, pair_df: pd.DataFrame) -> None:
        """``update()`` advances ``train_end`` + ``n_train_samples`` on the
        combined window while preserving ``train_start`` and
        ``fit_timestamp``."""
        # First half = initial train; second half = delta passed to update().
        # The halves must be disjoint — ``extend()`` raises if ``new_end`` is
        # not strictly after ``train_end``.
        split = len(pair_df) // 2
        train = pair_df.iloc[:split]
        new = pair_df.iloc[split:]

        s = PairsTradingStrategy(
            entry_zscore=ENTRY_Z,
            exit_zscore=EXIT_Z,
            stop_loss_zscore=STOP_Z,
            zscore_lookback=LOOKBACK,
        )
        s.train(train)
        first_meta = s.training_metadata
        assert first_meta is not None
        assert first_meta.n_train_samples == len(train)

        s.update(new)
        second_meta = s.training_metadata
        assert second_meta is not None
        assert second_meta.n_train_samples == len(train) + len(new)
        assert second_meta.train_start == first_meta.train_start
        assert second_meta.train_end == pd.Timestamp(new.index[-1])
        assert second_meta.fit_timestamp == first_meta.fit_timestamp
        signals = s.generate_signals(new)
        assert isinstance(signals, pd.Series)
