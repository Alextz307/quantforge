"""
Tests for AdaptiveBollingerStrategy (GARCH-adaptive Bollinger bands).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.registry import strategy_registry
from src.core.types import Interval
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_declining_close_df, make_synthetic_close_df

COMPACT_GARCH_P_MAX = 2
COMPACT_GARCH_Q_MAX = 2

BOLLINGER_WINDOW = 20
BOLLINGER_TREND_WINDOW = 50
BOLLINGER_K = 2.0

EVAL_ROW_COUNT = 60
EVAL_START_DATE = "2021-01-04"
EVAL_SEED = 99

VALID_SIGNALS = {-1.0, 0.0, 1.0}


@pytest.fixture
def train_df() -> pd.DataFrame:
    return make_synthetic_close_df()


@pytest.fixture
def eval_df() -> pd.DataFrame:
    return make_synthetic_close_df(n_rows=EVAL_ROW_COUNT, start=EVAL_START_DATE, seed=EVAL_SEED)


@pytest.fixture
def fitted_strategy(train_df: pd.DataFrame) -> AdaptiveBollingerStrategy:
    s = AdaptiveBollingerStrategy(
        window=BOLLINGER_WINDOW,
        k=BOLLINGER_K,
        trend_window=BOLLINGER_TREND_WINDOW,
        garch_p_max=COMPACT_GARCH_P_MAX,
        garch_q_max=COMPACT_GARCH_Q_MAX,
    )
    s.train(train_df)
    return s


class TestAdaptiveBollingerStrategy:
    def test_generate_signals_before_train_raises(self, train_df: pd.DataFrame) -> None:
        s = AdaptiveBollingerStrategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.generate_signals(train_df)

    def test_train_generate_basic(
        self, fitted_strategy: AdaptiveBollingerStrategy, eval_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(eval_df)
        assert isinstance(signals, pd.Series)
        assert signals.index.equals(eval_df.index)
        assert signals.name == "adaptive_bollinger_signal"
        assert len(signals) == len(eval_df)

    def test_signals_in_discrete_set(
        self, fitted_strategy: AdaptiveBollingerStrategy, train_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(train_df)
        non_nan = signals.dropna()
        assert set(non_nan.unique()).issubset(VALID_SIGNALS)

    def test_leading_warmup_is_nan(
        self, fitted_strategy: AdaptiveBollingerStrategy, train_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(train_df)
        warmup = max(BOLLINGER_WINDOW, BOLLINGER_TREND_WINDOW)
        assert signals.iloc[: warmup - 1].isna().all()

    def test_training_metadata_populated(
        self, fitted_strategy: AdaptiveBollingerStrategy, train_df: pd.DataFrame
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(train_df)
        assert meta.interval == Interval.DAILY
        assert meta.feature_columns == ("close",)

    def test_training_metadata_validates_overlap(
        self, fitted_strategy: AdaptiveBollingerStrategy, train_df: pd.DataFrame
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(train_df)

    def test_registry_registration(self) -> None:
        assert "AdaptiveBollinger" in strategy_registry

    def test_required_warmup_bars(self) -> None:
        s = AdaptiveBollingerStrategy(window=BOLLINGER_WINDOW, trend_window=BOLLINGER_TREND_WINDOW)
        assert s.required_warmup_bars == max(BOLLINGER_WINDOW, BOLLINGER_TREND_WINDOW)

    def test_suggest_params_keys(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = AdaptiveBollingerStrategy.suggest_params(trial)
        assert set(params.keys()) == {
            "window",
            "k",
            "trend_window",
            "garch_p_max",
            "garch_q_max",
        }

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            AdaptiveBollingerStrategy(window=1)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k"):
            AdaptiveBollingerStrategy(k=0.0)

    def test_deterministic_signals(
        self, fitted_strategy: AdaptiveBollingerStrategy, train_df: pd.DataFrame
    ) -> None:
        s1 = fitted_strategy.generate_signals(train_df)
        s2 = fitted_strategy.generate_signals(train_df)
        pd.testing.assert_series_equal(s1, s2)

    def test_bearish_regime_shorts_only(self, fitted_strategy: AdaptiveBollingerStrategy) -> None:
        """
        In a declining-trend window, non-zero signals must be short (-1), never long (+1).
        """

        df = make_declining_close_df()
        signals = fitted_strategy.generate_signals(df)
        non_nan = signals.dropna()
        assert (non_nan != 1.0).all()
