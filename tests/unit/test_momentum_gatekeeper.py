"""Tests for MomentumGatekeeperStrategy (FeaturePipeline + XGBoost + trend gate)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.registry import strategy_registry
from src.core.types import Interval
from src.strategies.momentum_gatekeeper import MomentumGatekeeperStrategy
from tests.conftest import make_declining_close_df, make_synthetic_close_df

# Compact XGBoost params for fast CI
COMPACT_N_ESTIMATORS = 20
COMPACT_MAX_DEPTH = 3

# Strategy defaults for tests
MA_WINDOW = 20
PROB_THRESHOLD = 0.55
FEATURE_PIPELINE_WARMUP = 33  # MACD signal (slow=26 + signal=9 - 2) dominates hard-NaN horizons

# Out-of-sample eval fixture
EVAL_ROW_COUNT = 80
EVAL_START_DATE = "2021-01-04"
EVAL_SEED = 99

# Valid binary signal values
VALID_SIGNALS = {0.0, 1.0}


@pytest.fixture
def train_df() -> pd.DataFrame:
    return make_synthetic_close_df()


@pytest.fixture
def eval_df() -> pd.DataFrame:
    return make_synthetic_close_df(n_rows=EVAL_ROW_COUNT, start=EVAL_START_DATE, seed=EVAL_SEED)


@pytest.fixture
def fitted_strategy(train_df: pd.DataFrame) -> MomentumGatekeeperStrategy:
    s = MomentumGatekeeperStrategy(
        ma_window=MA_WINDOW,
        prob_threshold=PROB_THRESHOLD,
        n_estimators=COMPACT_N_ESTIMATORS,
        max_depth=COMPACT_MAX_DEPTH,
    )
    s.train(train_df)
    return s


class TestMomentumGatekeeperStrategy:
    def test_generate_signals_before_train_raises(self, train_df: pd.DataFrame) -> None:
        s = MomentumGatekeeperStrategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.generate_signals(train_df)

    def test_train_generate_basic(
        self, fitted_strategy: MomentumGatekeeperStrategy, eval_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(eval_df)
        assert isinstance(signals, pd.Series)
        assert signals.index.equals(eval_df.index)
        assert signals.name == "momentum_gatekeeper_signal"

    def test_signals_in_binary_set(
        self, fitted_strategy: MomentumGatekeeperStrategy, train_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(train_df)
        non_nan = signals.dropna()
        assert set(non_nan.unique()).issubset(VALID_SIGNALS)

    def test_leading_warmup_is_nan(
        self, fitted_strategy: MomentumGatekeeperStrategy, train_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(train_df)
        # MACD signal line warmup dominates FeaturePipeline hard-NaN horizon
        assert signals.iloc[:FEATURE_PIPELINE_WARMUP].isna().all()

    def test_training_metadata_populated(
        self, fitted_strategy: MomentumGatekeeperStrategy, train_df: pd.DataFrame
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(train_df)
        assert meta.interval == Interval.DAILY
        # resolved features should include the standard FeaturePipeline outputs
        assert "return_1d" in meta.feature_columns
        assert "macd" in meta.feature_columns

    def test_training_metadata_validates_overlap(
        self, fitted_strategy: MomentumGatekeeperStrategy, train_df: pd.DataFrame
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(train_df)

    def test_registry_registration(self) -> None:
        assert "MomentumGatekeeper" in strategy_registry

    def test_required_warmup_bars(self) -> None:
        s = MomentumGatekeeperStrategy(ma_window=MA_WINDOW)
        assert s.required_warmup_bars == max(MA_WINDOW, FEATURE_PIPELINE_WARMUP)

    def test_suggest_params_keys(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = MomentumGatekeeperStrategy.suggest_params(trial)
        assert set(params.keys()) == {
            "ma_window",
            "prob_threshold",
            "rsi_period",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "vol_window",
            "ma_ratio_window",
            "short_return_period",
            "long_return_period",
            "n_estimators",
            "learning_rate",
            "max_depth",
        }

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="prob_threshold"):
            MomentumGatekeeperStrategy(prob_threshold=1.5)
        with pytest.raises(ValueError, match="prob_threshold"):
            MomentumGatekeeperStrategy(prob_threshold=0.0)

    def test_invalid_ma_window_raises(self) -> None:
        with pytest.raises(ValueError, match="ma_window"):
            MomentumGatekeeperStrategy(ma_window=1)

    def test_trend_gate_zeroes_bearish_regime(
        self, fitted_strategy: MomentumGatekeeperStrategy
    ) -> None:
        """In a monotone decline, the trend gate must suppress all long signals."""
        df = make_declining_close_df()
        signals = fitted_strategy.generate_signals(df)
        non_nan = signals.dropna()
        assert (non_nan == 0.0).all()

    def test_explicit_feature_subset_honored(self, train_df: pd.DataFrame) -> None:
        """Configured feature_columns subset is passed to DirectionalClassifier."""
        subset = ["return_1d", "vol_20", "rsi_14"]
        s = MomentumGatekeeperStrategy(
            feature_columns=subset,
            n_estimators=COMPACT_N_ESTIMATORS,
            max_depth=COMPACT_MAX_DEPTH,
        )
        s.train(train_df)
        meta = s.training_metadata
        assert meta is not None
        assert set(meta.feature_columns) == set(subset)

    def test_deterministic_signals(
        self, fitted_strategy: MomentumGatekeeperStrategy, train_df: pd.DataFrame
    ) -> None:
        s1 = fitted_strategy.generate_signals(train_df)
        s2 = fitted_strategy.generate_signals(train_df)
        pd.testing.assert_series_equal(s1, s2)

    def test_invalid_feature_subset_raises(self, train_df: pd.DataFrame) -> None:
        s = MomentumGatekeeperStrategy(
            feature_columns=["does_not_exist"],
            n_estimators=COMPACT_N_ESTIMATORS,
        )
        with pytest.raises(ValueError, match="not produced by pipeline"):
            s.train(train_df)
