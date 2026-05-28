"""
Tests for ReturnForecastStrategy (HybridReturnModel-backed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.exceptions import LeakageError
from src.core.registry import strategy_registry
from src.core.types import Interval
from src.strategies.return_forecast import (
    ReturnForecastStrategy,
    _HybridReturnParams,
)
from tests.conftest import (
    GLOBAL_NUMPY_SEED,
    GLOBAL_TORCH_SEED,
    assert_params_match_constructor,
    make_synthetic_close_df,
)

TRAIN_ROW_COUNT = 200
FEATURE_RNG_SEED = 13

COMPACT_ARMA_P_MAX = 2
COMPACT_ARMA_Q_MAX = 2
COMPACT_LSTM_HIDDEN_DIM = 16
COMPACT_LSTM_NUM_LAYERS = 1
COMPACT_LSTM_LOOKBACK = 10
COMPACT_LSTM_EPOCHS = 2

POSITION_SCALE = 20.0
MAX_LEVERAGE = 1.5


@pytest.fixture
def train_df(synthetic_feature_columns: list[str]) -> pd.DataFrame:
    df = make_synthetic_close_df(n_rows=TRAIN_ROW_COUNT)
    rng = np.random.default_rng(FEATURE_RNG_SEED)
    for col in synthetic_feature_columns:
        df[col] = rng.normal(0, 1, len(df))
    return df


def _build_strategy(features: list[str]) -> ReturnForecastStrategy:
    torch.manual_seed(GLOBAL_TORCH_SEED)
    np.random.seed(GLOBAL_NUMPY_SEED)
    return ReturnForecastStrategy(
        feature_columns=features,
        position_scale=POSITION_SCALE,
        max_leverage=MAX_LEVERAGE,
        arma_p_max=COMPACT_ARMA_P_MAX,
        arma_q_max=COMPACT_ARMA_Q_MAX,
        lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
        lstm_num_layers=COMPACT_LSTM_NUM_LAYERS,
        lstm_lookback=COMPACT_LSTM_LOOKBACK,
        lstm_epochs=COMPACT_LSTM_EPOCHS,
    )


@pytest.fixture
def fitted_strategy(
    train_df: pd.DataFrame, synthetic_feature_columns: list[str]
) -> ReturnForecastStrategy:
    s = _build_strategy(synthetic_feature_columns)
    s.train(train_df)
    return s


class TestReturnForecastStrategy:
    def test_empty_feature_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_columns"):
            ReturnForecastStrategy(feature_columns=[])

    def test_generate_signals_before_train_raises(
        self, train_df: pd.DataFrame, synthetic_feature_columns: list[str]
    ) -> None:
        s = _build_strategy(synthetic_feature_columns)
        with pytest.raises(RuntimeError, match="before train"):
            s.generate_signals(train_df)

    def test_train_generate_basic(
        self, fitted_strategy: ReturnForecastStrategy, train_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(train_df)
        assert isinstance(signals, pd.Series)
        assert signals.index.equals(train_df.index)
        assert signals.name == "return_forecast_signal"

    def test_position_clipped_to_bounds(
        self, fitted_strategy: ReturnForecastStrategy, train_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(train_df).dropna()
        assert (signals >= -MAX_LEVERAGE).all()
        assert (signals <= MAX_LEVERAGE).all()

    def test_sign_matches_forecast(
        self, fitted_strategy: ReturnForecastStrategy, train_df: pd.DataFrame
    ) -> None:
        """
        Position sign must agree with the underlying return-forecast sign.
        """

        signals = fitted_strategy.generate_signals(train_df).dropna()
        forecast = fitted_strategy._hybrid_return.predict(train_df).dropna()
        aligned = signals.align(forecast, join="inner")
        signals_aligned, forecast_aligned = aligned
        # Raw position is position_scale * forecast (signs match); clipping
        # at +/- max_leverage also preserves sign.
        non_zero = signals_aligned[signals_aligned != 0.0]
        f_aligned = forecast_aligned.loc[non_zero.index]
        non_zero_fc = f_aligned[f_aligned != 0.0]
        assert (np.sign(non_zero.loc[non_zero_fc.index]) == np.sign(non_zero_fc)).all()

    def test_training_metadata_populated(
        self,
        fitted_strategy: ReturnForecastStrategy,
        train_df: pd.DataFrame,
        synthetic_feature_columns: list[str],
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(train_df)
        assert meta.interval == Interval.DAILY
        assert set(meta.feature_columns) == set(synthetic_feature_columns)

    def test_training_metadata_validates_overlap(
        self, fitted_strategy: ReturnForecastStrategy, train_df: pd.DataFrame
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(train_df)

    def test_registry_registration(self) -> None:
        assert "ReturnForecast" in strategy_registry

    def test_required_warmup_bars(self, synthetic_feature_columns: list[str]) -> None:
        s = _build_strategy(synthetic_feature_columns)
        assert s.required_warmup_bars == COMPACT_LSTM_LOOKBACK

    def test_suggest_params_keys(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = ReturnForecastStrategy.suggest_params(trial)
        expected = {
            "position_scale",
            "max_leverage",
            "arma_p_max",
            "arma_q_max",
            "lstm_hidden_dim",
            "lstm_num_layers",
            "lstm_dropout",
            "lstm_lookback",
            "lstm_lr",
        }
        assert set(params.keys()) == expected

    def test_deterministic_signals(
        self, fitted_strategy: ReturnForecastStrategy, train_df: pd.DataFrame
    ) -> None:
        s1 = fitted_strategy.generate_signals(train_df)
        s2 = fitted_strategy.generate_signals(train_df)
        pd.testing.assert_series_equal(s1, s2)

    def test_invalid_params_raise(self, synthetic_feature_columns: list[str]) -> None:
        with pytest.raises(ValueError, match="position_scale"):
            ReturnForecastStrategy(feature_columns=synthetic_feature_columns, position_scale=0.0)
        with pytest.raises(ValueError, match="max_leverage"):
            ReturnForecastStrategy(feature_columns=synthetic_feature_columns, max_leverage=0.0)

    def test_hybrid_params_match_constructor(self) -> None:
        from src.models.hybrid_return import HybridReturnModel

        assert_params_match_constructor(_HybridReturnParams, HybridReturnModel)
