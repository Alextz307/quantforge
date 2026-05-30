"""
Tests for VolatilityTargetingStrategy (HybridVolatilityModel-backed).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.constants import TRADING_DAYS_PER_YEAR
from src.core.exceptions import LeakageError
from src.core.registry import strategy_registry
from src.core.types import Interval
from src.strategies.volatility_targeting import (
    VolatilityTargetingStrategy,
    _HybridVolParams,
)
from tests.conftest import (
    GLOBAL_NUMPY_SEED,
    GLOBAL_TORCH_SEED,
    assert_params_match_constructor,
    make_declining_ohlcv_df,
    make_synthetic_ohlcv_df,
)

RESCALE_RATIO_RTOL = 1e-12

TRAIN_ROW_COUNT = 200
FEATURE_RNG_SEED = 7

COMPACT_GARCH_P_MAX = 2
COMPACT_GARCH_Q_MAX = 2
COMPACT_LSTM_HIDDEN_DIM = 16
COMPACT_LSTM_NUM_LAYERS = 1
COMPACT_LSTM_LOOKBACK = 10
COMPACT_LSTM_EPOCHS = 2

TARGET_VOL = 0.15
MAX_LEVERAGE = 1.5
BEARISH_EXPOSURE = 0.0
TREND_WINDOW = 50
REALIZED_VOL_WINDOW = 20


@pytest.fixture
def train_df(synthetic_feature_columns: list[str]) -> pd.DataFrame:
    df = make_synthetic_ohlcv_df(n_rows=TRAIN_ROW_COUNT)
    rng = np.random.default_rng(FEATURE_RNG_SEED)
    for col in synthetic_feature_columns:
        df[col] = rng.normal(0, 1, len(df))
    return df


def _build_strategy(features: list[str]) -> VolatilityTargetingStrategy:
    torch.manual_seed(GLOBAL_TORCH_SEED)
    np.random.seed(GLOBAL_NUMPY_SEED)
    return VolatilityTargetingStrategy(
        feature_columns=features,
        target_vol=TARGET_VOL,
        max_leverage=MAX_LEVERAGE,
        bearish_exposure=BEARISH_EXPOSURE,
        trend_window=TREND_WINDOW,
        realized_vol_window=REALIZED_VOL_WINDOW,
        garch_p_max=COMPACT_GARCH_P_MAX,
        garch_q_max=COMPACT_GARCH_Q_MAX,
        lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
        lstm_num_layers=COMPACT_LSTM_NUM_LAYERS,
        lstm_lookback=COMPACT_LSTM_LOOKBACK,
        lstm_epochs=COMPACT_LSTM_EPOCHS,
    )


@pytest.fixture
def fitted_strategy(
    train_df: pd.DataFrame, synthetic_feature_columns: list[str]
) -> VolatilityTargetingStrategy:
    s = _build_strategy(synthetic_feature_columns)
    s.train(train_df)
    return s


class TestVolatilityTargetingStrategy:
    def test_empty_feature_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_columns"):
            VolatilityTargetingStrategy(feature_columns=[])

    def test_generate_signals_before_train_raises(
        self, train_df: pd.DataFrame, synthetic_feature_columns: list[str]
    ) -> None:
        s = _build_strategy(synthetic_feature_columns)
        with pytest.raises(RuntimeError, match="before train"):
            s.generate_signals(train_df)

    def test_train_generate_basic(
        self, fitted_strategy: VolatilityTargetingStrategy, train_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(train_df)
        assert isinstance(signals, pd.Series)
        assert signals.index.equals(train_df.index)
        assert signals.name == "vol_target_signal"

    def test_fold_diagnostics_empty_before_predict(
        self, fitted_strategy: VolatilityTargetingStrategy
    ) -> None:
        """
        No predict yet - the floor-bind metric is not reported.
        """

        assert dict(fitted_strategy.get_fold_diagnostics()) == {}

    def test_fold_diagnostics_after_predict_carries_floor_bind_fraction(
        self, fitted_strategy: VolatilityTargetingStrategy, train_df: pd.DataFrame
    ) -> None:
        _ = fitted_strategy.generate_signals(train_df)
        diagnostics = fitted_strategy.get_fold_diagnostics()
        assert "floor_bind_fraction" in diagnostics
        assert 0.0 <= diagnostics["floor_bind_fraction"] <= 1.0

    def test_leverage_clipped_to_bounds(
        self, fitted_strategy: VolatilityTargetingStrategy, train_df: pd.DataFrame
    ) -> None:
        signals = fitted_strategy.generate_signals(train_df).dropna()
        assert (signals >= 0.0).all()
        assert (signals <= MAX_LEVERAGE).all()

    def test_training_metadata_populated(
        self,
        fitted_strategy: VolatilityTargetingStrategy,
        train_df: pd.DataFrame,
        synthetic_feature_columns: list[str],
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(train_df)
        assert meta.interval == Interval.DAILY
        assert set(meta.feature_columns) == set(synthetic_feature_columns)

    def test_training_metadata_validates_overlap(
        self, fitted_strategy: VolatilityTargetingStrategy, train_df: pd.DataFrame
    ) -> None:
        meta = fitted_strategy.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(train_df)

    def test_registry_registration(self) -> None:
        assert "VolatilityTargeting" in strategy_registry

    def test_required_warmup_bars(self, synthetic_feature_columns: list[str]) -> None:
        s = _build_strategy(synthetic_feature_columns)
        assert s.required_warmup_bars == max(
            TREND_WINDOW, COMPACT_LSTM_LOOKBACK, REALIZED_VOL_WINDOW
        )

    def test_suggest_params_keys(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = VolatilityTargetingStrategy.suggest_params(trial)
        expected = {
            "target_vol",
            "trend_window",
            "max_leverage",
            "bearish_exposure",
            "realized_vol_window",
            "lstm_hidden_dim",
            "lstm_num_layers",
            "lstm_dropout",
            "lstm_lookback",
            "lstm_lr",
        }
        assert set(params.keys()) == expected

    def test_invalid_params_raise(self, synthetic_feature_columns: list[str]) -> None:
        with pytest.raises(ValueError, match="target_vol"):
            VolatilityTargetingStrategy(feature_columns=synthetic_feature_columns, target_vol=0.0)
        with pytest.raises(ValueError, match="max_leverage"):
            VolatilityTargetingStrategy(feature_columns=synthetic_feature_columns, max_leverage=0.0)

    def test_hybrid_params_match_constructor(self) -> None:
        from src.models.hybrid_volatility import HybridVolatilityModel

        assert_params_match_constructor(_HybridVolParams, HybridVolatilityModel)

    def test_deterministic_signals(
        self, fitted_strategy: VolatilityTargetingStrategy, train_df: pd.DataFrame
    ) -> None:
        s1 = fitted_strategy.generate_signals(train_df)
        s2 = fitted_strategy.generate_signals(train_df)
        pd.testing.assert_series_equal(s1, s2)

    def test_hourly_interval_rescales_realized_vol(
        self, train_df: pd.DataFrame, synthetic_feature_columns: list[str]
    ) -> None:
        """
        HOUR interval multiplies the C++ GK output by sqrt(ann_factor / 252).
        """

        daily = VolatilityTargetingStrategy(
            feature_columns=synthetic_feature_columns,
            realized_vol_window=REALIZED_VOL_WINDOW,
        )
        hourly = VolatilityTargetingStrategy(
            feature_columns=synthetic_feature_columns,
            realized_vol_window=REALIZED_VOL_WINDOW,
            interval=Interval.HOUR,
        )
        rv_daily = daily._compute_realized_vol(train_df).dropna()
        rv_hourly = hourly._compute_realized_vol(train_df).dropna()
        expected_ratio = math.sqrt(Interval.HOUR.annualization_factor() / TRADING_DAYS_PER_YEAR)
        np.testing.assert_allclose(
            rv_hourly.to_numpy(),
            rv_daily.to_numpy() * expected_ratio,
            rtol=RESCALE_RATIO_RTOL,
        )

    def test_bearish_regime_reduces_exposure(
        self,
        fitted_strategy: VolatilityTargetingStrategy,
        synthetic_feature_columns: list[str],
    ) -> None:
        """
        In a declining-trend window, leverage collapses to 0 (bearish_exposure=0).
        """

        df = make_declining_ohlcv_df()
        rng = np.random.default_rng(FEATURE_RNG_SEED)
        for col in synthetic_feature_columns:
            df[col] = rng.normal(0, 1, len(df))
        signals = fitted_strategy.generate_signals(df).dropna()
        # Guard against a vacuous pass: an all-NaN signal makes (empty == 0.0).all() True.
        assert not signals.empty
        assert (signals == 0.0).all()
