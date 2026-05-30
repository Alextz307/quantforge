"""
Tests for HybridReturnModel (ARMA + LSTM residual).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.exceptions import LeakageError
from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.models.hybrid_return import HybridReturnModel
from tests.conftest import (
    GLOBAL_NUMPY_SEED,
    GLOBAL_TORCH_SEED,
    HOURLY_BASE_PRICE,
    HOURLY_RETURN_STD,
    HOURLY_ROW_COUNT,
    HOURLY_START,
    make_synthetic_close_df,
)

SYNTH_ROW_COUNT = 200
# Different from test_hybrid_volatility.py so the two suites see different data.
FEATURE_RNG_SEED = 11

COMPACT_ARMA_P_MAX = 2
COMPACT_ARMA_Q_MAX = 2
COMPACT_LSTM_HIDDEN_DIM = 16
COMPACT_LSTM_NUM_LAYERS = 1
COMPACT_LSTM_LOOKBACK = 10
COMPACT_LSTM_EPOCHS = 3
LSTM_EPOCHS_FOR_RESIDUAL_DIVERGENCE = 10
HOURLY_LSTM_EPOCHS = 2

RESIDUAL_DIVERGENCE_ATOL = 1e-12


@pytest.fixture
def hybrid_train_df(synthetic_feature_columns: list[str]) -> pd.DataFrame:
    df = make_synthetic_close_df(n_rows=SYNTH_ROW_COUNT)
    rng = np.random.default_rng(FEATURE_RNG_SEED)
    for col in synthetic_feature_columns:
        df[col] = rng.normal(0, 1, len(df))
    return df


@pytest.fixture
def log_return_target(hybrid_train_df: pd.DataFrame) -> pd.Series:
    """
    Log returns target - leading NaN dropped internally by HybridReturnModel.
    """

    return compute_log_returns(hybrid_train_df["close"])


def _fit_model(
    df: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    *,
    epochs: int = COMPACT_LSTM_EPOCHS,
    lookback: int = COMPACT_LSTM_LOOKBACK,
) -> HybridReturnModel:
    torch.manual_seed(GLOBAL_TORCH_SEED)
    np.random.seed(GLOBAL_NUMPY_SEED)
    model = HybridReturnModel(
        feature_columns=features,
        arma_p_max=COMPACT_ARMA_P_MAX,
        arma_q_max=COMPACT_ARMA_Q_MAX,
        lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
        lstm_num_layers=COMPACT_LSTM_NUM_LAYERS,
        lstm_lookback=lookback,
        lstm_epochs=epochs,
    )
    model.fit(df, target)
    return model


class TestHybridReturnModel:
    def test_predict_before_fit_raises(
        self, hybrid_train_df: pd.DataFrame, synthetic_feature_columns: list[str]
    ) -> None:
        m = HybridReturnModel(feature_columns=synthetic_feature_columns)
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(hybrid_train_df)

    def test_fit_missing_feature_column_raises(
        self, hybrid_train_df: pd.DataFrame, log_return_target: pd.Series
    ) -> None:
        """
        A configured feature absent from train_data (e.g. after a feature
        period override renames roc_63 -> roc_<period>) fails with a clear
        ValueError naming the column, not an opaque downstream KeyError.
        """

        model = HybridReturnModel(feature_columns=["absent_feature"])
        with pytest.raises(ValueError, match="absent_feature"):
            model.fit(hybrid_train_df, log_return_target)

    def test_empty_feature_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_columns"):
            HybridReturnModel(feature_columns=[])

    def test_fit_predict_basic(
        self,
        hybrid_train_df: pd.DataFrame,
        log_return_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, log_return_target, synthetic_feature_columns)
        result = model.predict(hybrid_train_df)
        assert isinstance(result, pd.Series)
        assert result.name == "hybrid_return"
        assert result.index.equals(hybrid_train_df.index)

    def test_fit_drops_nan_warmup_feature_rows(
        self,
        hybrid_train_df: pd.DataFrame,
        log_return_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        """
        Leading NaN feature rows (real-world: rsi_14 / return_21d warmup
        on a short-warmup target like log returns) must be dropped before the
        scaler + LSTM see them. Otherwise sklearn's StandardScaler propagates
        NaN through ``transform`` and the LSTM's first forward pass produces
        NaN gradients that corrupt every subsequent epoch."""

        df = hybrid_train_df.copy()
        first_feature = synthetic_feature_columns[0]
        nan_warmup_rows = 15
        df.loc[df.index[:nan_warmup_rows], first_feature] = np.nan

        model = _fit_model(df, log_return_target, synthetic_feature_columns)

        # Finite scaler stats prove NaN didn't reach .fit_transform().
        assert model._scaler is not None
        assert np.isfinite(model._scaler.mean_).all()
        assert np.isfinite(model._scaler.scale_).all()
        result = model.predict(hybrid_train_df)
        assert result.index.equals(hybrid_train_df.index)

    def test_scaler_refit_raises_leakage(
        self,
        hybrid_train_df: pd.DataFrame,
        log_return_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, log_return_target, synthetic_feature_columns)
        with pytest.raises(LeakageError):
            model.fit(hybrid_train_df, log_return_target)

    def test_arma_order_frozen_after_fit(
        self,
        hybrid_train_df: pd.DataFrame,
        log_return_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, log_return_target, synthetic_feature_columns)
        assert model._arma._model is not None
        order_before = model._arma._model.order
        model.predict(hybrid_train_df)
        assert model._arma._model is not None
        assert model._arma._model.order == order_before

    def test_training_metadata_populated(
        self,
        hybrid_train_df: pd.DataFrame,
        log_return_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, log_return_target, synthetic_feature_columns)
        meta = model.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(hybrid_train_df)
        assert meta.interval == Interval.DAILY
        assert tuple(meta.feature_columns) == tuple(synthetic_feature_columns)

    def test_training_metadata_validates_overlap(
        self,
        hybrid_train_df: pd.DataFrame,
        log_return_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, log_return_target, synthetic_feature_columns)
        meta = model.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(hybrid_train_df)

    def test_residual_correction_changes_output(
        self,
        hybrid_train_df: pd.DataFrame,
        log_return_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(
            hybrid_train_df,
            log_return_target,
            synthetic_feature_columns,
            epochs=LSTM_EPOCHS_FOR_RESIDUAL_DIVERGENCE,
            lookback=COMPACT_LSTM_LOOKBACK,
        )
        hybrid_out = model.predict(hybrid_train_df).dropna()
        arma_only = model._arma.predict(hybrid_train_df).loc[hybrid_out.index]
        assert not np.allclose(
            hybrid_out.to_numpy(), arma_only.to_numpy(), atol=RESIDUAL_DIVERGENCE_ATOL
        )

    def test_registry_registration(self) -> None:
        from src.core.registry import model_registry

        assert "hybrid_return" in model_registry

    def test_suggest_params_returns_combined_space(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = HybridReturnModel.suggest_params(trial)
        assert "arma_p_max" in params
        assert "arma_q_max" in params
        assert "arma_information_criterion" in params
        assert "lstm_hidden_dim" in params
        assert "lstm_lookback" in params

    def test_hourly_interval(self, synthetic_feature_columns: list[str]) -> None:
        np.random.seed(GLOBAL_NUMPY_SEED)
        idx = pd.date_range(start=HOURLY_START, periods=HOURLY_ROW_COUNT, freq="h")
        returns = np.random.normal(0, HOURLY_RETURN_STD, HOURLY_ROW_COUNT)
        close = HOURLY_BASE_PRICE * np.cumprod(1 + returns)
        df = pd.DataFrame({"close": close}, index=idx)
        rng = np.random.default_rng(FEATURE_RNG_SEED)
        for col in synthetic_feature_columns:
            df[col] = rng.normal(0, 1, HOURLY_ROW_COUNT)

        target = compute_log_returns(df["close"])

        torch.manual_seed(GLOBAL_TORCH_SEED)
        model = HybridReturnModel(
            feature_columns=synthetic_feature_columns,
            arma_p_max=COMPACT_ARMA_P_MAX,
            arma_q_max=COMPACT_ARMA_Q_MAX,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_num_layers=COMPACT_LSTM_NUM_LAYERS,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=HOURLY_LSTM_EPOCHS,
            interval=Interval.HOUR,
        )
        model.fit(df, target)
        meta = model.training_metadata
        assert meta is not None
        assert meta.interval == Interval.HOUR
