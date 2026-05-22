"""Tests for HybridVolatilityModel (GARCH + LSTM residual)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.exceptions import LeakageError
from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.models.hybrid_volatility import HybridVolatilityModel
from tests.conftest import (
    GLOBAL_NUMPY_SEED,
    GLOBAL_TORCH_SEED,
    HOURLY_BASE_PRICE,
    HOURLY_RETURN_STD,
    HOURLY_ROW_COUNT,
    HOURLY_START,
    make_synthetic_close_df,
)

# Synthetic data
SYNTH_ROW_COUNT = 200
FEATURE_RNG_SEED = 7  # arbitrary; differs from test_hybrid_return.py to vary fixture data
REALIZED_VOL_WINDOW = 20  # rolling window for synthetic realized-vol target

# Compact-model parameters (small for fast CI; not realistic for production)
COMPACT_GARCH_P_MAX = 2
COMPACT_GARCH_Q_MAX = 2
COMPACT_LSTM_HIDDEN_DIM = 16
COMPACT_LSTM_NUM_LAYERS = 1
COMPACT_LSTM_LOOKBACK = 10
COMPACT_LSTM_EPOCHS = 3
LSTM_EPOCHS_FOR_RESIDUAL_DIVERGENCE = 10  # higher epoch count to make LSTM signal visible
HOURLY_LSTM_EPOCHS = 2

# Numerical
RESIDUAL_DIVERGENCE_ATOL = 1e-12  # tightness for "hybrid != GARCH-only" assertion


@pytest.fixture
def hybrid_train_df(synthetic_feature_columns: list[str]) -> pd.DataFrame:
    """Close + two synthetic feature columns."""
    df = make_synthetic_close_df(n_rows=SYNTH_ROW_COUNT)
    rng = np.random.default_rng(FEATURE_RNG_SEED)
    for col in synthetic_feature_columns:
        df[col] = rng.normal(0, 1, len(df))
    return df


@pytest.fixture
def realized_vol_target(hybrid_train_df: pd.DataFrame) -> pd.Series:
    """Synthetic annualized realized vol — rolling std of log returns x sqrt(annualization).

    Leading NaN from the rolling window is preserved; the hybrid drops them
    via ``residuals.dropna()`` during fit.
    """
    log_ret = compute_log_returns(hybrid_train_df["close"])
    rolling_std = log_ret.rolling(REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW).std()
    rv: pd.Series = rolling_std * np.sqrt(Interval.DAILY.annualization_factor())
    return rv


def _fit_model(
    df: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    *,
    epochs: int = COMPACT_LSTM_EPOCHS,
    lookback: int = COMPACT_LSTM_LOOKBACK,
) -> HybridVolatilityModel:
    torch.manual_seed(GLOBAL_TORCH_SEED)
    np.random.seed(GLOBAL_NUMPY_SEED)
    model = HybridVolatilityModel(
        feature_columns=features,
        garch_p_max=COMPACT_GARCH_P_MAX,
        garch_q_max=COMPACT_GARCH_Q_MAX,
        lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
        lstm_num_layers=COMPACT_LSTM_NUM_LAYERS,
        lstm_lookback=lookback,
        lstm_epochs=epochs,
    )
    model.fit(df, target)
    return model


class TestHybridVolatilityModel:
    def test_predict_before_fit_raises(
        self, hybrid_train_df: pd.DataFrame, synthetic_feature_columns: list[str]
    ) -> None:
        m = HybridVolatilityModel(feature_columns=synthetic_feature_columns)
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(hybrid_train_df)

    def test_empty_feature_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_columns"):
            HybridVolatilityModel(feature_columns=[])

    def test_fit_predict_basic(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        result = model.predict(hybrid_train_df)
        assert isinstance(result, pd.Series)
        assert result.name == "hybrid_vol"
        assert result.index.equals(hybrid_train_df.index)

    def test_fit_drops_nan_warmup_feature_rows(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        """Symmetric with the HybridReturn test: defensive guard against
        NaN warmup feature rows reaching the scaler + LSTM. The vol target's
        20-bar warmup usually absorbs feature warmup in practice, but a future
        config with longer-warmup features (or shorter-warmup target) would
        otherwise expose the same NaN-loss path."""
        df = hybrid_train_df.copy()
        first_feature = synthetic_feature_columns[0]
        nan_warmup_rows = 15
        df.loc[df.index[:nan_warmup_rows], first_feature] = np.nan

        model = _fit_model(df, realized_vol_target, synthetic_feature_columns)

        assert model._scaler is not None
        assert np.isfinite(model._scaler.mean_).all()
        assert np.isfinite(model._scaler.scale_).all()
        result = model.predict(hybrid_train_df)
        assert result.index.equals(hybrid_train_df.index)

    def test_predict_output_clipped_to_min_vol(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        result = model.predict(hybrid_train_df).dropna()
        assert (result >= model._params.min_vol).all()

    def test_floor_bind_fraction_is_none_before_predict(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        assert model.last_floor_bind_fraction is None

    def test_floor_bind_fraction_within_unit_interval_after_predict(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        _ = model.predict(hybrid_train_df)
        frac = model.last_floor_bind_fraction
        assert frac is not None
        assert 0.0 <= frac <= 1.0

    def test_floor_bind_fraction_one_when_min_vol_dominates(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        """min_vol set far above any plausible volatility forecast forces
        every bar to clip; the tracked fraction should saturate at 1.0."""
        from src.models.hybrid_volatility import HybridVolatilityModel

        # Set min_vol so high that every (garch + lstm) bar gets clipped.
        # 1e6 is well above any realised-vol realisation our synthetic
        # frames produce — the floor is sure to bind on every non-NaN bar.
        model = HybridVolatilityModel(
            feature_columns=synthetic_feature_columns,
            lstm_epochs=5,
            lstm_lookback=5,
            min_vol=1e6,
        )
        model.fit(hybrid_train_df, realized_vol_target)
        _ = model.predict(hybrid_train_df)
        assert model.last_floor_bind_fraction == pytest.approx(1.0)

    def test_scaler_refit_raises_leakage(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        with pytest.raises(LeakageError):
            model.fit(hybrid_train_df, realized_vol_target)

    def test_garch_params_frozen_after_fit(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        snapshot = (
            model._garch._omega,
            model._garch._alpha.copy(),
            model._garch._beta.copy(),
            model._garch._best_p,
            model._garch._best_q,
        )
        model.predict(hybrid_train_df)
        assert model._garch._omega == snapshot[0]
        np.testing.assert_array_equal(model._garch._alpha, snapshot[1])
        np.testing.assert_array_equal(model._garch._beta, snapshot[2])
        assert model._garch._best_p == snapshot[3]
        assert model._garch._best_q == snapshot[4]

    def test_training_metadata_populated(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        meta = model.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(hybrid_train_df)
        assert meta.interval == Interval.DAILY
        assert tuple(meta.feature_columns) == tuple(synthetic_feature_columns)

    def test_training_metadata_validates_overlap(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        meta = model.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(hybrid_train_df)

    def test_residual_correction_changes_output(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(
            hybrid_train_df,
            realized_vol_target,
            synthetic_feature_columns,
            epochs=LSTM_EPOCHS_FOR_RESIDUAL_DIVERGENCE,
            lookback=COMPACT_LSTM_LOOKBACK,
        )
        hybrid_out = model.predict(hybrid_train_df).dropna()
        garch_only = model._garch.predict(hybrid_train_df).loc[hybrid_out.index]
        # LSTM residual should produce at least some non-trivial difference
        assert not np.allclose(
            hybrid_out.to_numpy(), garch_only.to_numpy(), atol=RESIDUAL_DIVERGENCE_ATOL
        )

    def test_predict_single_returns_float(
        self,
        hybrid_train_df: pd.DataFrame,
        realized_vol_target: pd.Series,
        synthetic_feature_columns: list[str],
    ) -> None:
        model = _fit_model(hybrid_train_df, realized_vol_target, synthetic_feature_columns)
        val = model.predict_single(hybrid_train_df)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_registry_registration(self) -> None:
        from src.core.registry import model_registry

        assert "hybrid_volatility" in model_registry

    def test_suggest_params_returns_combined_space(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = HybridVolatilityModel.suggest_params(trial)
        assert "garch_p_max" in params
        assert "garch_q_max" in params
        assert "lstm_hidden_dim" in params
        assert "lstm_lookback" in params
        assert "lstm_loss_fn" in params

    def test_hourly_interval(self, synthetic_feature_columns: list[str]) -> None:
        np.random.seed(GLOBAL_NUMPY_SEED)
        idx = pd.date_range(start=HOURLY_START, periods=HOURLY_ROW_COUNT, freq="h")
        returns = np.random.normal(0, HOURLY_RETURN_STD, HOURLY_ROW_COUNT)
        close = HOURLY_BASE_PRICE * np.cumprod(1 + returns)
        df = pd.DataFrame({"close": close}, index=idx)
        rng = np.random.default_rng(FEATURE_RNG_SEED)
        for col in synthetic_feature_columns:
            df[col] = rng.normal(0, 1, HOURLY_ROW_COUNT)

        log_ret = compute_log_returns(df["close"])
        rolling_std = log_ret.rolling(REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW).std()
        rv = rolling_std * np.sqrt(Interval.HOUR.annualization_factor())

        torch.manual_seed(GLOBAL_TORCH_SEED)
        model = HybridVolatilityModel(
            feature_columns=synthetic_feature_columns,
            garch_p_max=COMPACT_GARCH_P_MAX,
            garch_q_max=COMPACT_GARCH_Q_MAX,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_num_layers=COMPACT_LSTM_NUM_LAYERS,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=HOURLY_LSTM_EPOCHS,
            interval=Interval.HOUR,
        )
        model.fit(df, rv)
        meta = model.training_metadata
        assert meta is not None
        assert meta.interval == Interval.HOUR
