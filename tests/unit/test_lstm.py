"""
Tests for MarketLSTM and LSTMPredictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.exceptions import WarmupInsufficientError
from src.core.types import Device, Interval, LossFunction
from src.models.lstm import LSTMPredictor, MarketLSTM

SYNTH_ROW_COUNT = 150
SYNTH_START_DATE = "2020-01-02"
SYNTH_BASE_PRICE = 100.0
SYNTH_RETURN_MEAN = 0.0003
SYNTH_RETURN_STD = 0.01
SYNTH_FEATURE_NOISE_STD = 0.01
SYNTH_VOLUME_LOW = 1_000_000
SYNTH_VOLUME_HIGH = 5_000_000
SYNTH_FIXTURE_SEED = 42

FORWARD_BATCH_SIZE = 4
FORWARD_LOOKBACK = 10
FORWARD_FEATURE_COUNT = 3
EVAL_DROPOUT_BATCH_SIZE = 2
EVAL_DROPOUT_RATE = 0.5

COMPACT_HIDDEN_DIM = 16
COMPACT_NUM_LAYERS = 1
COMPACT_LOOKBACK = 10
QUICK_EPOCHS = 1
SHORT_EPOCHS = 3
MEDIUM_EPOCHS = 5
LONG_EPOCHS = 20
# Pair EXCESSIVE_EPOCHS with LOW_PATIENCE_FORCES_EARLY_STOP to trigger early stop.
EXCESSIVE_EPOCHS = 200
HIGH_PATIENCE_DISABLES_EARLY_STOP = 50
LOW_PATIENCE_FORCES_EARLY_STOP = 3
RECENT_WINDOW_SIZE = 15
TOO_SHORT_WINDOW_SIZE = 5

TORCH_SEED = 42
NUMPY_SEED = 42


@pytest.fixture
def lstm_df() -> pd.DataFrame:
    """
    DataFrame with multiple feature columns for LSTM testing.
    """

    np.random.seed(SYNTH_FIXTURE_SEED)
    idx = pd.bdate_range(start=SYNTH_START_DATE, periods=SYNTH_ROW_COUNT, freq="B")
    close = SYNTH_BASE_PRICE * np.cumprod(
        1 + np.random.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, SYNTH_ROW_COUNT)
    )
    return pd.DataFrame(
        {
            "close": close,
            "volume": np.random.randint(
                SYNTH_VOLUME_LOW, SYNTH_VOLUME_HIGH, SYNTH_ROW_COUNT
            ).astype(float),
            "return_1d": np.random.normal(0, SYNTH_FEATURE_NOISE_STD, SYNTH_ROW_COUNT),
        },
        index=idx,
    )


@pytest.fixture
def lstm_target(lstm_df: pd.DataFrame) -> pd.Series:
    """
    Target series for LSTM: next-day return.
    """

    returns = lstm_df["close"].pct_change().shift(-1)
    return returns.iloc[:-1]


@pytest.fixture
def lstm_features() -> list[str]:
    """
    Feature column list matching lstm_df.
    """

    return ["close", "volume", "return_1d"]


class TestMarketLSTM:
    def test_forward_shape(self) -> None:
        torch.manual_seed(TORCH_SEED)
        model = MarketLSTM(
            input_size=FORWARD_FEATURE_COUNT,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
        )
        x = torch.randn(FORWARD_BATCH_SIZE, FORWARD_LOOKBACK, FORWARD_FEATURE_COUNT)
        out = model(x)
        assert out.shape == (FORWARD_BATCH_SIZE, 1)

    def test_dropout_in_eval_mode(self) -> None:
        torch.manual_seed(TORCH_SEED)
        model = MarketLSTM(
            input_size=FORWARD_FEATURE_COUNT,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=2,
            dropout=EVAL_DROPOUT_RATE,
        )
        x = torch.randn(EVAL_DROPOUT_BATCH_SIZE, FORWARD_LOOKBACK, FORWARD_FEATURE_COUNT)

        model.eval()
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()
        torch.testing.assert_close(out1, out2)


class TestLSTMPredictor:
    def test_predict_before_fit_raises(
        self, lstm_df: pd.DataFrame, lstm_features: list[str]
    ) -> None:
        p = LSTMPredictor(lstm_features, lookback=COMPACT_LOOKBACK, epochs=QUICK_EPOCHS)
        with pytest.raises(RuntimeError, match="before fit"):
            p.predict(lstm_df)

    def test_predict_single_before_fit_raises(
        self, lstm_df: pd.DataFrame, lstm_features: list[str]
    ) -> None:
        p = LSTMPredictor(lstm_features, lookback=COMPACT_LOOKBACK, epochs=QUICK_EPOCHS)
        with pytest.raises(RuntimeError, match="before fit"):
            p.predict_single(lstm_df)

    def test_fit_and_predict(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series, lstm_features: list[str]
    ) -> None:
        torch.manual_seed(TORCH_SEED)
        np.random.seed(NUMPY_SEED)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(
            lstm_features,
            lookback=COMPACT_LOOKBACK,
            epochs=MEDIUM_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
        )
        p.fit(train, lstm_target)

        assert p.training_metadata is not None
        result = p.predict(train)
        assert isinstance(result, pd.Series)
        assert len(result) == len(train)
        assert result.iloc[:COMPACT_LOOKBACK].isna().all()
        assert result.iloc[COMPACT_LOOKBACK:].notna().all()

    def test_predict_single_shape(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series, lstm_features: list[str]
    ) -> None:
        torch.manual_seed(TORCH_SEED)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(
            lstm_features,
            lookback=COMPACT_LOOKBACK,
            epochs=SHORT_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
        )
        p.fit(train, lstm_target)

        val = p.predict_single(train.iloc[-RECENT_WINDOW_SIZE:])
        assert isinstance(val, float)

    def test_predict_single_too_short_raises(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series, lstm_features: list[str]
    ) -> None:
        torch.manual_seed(TORCH_SEED)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(
            lstm_features,
            lookback=COMPACT_LOOKBACK,
            epochs=SHORT_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
        )
        p.fit(train, lstm_target)

        with pytest.raises(WarmupInsufficientError, match="at least"):
            p.predict_single(train.iloc[:TOO_SHORT_WINDOW_SIZE])

    def test_training_reduces_loss(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series, lstm_features: list[str]
    ) -> None:
        torch.manual_seed(TORCH_SEED)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(
            lstm_features,
            lookback=COMPACT_LOOKBACK,
            epochs=LONG_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
            patience=HIGH_PATIENCE_DISABLES_EARLY_STOP,
        )
        p.fit(train, lstm_target)

        assert len(p._train_losses) > 0
        assert p._train_losses[0] > p._train_losses[-1]

    def test_early_stopping_fires(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series, lstm_features: list[str]
    ) -> None:
        torch.manual_seed(TORCH_SEED)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(
            lstm_features,
            lookback=COMPACT_LOOKBACK,
            epochs=EXCESSIVE_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
            patience=LOW_PATIENCE_FORCES_EARLY_STOP,
        )
        p.fit(train, lstm_target)
        assert len(p._train_losses) < EXCESSIVE_EPOCHS

    def test_val_losses_populated(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series, lstm_features: list[str]
    ) -> None:
        torch.manual_seed(TORCH_SEED)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(
            lstm_features,
            lookback=COMPACT_LOOKBACK,
            epochs=MEDIUM_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
        )
        p.fit(train, lstm_target)
        assert len(p._val_losses) > 0

    def test_configurable_loss_fn(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series, lstm_features: list[str]
    ) -> None:
        for loss_fn in LossFunction:
            torch.manual_seed(TORCH_SEED)
            train = lstm_df.iloc[:-1]
            p = LSTMPredictor(
                lstm_features,
                lookback=COMPACT_LOOKBACK,
                epochs=SHORT_EPOCHS,
                hidden_dim=COMPACT_HIDDEN_DIM,
                num_layers=COMPACT_NUM_LAYERS,
                loss_fn=loss_fn,
            )
            p.fit(train, lstm_target)
            assert p.training_metadata is not None

    def test_empty_feature_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_columns"):
            LSTMPredictor([])

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_val_split_ratio_raises(self, lstm_features: list[str], ratio: float) -> None:
        with pytest.raises(ValueError, match="val_split_ratio"):
            LSTMPredictor(lstm_features, val_split_ratio=ratio)

    def test_explicit_cpu_device_trains_and_predicts(
        self,
        lstm_df: pd.DataFrame,
        lstm_target: pd.Series,
        lstm_features: list[str],
    ) -> None:
        """
        End-to-end fit → predict on an explicitly-pinned CPU device (portable on CI).
        """

        torch.manual_seed(TORCH_SEED)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(
            lstm_features,
            lookback=COMPACT_LOOKBACK,
            epochs=SHORT_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            device=Device.CPU,
        )
        p.fit(train, lstm_target)
        assert p._device.type == "cpu"
        assert p._model is not None
        assert all(param.device.type == "cpu" for param in p._model.parameters())
        signal = p.predict(train)
        assert isinstance(signal, pd.Series)

    def test_feature_columns_honored(self, lstm_df: pd.DataFrame, lstm_target: pd.Series) -> None:
        """
        Explicit feature_columns restricts which columns LSTM trains on.
        """

        torch.manual_seed(TORCH_SEED)
        train = lstm_df.iloc[:-1]
        subset = ["close", "volume"]
        p = LSTMPredictor(
            subset,
            lookback=COMPACT_LOOKBACK,
            epochs=SHORT_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
        )
        p.fit(train, lstm_target)
        assert p._feature_columns == subset
        result = p.predict(train)
        assert isinstance(result, pd.Series)

    def test_training_metadata_populated(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series, lstm_features: list[str]
    ) -> None:
        torch.manual_seed(TORCH_SEED)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(
            lstm_features,
            lookback=COMPACT_LOOKBACK,
            epochs=SHORT_EPOCHS,
            hidden_dim=COMPACT_HIDDEN_DIM,
            num_layers=COMPACT_NUM_LAYERS,
        )
        p.fit(train, lstm_target)

        meta = p.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(train)
        assert meta.interval == Interval.DAILY

    def test_registry_registration(self) -> None:
        from src.core.registry import model_registry

        assert "lstm" in model_registry

    def test_suggest_params(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = LSTMPredictor.suggest_params(trial)
        assert "hidden_dim" in params
        assert "lr" in params
        assert "loss_fn" in params
