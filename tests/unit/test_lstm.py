"""Tests for MarketLSTM and LSTMPredictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.types import Interval
from src.models.lstm import LSTMPredictor, MarketLSTM


@pytest.fixture
def lstm_df() -> pd.DataFrame:
    """DataFrame with multiple feature columns for LSTM testing."""
    np.random.seed(42)
    n = 150
    idx = pd.bdate_range(start="2020-01-02", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))
    return pd.DataFrame(
        {
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
            "return_1d": np.random.normal(0, 0.01, n),
        },
        index=idx,
    )


@pytest.fixture
def lstm_target(lstm_df: pd.DataFrame) -> pd.Series:
    """Target series for LSTM: next-day return."""
    returns = lstm_df["close"].pct_change().shift(-1)
    # Drop last row (no future target); leading NaN from pct_change is
    # shifted out by shift(-1), so no fill needed
    return returns.iloc[:-1]


class TestMarketLSTM:
    def test_forward_shape(self) -> None:
        torch.manual_seed(42)
        model = MarketLSTM(input_size=3, hidden_dim=16, num_layers=1)
        x = torch.randn(4, 10, 3)  # batch=4, lookback=10, features=3
        out = model(x)
        assert out.shape == (4, 1)

    def test_dropout_in_eval_mode(self) -> None:
        torch.manual_seed(42)
        model = MarketLSTM(input_size=3, hidden_dim=16, num_layers=2, dropout=0.5)
        x = torch.randn(2, 10, 3)

        model.eval()
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()
        # In eval mode, dropout is disabled — outputs should be identical
        torch.testing.assert_close(out1, out2)


class TestLSTMPredictor:
    def test_predict_before_fit_raises(self, lstm_df: pd.DataFrame) -> None:
        p = LSTMPredictor(lookback=10, epochs=1)
        with pytest.raises(RuntimeError, match="before fit"):
            p.predict(lstm_df)

    def test_predict_single_before_fit_raises(self, lstm_df: pd.DataFrame) -> None:
        p = LSTMPredictor(lookback=10, epochs=1)
        with pytest.raises(RuntimeError, match="before fit"):
            p.predict_single(lstm_df)

    def test_fit_and_predict(self, lstm_df: pd.DataFrame, lstm_target: pd.Series) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        train = lstm_df.iloc[:-1]  # match target length
        p = LSTMPredictor(lookback=10, epochs=5, hidden_dim=16, num_layers=1)
        p.fit(train, lstm_target)

        assert p._fitted
        result = p.predict(train)
        assert isinstance(result, pd.Series)
        assert len(result) == len(train)
        # First lookback rows should be NaN
        assert result.iloc[:10].isna().all()
        # Remaining should have values
        assert result.iloc[10:].notna().all()

    def test_predict_single_shape(self, lstm_df: pd.DataFrame, lstm_target: pd.Series) -> None:
        torch.manual_seed(42)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(lookback=10, epochs=3, hidden_dim=16, num_layers=1)
        p.fit(train, lstm_target)

        val = p.predict_single(train.iloc[-15:])
        assert isinstance(val, float)

    def test_predict_single_too_short_raises(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series
    ) -> None:
        torch.manual_seed(42)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(lookback=10, epochs=3, hidden_dim=16, num_layers=1)
        p.fit(train, lstm_target)

        with pytest.raises(ValueError, match="at least"):
            p.predict_single(train.iloc[:5])

    def test_training_reduces_loss(self, lstm_df: pd.DataFrame, lstm_target: pd.Series) -> None:
        torch.manual_seed(42)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(lookback=10, epochs=20, hidden_dim=16, num_layers=1, patience=50)
        p.fit(train, lstm_target)

        assert len(p._train_losses) > 0
        # First loss should be larger than last (training should reduce loss)
        assert p._train_losses[0] > p._train_losses[-1]

    def test_early_stopping_fires(self, lstm_df: pd.DataFrame, lstm_target: pd.Series) -> None:
        torch.manual_seed(42)
        train = lstm_df.iloc[:-1]
        # Very low patience to trigger early stopping
        p = LSTMPredictor(lookback=10, epochs=200, hidden_dim=16, num_layers=1, patience=3)
        p.fit(train, lstm_target)
        # Should stop well before 200 epochs
        assert len(p._train_losses) < 200

    def test_val_losses_populated(self, lstm_df: pd.DataFrame, lstm_target: pd.Series) -> None:
        torch.manual_seed(42)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(lookback=10, epochs=5, hidden_dim=16, num_layers=1)
        p.fit(train, lstm_target)
        assert len(p._val_losses) > 0

    def test_configurable_loss_fn(self, lstm_df: pd.DataFrame, lstm_target: pd.Series) -> None:
        for loss_fn in ["mse", "mae", "huber"]:
            torch.manual_seed(42)
            train = lstm_df.iloc[:-1]
            p = LSTMPredictor(lookback=10, epochs=3, hidden_dim=16, num_layers=1, loss_fn=loss_fn)
            p.fit(train, lstm_target)
            assert p._fitted

    def test_invalid_loss_fn_raises(self) -> None:
        with pytest.raises(ValueError, match="loss_fn"):
            LSTMPredictor(loss_fn="invalid")

    def test_training_metadata_populated(
        self, lstm_df: pd.DataFrame, lstm_target: pd.Series
    ) -> None:
        torch.manual_seed(42)
        train = lstm_df.iloc[:-1]
        p = LSTMPredictor(lookback=10, epochs=3, hidden_dim=16, num_layers=1)
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
