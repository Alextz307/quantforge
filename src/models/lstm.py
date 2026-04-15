"""LSTM predictor for time-series forecasting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.device import select_device
from src.core.registry import model_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Device, Interval, LossFunction
from src.core.utils import validate_open_unit_interval
from src.models.dataset import TemporalDataset
from src.models.interface import IPredictor

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

_LOSS_FUNCTIONS: dict[LossFunction, type[nn.Module]] = {
    LossFunction.MSE: nn.MSELoss,
    LossFunction.MAE: nn.L1Loss,
    LossFunction.HUBER: nn.HuberLoss,
}


class MarketLSTM(nn.Module):
    """LSTM network for financial time-series prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, lookback, features) -> (batch, 1)."""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out: torch.Tensor = self.fc(last_hidden)
        return out


@model_registry.register("lstm")
class LSTMPredictor(IPredictor):
    """LSTM-based predictor with early stopping and configurable loss.

    Splits training data temporally for validation-based early stopping;
    the split fraction is controlled by ``val_split_ratio`` (default 20%).
    Tracks train and validation losses for learning curve analysis.
    """

    def __init__(
        self,
        feature_columns: list[str],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lookback: int = 30,
        lr: float = 1e-3,
        epochs: int = 100,
        loss_fn: LossFunction = LossFunction.MSE,
        patience: int = 10,
        batch_size: int = 32,
        val_split_ratio: float = 0.2,
        device: Device | None = None,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if not feature_columns:
            raise ValueError("LSTMPredictor requires a non-empty feature_columns list")
        validate_open_unit_interval(val_split_ratio, "val_split_ratio")

        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._lookback = lookback
        self._lr = lr
        self._epochs = epochs
        self._loss_fn = loss_fn
        self._patience = patience
        self._batch_size = batch_size
        self._val_split_ratio = val_split_ratio
        self._device = select_device(device)
        self._interval = interval

        self._fitted = False
        self._model: MarketLSTM | None = None
        self._feature_columns: list[str] = list(feature_columns)
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._training_metadata: TrainingMetadata | None = None

    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> None:
        """Train LSTM with early stopping on temporal validation split.

        Args:
            train_data: DataFrame with features and DatetimeIndex.
            target: Target series aligned with train_data.
            **kwargs: If 'trial' key present, used for Optuna pruning.
        """
        trial: optuna.Trial | None = kwargs.get("trial", None)  # type: ignore[assignment]

        df = train_data.copy()
        target_col = "_target"
        df[target_col] = target.values

        split_idx = int(len(df) * (1.0 - self._val_split_ratio))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        if len(train_df) <= self._lookback or len(val_df) <= self._lookback:
            # Not enough data for validation split — train on all data
            train_ds = TemporalDataset(df, target_col, self._lookback, self._feature_columns)
            val_ds = None
        else:
            train_ds = TemporalDataset(train_df, target_col, self._lookback, self._feature_columns)
            val_ds = TemporalDataset(val_df, target_col, self._lookback, self._feature_columns)

        train_loader = DataLoader(train_ds, batch_size=self._batch_size, shuffle=False)

        input_size = len(self._feature_columns)
        self._model = MarketLSTM(
            input_size=input_size,
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
            dropout=self._dropout,
        ).to(self._device)

        criterion = _LOSS_FUNCTIONS[self._loss_fn]()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        self._train_losses = []
        self._val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self._epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            for features, targets in train_loader:
                features = features.to(self._device)
                targets = targets.to(self._device)
                optimizer.zero_grad()
                preds = self._model(features).squeeze(-1)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self._train_losses.append(avg_train_loss)

            if val_ds is not None:
                val_loader = DataLoader(val_ds, batch_size=self._batch_size, shuffle=False)
                self._model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for features, targets in val_loader:
                        features = features.to(self._device)
                        targets = targets.to(self._device)
                        preds = self._model(features).squeeze(-1)
                        val_loss += criterion(preds, targets).item()
                        val_batches += 1
                avg_val_loss = val_loss / max(val_batches, 1)
                self._val_losses.append(avg_val_loss)

                if trial is not None:
                    trial.report(avg_val_loss, epoch)
                    if trial.should_prune():
                        from optuna.exceptions import TrialPruned

                        raise TrialPruned()

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self._patience:
                        logger.info("LSTM early stopping at epoch %d", epoch + 1)
                        break

        if best_state is not None and self._model is not None:
            self._model.load_state_dict(best_state)

        self._fitted = True
        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, tuple(self._feature_columns)
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Batch inference on data.

        First `lookback` rows return NaN (insufficient history).

        Args:
            data: DataFrame with same feature columns as training data.

        Returns:
            Series of predictions aligned with data index.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("LSTMPredictor.predict() called before fit()")

        self._model.eval()
        features = torch.from_numpy(data[self._feature_columns].to_numpy(dtype=np.float32)).to(
            self._device
        )

        predictions = np.full(len(data), np.nan)
        n_windows = len(data) - self._lookback

        if n_windows > 0:
            # Batch every lookback window into one forward pass — avoids
            # Python-per-bar overhead. The unfold view is non-contiguous, so
            # call .contiguous() explicitly to make the unavoidable copy
            # visible (nn.LSTM would do it implicitly otherwise).
            with torch.no_grad():
                windows = features[:-1].unfold(0, self._lookback, 1).transpose(1, 2).contiguous()
                preds = self._model(windows).squeeze(-1).cpu().numpy()
            predictions[self._lookback : self._lookback + n_windows] = preds

        return pd.Series(predictions, index=data.index, name="lstm_pred")

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Predict single value from a recent data window."""
        if not self._fitted or self._model is None:
            raise RuntimeError("LSTMPredictor.predict_single() called before fit()")

        if len(recent_window) < self._lookback:
            raise ValueError(f"Need at least {self._lookback} rows, got {len(recent_window)}")

        self._model.eval()
        features = torch.from_numpy(
            recent_window[self._feature_columns].iloc[-self._lookback :].to_numpy(dtype=np.float32)
        ).to(self._device)

        with torch.no_grad():
            pred = self._model(features.unsqueeze(0)).item()

        return float(pred)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for LSTM hyperparameters."""
        return {
            "hidden_dim": trial.suggest_int("lstm_hidden_dim", 32, 128),
            "num_layers": trial.suggest_int("lstm_num_layers", 1, 3),
            "dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
            "lookback": trial.suggest_int("lstm_lookback", 10, 60),
            "lr": trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True),
            "loss_fn": LossFunction(
                trial.suggest_categorical("lstm_loss_fn", [e.value for e in LossFunction])
            ),
            "batch_size": trial.suggest_categorical("lstm_batch_size", [16, 32, 64]),
        }
