"""Return-forecast strategy driven by HybridReturnModel."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import pandas as pd

from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import InformationCriterion, Interval, LossFunction
from src.core.utils import compute_log_returns
from src.models.hybrid_return import HybridReturnModel
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _HybridReturnParams:
    """Immutable bundle of HybridReturnModel constructor kwargs.

    Stored on the strategy so ``train()`` can rebuild a fresh hybrid with a
    clean scaler each invocation (the hybrid's fit-once guard rejects a
    second fit on the same instance). ``feature_columns`` is a tuple so the
    bundle is truly immutable — frozen=True alone wouldn't prevent mutation
    of a list field.
    """

    feature_columns: tuple[str, ...]
    arma_p_max: int
    arma_q_max: int
    arma_information_criterion: InformationCriterion | str
    lstm_hidden_dim: int
    lstm_num_layers: int
    lstm_dropout: float
    lstm_lookback: int
    lstm_lr: float
    lstm_epochs: int
    lstm_loss_fn: LossFunction | str
    lstm_patience: int
    lstm_batch_size: int
    interval: Interval


@strategy_registry.register("ReturnForecast")
class ReturnForecastStrategy(IStrategy):
    """Position = clip(``position_scale * forecast_return``, ±``max_leverage``).

    Uses ``HybridReturnModel`` (ARMA + LSTM residual) for the conditional-mean
    forecast of next-bar log returns. Positive forecast → long, negative
    forecast → short, scaled linearly and then clipped.
    """

    def __init__(
        self,
        *,
        feature_columns: list[str],
        position_scale: float = 20.0,
        max_leverage: float = 1.5,
        arma_p_max: int = 5,
        arma_q_max: int = 5,
        arma_information_criterion: InformationCriterion | str = InformationCriterion.AIC,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        lstm_lookback: int = 30,
        lstm_lr: float = 1e-3,
        lstm_epochs: int = 100,
        lstm_loss_fn: LossFunction | str = LossFunction.MSE,
        lstm_patience: int = 10,
        lstm_batch_size: int = 32,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if position_scale <= 0:
            raise ValueError(f"position_scale must be > 0, got {position_scale}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}")

        self._position_scale = position_scale
        self._max_leverage = max_leverage
        self._lstm_lookback = lstm_lookback
        self._interval = interval

        self._hybrid_params = _HybridReturnParams(
            feature_columns=tuple(feature_columns),
            arma_p_max=arma_p_max,
            arma_q_max=arma_q_max,
            arma_information_criterion=arma_information_criterion,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout,
            lstm_lookback=lstm_lookback,
            lstm_lr=lstm_lr,
            lstm_epochs=lstm_epochs,
            lstm_loss_fn=lstm_loss_fn,
            lstm_patience=lstm_patience,
            lstm_batch_size=lstm_batch_size,
            interval=interval,
        )

        self._hybrid_return = self._build_hybrid_return()
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None

    def _build_hybrid_return(self) -> HybridReturnModel:
        kwargs = asdict(self._hybrid_params)
        kwargs["feature_columns"] = list(self._hybrid_params.feature_columns)
        return HybridReturnModel(**kwargs)

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Fit HybridReturnModel on training log returns."""
        self._hybrid_return = self._build_hybrid_return()
        log_returns = compute_log_returns(train_data["close"]).dropna()
        aligned = train_data.loc[log_returns.index]

        self._hybrid_return.fit(aligned, log_returns, **kwargs)

        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, self._hybrid_params.feature_columns
        )
        self._fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce signed positions in ``[-max_leverage, +max_leverage]``."""
        if not self._fitted:
            raise RuntimeError("ReturnForecastStrategy.generate_signals() called before train()")

        forecast = self._hybrid_return.predict(data)
        raw_position = forecast * self._position_scale
        position = raw_position.clip(lower=-self._max_leverage, upper=self._max_leverage)
        position.name = "return_forecast_signal"
        return position

    @property
    def name(self) -> str:
        return "ReturnForecast"

    @property
    def required_warmup_bars(self) -> int:
        return self._lstm_lookback

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for ReturnForecast hyperparameters."""
        return {
            "position_scale": trial.suggest_float("retf_position_scale", 5.0, 50.0),
            "max_leverage": trial.suggest_float("retf_max_leverage", 1.0, 3.0),
            "arma_p_max": trial.suggest_int("retf_arma_p_max", 1, 5),
            "arma_q_max": trial.suggest_int("retf_arma_q_max", 1, 5),
            "lstm_hidden_dim": trial.suggest_int("retf_lstm_hidden_dim", 32, 128),
            "lstm_num_layers": trial.suggest_int("retf_lstm_num_layers", 1, 3),
            "lstm_dropout": trial.suggest_float("retf_lstm_dropout", 0.0, 0.5),
            "lstm_lookback": trial.suggest_int("retf_lstm_lookback", 10, 60),
            "lstm_lr": trial.suggest_float("retf_lstm_lr", 1e-4, 1e-2, log=True),
        }
