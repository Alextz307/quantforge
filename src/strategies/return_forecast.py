"""
Return-forecast strategy driven by HybridReturnModel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd

from src.core import json_io
from src.core.logging import get_logger
from src.core.persistence import (
    CONFIG_JSON,
    HYBRID_RETURN_SUBDIR,
    METADATA_JSON,
    assert_save_complete,
    save_model_skeleton,
)
from src.core.registry import strategy_registry
from src.core.temporal import (
    TrackedMetadata,
    TrainingMetadata,
    collect_metadata,
)
from src.core.types import Device, InformationCriterion, Interval, LossFunction
from src.core.utils import compute_log_returns
from src.models.hybrid_return import HybridReturnModel
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = get_logger(__name__)


@dataclass(frozen=True)
class _HybridReturnParams:
    """
    Immutable bundle of HybridReturnModel constructor kwargs.

    Stored on the strategy so ``train()`` can rebuild a fresh hybrid with a
    clean scaler each invocation (the hybrid's fit-once guard rejects a
    second fit on the same instance). ``feature_columns`` is a tuple so the
    bundle is truly immutable — frozen=True alone wouldn't prevent mutation
    of a list field.
    """

    feature_columns: tuple[str, ...]
    arma_p_max: int
    arma_q_max: int
    arma_information_criterion: InformationCriterion
    lstm_hidden_dim: int
    lstm_num_layers: int
    lstm_dropout: float
    lstm_lookback: int
    lstm_lr: float
    lstm_epochs: int
    lstm_loss_fn: LossFunction
    lstm_patience: int
    lstm_batch_size: int
    lstm_val_split_ratio: float
    lstm_device: Device | None
    lstm_amp: bool
    interval: Interval


@strategy_registry.register("ReturnForecast")
class ReturnForecastStrategy(IStrategy):
    """
    Position = clip(``position_scale * forecast_return``, ±``max_leverage``).

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
        arma_information_criterion: InformationCriterion = InformationCriterion.AIC,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        lstm_lookback: int = 30,
        lstm_lr: float = 1e-3,
        lstm_epochs: int = 100,
        lstm_loss_fn: LossFunction = LossFunction.MSE,
        lstm_patience: int = 10,
        lstm_batch_size: int = 32,
        lstm_val_split_ratio: float = 0.2,
        lstm_device: Device | None = None,
        lstm_amp: bool = False,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if position_scale <= 0:
            raise ValueError(
                f"position_scale must be > 0, got {position_scale}; fix by passing "
                f"a strictly positive multiplier on the forecasted return "
                f"(typical: 1.0 — adjust to control aggressiveness)."
            )
        if max_leverage <= 0:
            raise ValueError(
                f"max_leverage must be > 0, got {max_leverage}; fix by passing "
                f"the leverage cap as a strictly positive multiplier (typical: 1.0)."
            )

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
            lstm_val_split_ratio=lstm_val_split_ratio,
            lstm_device=lstm_device,
            lstm_amp=lstm_amp,
            interval=interval,
        )

        self._hybrid_return = self._build_hybrid_return()

    def _build_hybrid_return(self) -> HybridReturnModel:
        kwargs = asdict(self._hybrid_params)
        kwargs["feature_columns"] = list(self._hybrid_params.feature_columns)
        return HybridReturnModel(**kwargs)

    def train(
        self,
        train_data: pd.DataFrame,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        """
        Fit HybridReturnModel on training log returns.
        """

        logger.info("%s train: %d bars", type(self).__name__, len(train_data))
        self._hybrid_return = self._build_hybrid_return()
        log_returns = compute_log_returns(train_data["close"]).dropna()
        aligned = train_data.loc[log_returns.index]
        self._hybrid_return.fit(aligned, log_returns, checkpoint_path=checkpoint_path, **kwargs)

        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(
                train_data, self._interval, self._hybrid_params.feature_columns
            )
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Produce signed positions in ``[-max_leverage, +max_leverage]``.
        """

        self._assert_fitted_with_metadata()

        forecast = self._hybrid_return.predict(data)
        raw_position = forecast * self._position_scale
        position = raw_position.clip(lower=-self._max_leverage, upper=self._max_leverage)
        position.name = "return_forecast_signal"
        return position

    def save(self, path: str | Path) -> None:
        """
        Persist ReturnForecast config + nested HybridReturn to ``path``.

        Strategy-specific kwargs (``position_scale``, ``max_leverage``) are
        written alongside every passthrough ``_HybridReturnParams`` field —
        the two together reconstruct the full ctor signature on load. Leaf
        device preference is NOT persisted (the hybrid subdir carries the
        fitted state; device re-resolves on load).
        """

        metadata = self._assert_fitted_with_metadata()

        def write_weights(root: Path) -> None:
            self._hybrid_return.save(root / HYBRID_RETURN_SUBDIR)

        save_model_skeleton(
            path,
            config=self._ctor_kwargs_as_json(),
            training_metadata=metadata,
            write_weights=write_weights,
        )

    def _ctor_kwargs_as_json(self) -> dict[str, object]:
        """
        Snapshot of this strategy's constructor kwargs as JSON-ready values.
        """

        p = self._hybrid_params
        return {
            "position_scale": self._position_scale,
            "max_leverage": self._max_leverage,
            "feature_columns": list(p.feature_columns),
            "arma_p_max": p.arma_p_max,
            "arma_q_max": p.arma_q_max,
            "arma_information_criterion": p.arma_information_criterion.value,
            "lstm_hidden_dim": p.lstm_hidden_dim,
            "lstm_num_layers": p.lstm_num_layers,
            "lstm_dropout": p.lstm_dropout,
            "lstm_lookback": p.lstm_lookback,
            "lstm_lr": p.lstm_lr,
            "lstm_epochs": p.lstm_epochs,
            "lstm_loss_fn": p.lstm_loss_fn.value,
            "lstm_patience": p.lstm_patience,
            "lstm_batch_size": p.lstm_batch_size,
            "lstm_val_split_ratio": p.lstm_val_split_ratio,
            "lstm_amp": p.lstm_amp,
            "interval": p.interval.value,
        }

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Reconstruct a trained ReturnForecastStrategy from ``path``.

        Narrow the strategy's ``config.json`` into ctor kwargs BEFORE loading
        the nested ``hybrid_return/`` subdir — a corrupt strategy config
        fast-fails with a named-field error, without wasting I/O on the
        HybridReturnModel's nested ARMA + LSTM + scaler loads.
        """

        root = assert_save_complete(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            feature_columns=json_io.get_str_list(config, "feature_columns"),
            position_scale=json_io.get_float(config, "position_scale"),
            max_leverage=json_io.get_float(config, "max_leverage"),
            arma_p_max=json_io.get_int(config, "arma_p_max"),
            arma_q_max=json_io.get_int(config, "arma_q_max"),
            arma_information_criterion=InformationCriterion(
                json_io.get_str(config, "arma_information_criterion")
            ),
            lstm_hidden_dim=json_io.get_int(config, "lstm_hidden_dim"),
            lstm_num_layers=json_io.get_int(config, "lstm_num_layers"),
            lstm_dropout=json_io.get_float(config, "lstm_dropout"),
            lstm_lookback=json_io.get_int(config, "lstm_lookback"),
            lstm_lr=json_io.get_float(config, "lstm_lr"),
            lstm_epochs=json_io.get_int(config, "lstm_epochs"),
            lstm_loss_fn=LossFunction(json_io.get_str(config, "lstm_loss_fn")),
            lstm_patience=json_io.get_int(config, "lstm_patience"),
            lstm_batch_size=json_io.get_int(config, "lstm_batch_size"),
            lstm_val_split_ratio=json_io.get_float(config, "lstm_val_split_ratio"),
            lstm_amp=json_io.get_bool(config, "lstm_amp"),
            interval=Interval(json_io.get_str(config, "interval")),
        )

        instance._hybrid_return = HybridReturnModel.load(root / HYBRID_RETURN_SUBDIR)
        instance._set_fitted_with_metadata(TrainingMetadata.from_dict(metadata))
        return instance

    @property
    def name(self) -> str:
        return "ReturnForecast"

    @property
    def required_warmup_bars(self) -> int:
        return self._lstm_lookback

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """
        Expose strategy + recursively-owned hybrid-return leaves (arma + lstm).
        """

        return (
            collect_metadata(
                ("strategy", self.training_metadata),
            )
            + self._hybrid_return.get_all_training_metadata()
        )

    @staticmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
        """
        Optuna search space for ReturnForecast hyperparameters.
        """

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
