"""Volatility-targeting strategy driven by HybridVolatilityModel forecasts."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd

import quant_engine
from src.core import json_io
from src.core.constants import TRADING_DAYS_PER_YEAR
from src.core.persistence import (
    CONFIG_JSON,
    HYBRID_VOL_SUBDIR,
    METADATA_JSON,
    save_model_skeleton,
)
from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Device, Interval, LossFunction
from src.models.hybrid_volatility import HybridVolatilityModel
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _HybridVolParams:
    """Immutable bundle of HybridVolatilityModel constructor kwargs.

    Stored on the strategy so ``train()`` can rebuild a fresh hybrid with a
    clean scaler each invocation (the hybrid's fit-once guard rejects a
    second fit on the same instance). ``feature_columns`` is a tuple so the
    bundle is truly immutable — frozen=True alone wouldn't prevent mutation
    of a list field.
    """

    feature_columns: tuple[str, ...]
    garch_p_max: int
    garch_q_max: int
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
    min_vol: float
    interval: Interval


@strategy_registry.register("VolatilityTargeting")
class VolatilityTargetingStrategy(IStrategy):
    """Scale long exposure to hit a target portfolio volatility.

    Leverage = ``target_vol / forecast_vol`` clipped to ``[0, max_leverage]``.
    A trend MA gates the regime: in bearish windows, leverage is multiplied
    by ``bearish_exposure`` (default 0 → flat).

    The realized-volatility training target is the annualized Garman-Klass
    OHLC estimator over ``realized_vol_window`` bars. Input DataFrames must
    carry ``open``/``high``/``low``/``close`` columns.
    """

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_vol: float = 0.15,
        trend_window: int = 100,
        max_leverage: float = 1.5,
        bearish_exposure: float = 0.0,
        realized_vol_window: int = 20,
        garch_p_max: int = 5,
        garch_q_max: int = 5,
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
        min_vol: float = 1e-3,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if target_vol <= 0:
            raise ValueError(f"target_vol must be > 0, got {target_vol}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}")
        if bearish_exposure < 0:
            raise ValueError(f"bearish_exposure must be >= 0, got {bearish_exposure}")
        if trend_window < 2:
            raise ValueError(f"trend_window must be >= 2, got {trend_window}")
        if realized_vol_window < 2:
            raise ValueError(f"realized_vol_window must be >= 2, got {realized_vol_window}")

        self._target_vol = target_vol
        self._trend_window = trend_window
        self._max_leverage = max_leverage
        self._bearish_exposure = bearish_exposure
        self._realized_vol_window = realized_vol_window
        self._lstm_lookback = lstm_lookback
        self._interval = interval

        self._hybrid_params = _HybridVolParams(
            feature_columns=tuple(feature_columns),
            garch_p_max=garch_p_max,
            garch_q_max=garch_q_max,
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
            min_vol=min_vol,
            interval=interval,
        )

        self._hybrid_vol = self._build_hybrid_vol()
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None

    def _build_hybrid_vol(self) -> HybridVolatilityModel:
        kwargs = asdict(self._hybrid_params)
        kwargs["feature_columns"] = list(self._hybrid_params.feature_columns)
        return HybridVolatilityModel(**kwargs)

    def _compute_realized_vol(self, bars: pd.DataFrame) -> pd.Series:
        """Annualized Garman-Klass realized volatility at the strategy's interval.

        The underlying C++ estimator annualizes assuming daily bars; we rescale
        so ``Interval.HOUR`` and friends land on the same annualized horizon as
        the rest of the framework.
        """
        gk = quant_engine.GarmanKlass(self._realized_vol_window).compute(
            bars["open"].to_numpy(dtype=float, copy=False),
            bars["high"].to_numpy(dtype=float, copy=False),
            bars["low"].to_numpy(dtype=float, copy=False),
            bars["close"].to_numpy(dtype=float, copy=False),
        )
        interval_scale = math.sqrt(self._interval.annualization_factor() / TRADING_DAYS_PER_YEAR)
        return pd.Series(gk * interval_scale, index=bars.index)

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Fit HybridVolatilityModel on an internally-computed realized-vol target."""
        self._hybrid_vol = self._build_hybrid_vol()
        realized_vol = self._compute_realized_vol(train_data)
        target = realized_vol.dropna()
        aligned = train_data.loc[target.index]

        self._hybrid_vol.fit(aligned, target, **kwargs)

        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, self._hybrid_params.feature_columns
        )
        self._fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce leverage signals in ``[0, max_leverage]``. Warmup bars are NaN."""
        if not self._fitted:
            raise RuntimeError(
                "VolatilityTargetingStrategy.generate_signals() called before train()"
            )

        forecast_vol = self._hybrid_vol.predict(data)
        raw_leverage = self._target_vol / forecast_vol
        leverage = raw_leverage.clip(lower=0.0, upper=self._max_leverage)

        trend_ma = data["close"].rolling(self._trend_window).mean()
        is_bull = data["close"] > trend_ma
        gated = leverage.where(is_bull, leverage * self._bearish_exposure)
        gated = gated.where(trend_ma.notna())
        gated.name = "vol_target_signal"
        return gated

    def save(self, path: str | Path) -> None:
        """Persist VolatilityTargeting config + nested HybridVolatility.

        Strategy-specific kwargs (``target_vol``, ``trend_window``,
        ``max_leverage``, ``bearish_exposure``, ``realized_vol_window``) are
        written alongside every passthrough ``_HybridVolParams`` field.
        """
        if not self._fitted:
            raise RuntimeError("VolatilityTargetingStrategy.save() called before train()")
        if self._training_metadata is None:
            raise RuntimeError("VolatilityTargetingStrategy.save() missing training metadata")

        def write_weights(root: Path) -> None:
            self._hybrid_vol.save(root / HYBRID_VOL_SUBDIR)

        save_model_skeleton(
            path,
            config=self._ctor_kwargs_as_json(),
            training_metadata=self._training_metadata,
            write_weights=write_weights,
        )

    def _ctor_kwargs_as_json(self) -> dict[str, object]:
        """Snapshot of this strategy's constructor kwargs as JSON-ready values."""
        p = self._hybrid_params
        return {
            "target_vol": self._target_vol,
            "trend_window": self._trend_window,
            "max_leverage": self._max_leverage,
            "bearish_exposure": self._bearish_exposure,
            "realized_vol_window": self._realized_vol_window,
            "feature_columns": list(p.feature_columns),
            "garch_p_max": p.garch_p_max,
            "garch_q_max": p.garch_q_max,
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
            "min_vol": p.min_vol,
            "interval": p.interval.value,
        }

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a trained VolatilityTargetingStrategy from ``path``.

        Narrow the strategy's ``config.json`` into ctor kwargs BEFORE loading
        the nested ``hybrid_vol/`` subdir — a corrupt strategy config
        fast-fails with a named-field error, without wasting I/O on the
        HybridVolatilityModel's nested GARCH + LSTM + scaler loads.
        """
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            feature_columns=json_io.get_str_list(config, "feature_columns"),
            target_vol=json_io.get_float(config, "target_vol"),
            trend_window=json_io.get_int(config, "trend_window"),
            max_leverage=json_io.get_float(config, "max_leverage"),
            bearish_exposure=json_io.get_float(config, "bearish_exposure"),
            realized_vol_window=json_io.get_int(config, "realized_vol_window"),
            garch_p_max=json_io.get_int(config, "garch_p_max"),
            garch_q_max=json_io.get_int(config, "garch_q_max"),
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
            min_vol=json_io.get_float(config, "min_vol"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        instance._hybrid_vol = HybridVolatilityModel.load(root / HYBRID_VOL_SUBDIR)
        instance._training_metadata = TrainingMetadata.from_dict(metadata)
        instance._fitted = True
        return instance

    @property
    def name(self) -> str:
        return "VolatilityTargeting"

    @property
    def required_warmup_bars(self) -> int:
        return max(self._trend_window, self._lstm_lookback, self._realized_vol_window)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for VolatilityTargeting hyperparameters."""
        return {
            "target_vol": trial.suggest_float("volt_target_vol", 0.05, 0.30),
            "trend_window": trial.suggest_int("volt_trend_window", 50, 200),
            "max_leverage": trial.suggest_float("volt_max_leverage", 1.0, 3.0),
            "bearish_exposure": trial.suggest_float("volt_bearish_exposure", 0.0, 1.0),
            "realized_vol_window": trial.suggest_int("volt_rvol_window", 10, 40),
            "lstm_hidden_dim": trial.suggest_int("volt_lstm_hidden_dim", 32, 128),
            "lstm_num_layers": trial.suggest_int("volt_lstm_num_layers", 1, 3),
            "lstm_dropout": trial.suggest_float("volt_lstm_dropout", 0.0, 0.5),
            "lstm_lookback": trial.suggest_int("volt_lstm_lookback", 10, 60),
            "lstm_lr": trial.suggest_float("volt_lstm_lr", 1e-4, 1e-2, log=True),
        }
