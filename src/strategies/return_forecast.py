"""Return-forecast strategy driven by HybridReturnModel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, cast

import pandas as pd

from src.core import json_io
from src.core.logging import get_logger
from src.core.persistence import (
    CONFIG_JSON,
    HYBRID_RETURN_SUBDIR,
    METADATA_JSON,
    save_model_skeleton,
)
from src.core.registry import strategy_registry
from src.core.temporal import (
    TrackedMetadata,
    TrainingMetadata,
    collect_metadata,
    mark_pretrained,
)
from src.core.types import Device, InformationCriterion, Interval, LossFunction
from src.core.utils import compute_log_returns
from src.models.hybrid_return import HybridReturnModel
from src.orchestration.pretrained_leaves import (
    normalize_pretrained_leaves,
    validate_pretrained_leaf,
)
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = get_logger(__name__)


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
    interval: Interval


_LEAF_KEY_RETURN_MODEL = "return_model"


@strategy_registry.register("ReturnForecast")
class ReturnForecastStrategy(IStrategy):
    """Position = clip(``position_scale * forecast_return``, ±``max_leverage``).

    Uses ``HybridReturnModel`` (ARMA + LSTM residual) for the conditional-mean
    forecast of next-bar log returns. Positive forecast → long, negative
    forecast → short, scaled linearly and then clipped.

    Supports pretrained-leaf injection: passing
    ``pretrained_leaves={"return_model": loaded_hybrid}`` freezes the leaf
    (skips rebuild + fit across every ``train()`` call). Strategy-level
    state (``position_scale``, ``max_leverage``) is ctor-only, so a frozen
    leaf means ``train()`` updates only ``_training_metadata``.
    """

    _leaf_keys: ClassVar[frozenset[str]] = frozenset({_LEAF_KEY_RETURN_MODEL})

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
        interval: Interval = Interval.DAILY,
        pretrained_leaves: Mapping[str, object] | None = None,
    ) -> None:
        if position_scale <= 0:
            raise ValueError(f"position_scale must be > 0, got {position_scale}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}")

        self._pretrained_leaves = normalize_pretrained_leaves(
            pretrained_leaves, self._leaf_keys, type(self).__name__
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
            interval=interval,
        )

        if _LEAF_KEY_RETURN_MODEL in self._pretrained_leaves:
            injected = self._pretrained_leaves[_LEAF_KEY_RETURN_MODEL]
            validate_pretrained_leaf(
                injected,
                interval=interval,
                feature_columns=self._hybrid_params.feature_columns,
                lstm_lookback=lstm_lookback,
            )
            self._hybrid_return = cast(HybridReturnModel, injected)
            # Sync passthrough params from the frozen leaf so ``save()``
            # round-trips to the real artifact (not ctor defaults). ``getattr``
            # fallback accepts test fakes that duck-type the leaf surface.
            leaf_config = getattr(self._hybrid_return, "params", None)
            if leaf_config is not None:
                self._hybrid_params = _HybridReturnParams(**asdict(leaf_config))
        else:
            self._hybrid_return = self._build_hybrid_return()
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None

    def _build_hybrid_return(self) -> HybridReturnModel:
        kwargs = asdict(self._hybrid_params)
        kwargs["feature_columns"] = list(self._hybrid_params.feature_columns)
        return HybridReturnModel(**kwargs)

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Fit HybridReturnModel on training log returns.

        When ``pretrained_leaves["return_model"]`` was injected at ctor time,
        the leaf stays frozen: neither rebuild nor ``fit()`` runs. Only
        ``_training_metadata`` advances so the walk-forward deep metadata
        check records the strategy's own fold window.
        """
        if _LEAF_KEY_RETURN_MODEL not in self._pretrained_leaves:
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

    def update(self, new_data: pd.DataFrame, **kwargs: object) -> None:
        """Delegate to HybridReturnModel's warm-start update.

        When the leaf was pretrained-injected, the frozen-leaf contract
        forbids mutating it: ``leaf.update()`` is skipped and only the
        strategy's own ``_training_metadata`` advances. Without this guard
        a forward-run loop silently drifts the "pinned" leaf fold by fold.

        See :meth:`IStrategy.update` for the shared contract.
        """
        metadata = self._assert_fitted_with_metadata(caller="update")
        new_metadata = metadata.extend_from(new_data)

        if _LEAF_KEY_RETURN_MODEL not in self._pretrained_leaves:
            new_returns = compute_log_returns(new_data["close"]).dropna()
            aligned = new_data.loc[new_returns.index]
            self._hybrid_return.update(aligned, new_returns, **kwargs)
        self._training_metadata = new_metadata

    def save(self, path: str | Path) -> None:
        """Persist ReturnForecast config + nested HybridReturn to ``path``.

        Strategy-specific kwargs (``position_scale``, ``max_leverage``) are
        written alongside every passthrough ``_HybridReturnParams`` field —
        the two together reconstruct the full ctor signature on load. Leaf
        device preference is NOT persisted (the hybrid subdir carries the
        fitted state; device re-resolves on load).
        """
        metadata = self._assert_fitted_with_metadata(caller="save")

        def write_weights(root: Path) -> None:
            self._hybrid_return.save(root / HYBRID_RETURN_SUBDIR)

        save_model_skeleton(
            path,
            config=self._ctor_kwargs_as_json(),
            training_metadata=metadata,
            write_weights=write_weights,
        )

    def _ctor_kwargs_as_json(self) -> dict[str, object]:
        """Snapshot of this strategy's constructor kwargs as JSON-ready values."""
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
            "interval": p.interval.value,
        }

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a trained ReturnForecastStrategy from ``path``.

        Narrow the strategy's ``config.json`` into ctor kwargs BEFORE loading
        the nested ``hybrid_return/`` subdir — a corrupt strategy config
        fast-fails with a named-field error, without wasting I/O on the
        HybridReturnModel's nested ARMA + LSTM + scaler loads.
        """
        root = Path(path)
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
            interval=Interval(json_io.get_str(config, "interval")),
        )
        instance._hybrid_return = HybridReturnModel.load(root / HYBRID_RETURN_SUBDIR)
        instance._training_metadata = TrainingMetadata.from_dict(metadata)
        instance._fitted = True
        return instance

    @property
    def name(self) -> str:
        return "ReturnForecast"

    @property
    def required_warmup_bars(self) -> int:
        return self._lstm_lookback

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Expose strategy + recursively-owned hybrid-return leaves (arma + lstm).

        When ``return_model`` was pretrained-injected, every entry from the
        hybrid's own walk gets ``is_pretrained=True`` so the walk-forward
        orchestrator enforces the strict-no-overlap invariant against the
        fold's train window (not just its test window).
        """
        leaf_metadata = self._hybrid_return.get_all_training_metadata()
        if _LEAF_KEY_RETURN_MODEL in self._pretrained_leaves:
            leaf_metadata = mark_pretrained(leaf_metadata)
        return collect_metadata(("strategy", self._training_metadata)) + leaf_metadata

    @staticmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
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
