"""Pairs trading strategy using Engle-Granger cointegration and z-score mean reversion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd

import quant_engine
from src.core import json_io
from src.core.persistence import (
    CONFIG_JSON,
    METADATA_JSON,
    TRAIN_PAIR_NPZ,
    WEIGHTS_JSON,
    save_model_skeleton,
)
from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.models.cointegration import CointegrationTester
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

_STRATEGY_NAME = "PairsTrading"


@strategy_registry.register(_STRATEGY_NAME)
class PairsTradingStrategy(IStrategy):
    """Pairs trading on a cointegrated spread via rolling z-score.

    Expects ``train_data`` / ``data`` with ``close_a`` and ``close_b`` columns.
    ``generate_signals()`` returns leg_a position in ``{-1, 0, +1}``; the
    backtest engine can derive leg_b position as
    ``-hedge_ratio * leg_a_position`` via the ``hedge_ratio`` property.
    """

    def __init__(
        self,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: float = 4.0,
        zscore_lookback: int = 60,
        p_value_threshold: float = 0.05,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if entry_zscore <= 0 or exit_zscore < 0 or stop_loss_zscore <= 0:
            raise ValueError("z-score thresholds must be positive")
        if exit_zscore >= entry_zscore:
            raise ValueError(f"exit_zscore ({exit_zscore}) must be < entry_zscore ({entry_zscore})")
        if stop_loss_zscore <= entry_zscore:
            raise ValueError(
                f"stop_loss_zscore ({stop_loss_zscore}) must be > entry_zscore ({entry_zscore})"
            )
        if zscore_lookback < 2:
            raise ValueError(f"zscore_lookback must be >= 2, got {zscore_lookback}")

        self._entry_zscore = entry_zscore
        self._exit_zscore = exit_zscore
        self._stop_loss_zscore = stop_loss_zscore
        self._zscore_lookback = zscore_lookback
        self._p_value_threshold = p_value_threshold
        self._interval = interval

        self._hedge_ratio = 0.0
        self._spread_mean = 0.0
        self._spread_std = 0.0
        self._is_cointegrated = False
        self._cpp_coint: quant_engine.CointegrationParams | None = None
        # Cached training price series (values only — cointegration refit
        # doesn't need the index). ``update()`` concatenates with new bars and
        # re-runs the Engle-Granger test on the extended window, so the
        # ``is_cointegrated`` flag may flip.
        self._train_close_a: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([])
        self._train_close_b: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([])
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None
        self._cpp_strategy = quant_engine.PairsTradingStrategy(
            quant_engine.PairsTradingStrategy.Config(
                entry_zscore=self._entry_zscore,
                exit_zscore=self._exit_zscore,
                stop_loss_zscore=self._stop_loss_zscore,
                zscore_lookback=self._zscore_lookback,
            )
        )

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Run Engle-Granger cointegration and cache hedge ratio / spread stats."""
        if "close_a" not in train_data.columns or "close_b" not in train_data.columns:
            raise ValueError(
                "PairsTradingStrategy.train() requires 'close_a' and 'close_b' columns"
            )

        result = CointegrationTester.engle_granger(
            train_data["close_a"],
            train_data["close_b"],
            self._p_value_threshold,
        )
        if not result.is_cointegrated:
            raise ValueError(
                f"Pair not cointegrated (p-value {result.p_value:.4f} "
                f">= {self._p_value_threshold:.4f})"
            )

        self._hedge_ratio = result.hedge_ratio
        self._spread_mean = result.spread_mean
        self._spread_std = result.spread_std
        self._is_cointegrated = True
        self._train_close_a = np.asarray(train_data["close_a"], dtype=np.float64)
        self._train_close_b = np.asarray(train_data["close_b"], dtype=np.float64)
        self._cpp_coint = self._build_coint_params()

        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, ("close_a", "close_b")
        )
        self._fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce {-1, 0, +1} leg_a position. Leading lookback bars are NaN."""
        if not self._fitted or self._cpp_coint is None:
            raise RuntimeError("PairsTradingStrategy.generate_signals() called before train()")
        if "close_a" not in data.columns or "close_b" not in data.columns:
            raise ValueError(
                "PairsTradingStrategy.generate_signals() requires 'close_a' and 'close_b' columns"
            )

        prices_a = np.asarray(data["close_a"], dtype=np.float64)
        prices_b = np.asarray(data["close_b"], dtype=np.float64)
        # The C++ rolling z-score uses a Welford accumulator that cannot
        # recover once any NaN enters — unlike pandas' rolling(w).std().
        # Fail loud at the boundary rather than silently emit all-NaN
        # signals from the first corrupted bar onward.
        if not np.isfinite(prices_a).all() or not np.isfinite(prices_b).all():
            raise ValueError(
                "PairsTradingStrategy.generate_signals() requires finite close_a / close_b "
                "(NaN or inf in price inputs would poison the rolling z-score)"
            )

        signal = self._cpp_strategy.generate_signals(
            prices_a=prices_a,
            prices_b=prices_b,
            coint=self._cpp_coint,
        )
        return pd.Series(signal, index=data.index, name="pairs_signal")

    def _build_coint_params(self) -> quant_engine.CointegrationParams:
        return quant_engine.CointegrationParams(
            hedge_ratio=self._hedge_ratio,
            spread_mean=self._spread_mean,
            spread_std=self._spread_std,
        )

    def update(self, new_data: pd.DataFrame, **kwargs: object) -> None:
        """Re-test cointegration on the extended window.

        Concatenates the cached training prices with ``new_data``'s
        ``close_a`` / ``close_b`` and reruns Engle-Granger. The
        ``is_cointegrated`` flag may flip to False on a regime change — the
        strategy will still generate signals (the state machine doesn't
        short-circuit on ``_is_cointegrated``); a logger warning fires on
        flip so the caller can pull the pair from the live book. See
        :meth:`IStrategy.update` for the shared contract.
        """
        metadata = self._assert_fitted_with_metadata(caller="update")
        if "close_a" not in new_data.columns or "close_b" not in new_data.columns:
            raise ValueError(
                "PairsTradingStrategy.update() requires 'close_a' and 'close_b' columns"
            )
        if len(new_data) < self._zscore_lookback:
            # Not a correctness issue for update() itself (cointegration runs on
            # the extended window), but a downstream ``generate_signals(new_data)``
            # call would produce all-NaN signals because the rolling z-score
            # needs at least ``zscore_lookback`` bars. Warn so the caller
            # notices rather than staring at a silent NaN stream.
            logger.warning(
                "PairsTradingStrategy.update(): new_data has %d bars, fewer than "
                "zscore_lookback=%d. generate_signals() on this window alone will "
                "emit NaN — prepend prior bars or pass a longer window to the "
                "next generate_signals() call.",
                len(new_data),
                self._zscore_lookback,
            )

        new_metadata = metadata.extend_from(new_data)

        new_a = np.asarray(new_data["close_a"], dtype=np.float64)
        new_b = np.asarray(new_data["close_b"], dtype=np.float64)
        combined_a = np.concatenate([self._train_close_a, new_a])
        combined_b = np.concatenate([self._train_close_b, new_b])

        # The Engle-Granger tester calls ``.values`` on inputs, so passing
        # plain Series without a meaningful index is enough — no need to
        # round-trip the timestamp index of the training bars.
        result = CointegrationTester.engle_granger(
            pd.Series(combined_a),
            pd.Series(combined_b),
            self._p_value_threshold,
        )
        if self._is_cointegrated and not result.is_cointegrated:
            logger.warning(
                "PairsTradingStrategy.update(): pair de-cointegrated after extension "
                "(p-value %.4f >= threshold %.4f). Strategy will keep generating "
                "signals — callers should inspect `is_cointegrated` before trading.",
                result.p_value,
                self._p_value_threshold,
            )
        self._hedge_ratio = result.hedge_ratio
        self._spread_mean = result.spread_mean
        self._spread_std = result.spread_std
        self._is_cointegrated = result.is_cointegrated
        self._train_close_a = combined_a
        self._train_close_b = combined_b
        self._cpp_coint = self._build_coint_params()
        self._training_metadata = new_metadata

    def save(self, path: str | Path) -> None:
        """Persist PairsTrading config + cointegration stats to ``path``."""
        metadata = self._assert_fitted_with_metadata(caller="save")

        def write_weights(root: Path) -> None:
            json_io.write(
                root / WEIGHTS_JSON,
                {
                    "hedge_ratio": self._hedge_ratio,
                    "spread_mean": self._spread_mean,
                    "spread_std": self._spread_std,
                    "is_cointegrated": self._is_cointegrated,
                },
            )
            # Training prices go into a single ``.npz`` for compact float64
            # storage (tighter + faster to load than JSON lists).
            np.savez(
                root / TRAIN_PAIR_NPZ,
                close_a=self._train_close_a,
                close_b=self._train_close_b,
            )

        save_model_skeleton(
            path,
            config=self._ctor_kwargs_as_json(),
            training_metadata=metadata,
            write_weights=write_weights,
        )

    def _ctor_kwargs_as_json(self) -> dict[str, object]:
        """Snapshot of this strategy's constructor kwargs as JSON-ready values.

        Single source of truth for the save-time config — the load path reads
        the same keys back. Exercised by a parametrized drift test that
        compares these keys against ``__init__``'s parameter set.
        """
        return {
            "entry_zscore": self._entry_zscore,
            "exit_zscore": self._exit_zscore,
            "stop_loss_zscore": self._stop_loss_zscore,
            "zscore_lookback": self._zscore_lookback,
            "p_value_threshold": self._p_value_threshold,
            "interval": self._interval.value,
        }

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a trained PairsTradingStrategy from ``path``."""
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        weights = json_io.read_dict(root / WEIGHTS_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            entry_zscore=json_io.get_float(config, "entry_zscore"),
            exit_zscore=json_io.get_float(config, "exit_zscore"),
            stop_loss_zscore=json_io.get_float(config, "stop_loss_zscore"),
            zscore_lookback=json_io.get_int(config, "zscore_lookback"),
            p_value_threshold=json_io.get_float(config, "p_value_threshold"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        instance._hedge_ratio = json_io.get_float(weights, "hedge_ratio")
        instance._spread_mean = json_io.get_float(weights, "spread_mean")
        instance._spread_std = json_io.get_float(weights, "spread_std")
        instance._is_cointegrated = json_io.get_bool(weights, "is_cointegrated")
        with np.load(root / TRAIN_PAIR_NPZ, allow_pickle=False) as train_pair:
            instance._train_close_a = np.asarray(train_pair["close_a"], dtype=np.float64)
            instance._train_close_b = np.asarray(train_pair["close_b"], dtype=np.float64)
        instance._cpp_coint = instance._build_coint_params()
        instance._training_metadata = TrainingMetadata.from_dict(metadata)
        instance._fitted = True
        return instance

    @property
    def hedge_ratio(self) -> float:
        """Cointegration hedge ratio (slope of OLS regression of a on b)."""
        if not self._fitted:
            raise RuntimeError("PairsTradingStrategy.hedge_ratio accessed before train()")
        return self._hedge_ratio

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

    @property
    def required_warmup_bars(self) -> int:
        return self._zscore_lookback

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for PairsTrading hyperparameters."""
        return {
            "entry_zscore": trial.suggest_float("pairs_entry_z", 1.5, 3.0),
            "exit_zscore": trial.suggest_float("pairs_exit_z", 0.0, 1.0),
            "stop_loss_zscore": trial.suggest_float("pairs_stop_z", 3.5, 5.0),
            "zscore_lookback": trial.suggest_int("pairs_lookback", 30, 120),
        }
