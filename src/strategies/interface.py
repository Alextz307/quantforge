"""Strategy abstract interface with temporal contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd

from src.core.temporal import TrackedMetadata, collect_metadata

if TYPE_CHECKING:
    import optuna

    from src.core.temporal import TrainingMetadata


class IStrategy(ABC):
    """Every strategy MUST implement this interface.

    The train/generate split enforces that signal generation
    never accesses data that wasn't available at signal time.
    The engine will shift signals by 1 day automatically —
    strategies must NOT shift themselves.
    """

    @abstractmethod
    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Learn parameters from historical data. Called once before backtesting."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate position signals for the given data.

        Returns a Series of position values. The engine shifts these
        by 1 bar automatically — do NOT shift inside strategy logic.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""

    @property
    @abstractmethod
    def required_warmup_bars(self) -> int:
        """Number of initial bars needed before signals are valid."""

    @staticmethod
    @abstractmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for this strategy's ctor kwargs.

        Every strategy declares the joint feature / model / strategy
        hyperparameters it wants tuned — leaf knobs that pass through to
        wrapped models (e.g. ``arma_p_max`` on ReturnForecast) are
        flattened here, not resolved via a separate leaf ``suggest_params``
        call. The ``StrategyTuner`` merges the returned dict into
        ``ExperimentConfig.strategy.params`` per trial, with any keys
        owned by pinned pretrained leaves filtered out by
        :func:`src.optimization.sampling.sample_trial_params`.
        """

    @property
    def training_metadata(self) -> TrainingMetadata | None:
        """Training period metadata, populated after train()."""
        return getattr(self, "_training_metadata", None)

    def _assert_fitted_with_metadata(self, *, caller: str) -> TrainingMetadata:
        """Return ``self.training_metadata`` narrowed to non-None, raising otherwise.

        ``save()`` / ``update()`` overrides use this to collapse the
        ``if self._training_metadata is None: raise ...`` guard + type-narrowing
        dance into one call. The ``caller`` name reaches the error message so
        tracebacks still point at the specific method.

        ``_fitted`` remains a separate check on subclasses that maintain it —
        it tracks a different invariant (weights / scaler / leaf-model presence)
        that this helper deliberately does not touch.
        """
        meta = self.training_metadata
        if meta is None:
            raise RuntimeError(
                f"{type(self).__name__}.{caller}() called before train() "
                "completed; fix by calling strategy.train(df) first."
            )
        return meta

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Every metadata object a fold's leakage check must validate.

        Default: just the strategy's own metadata. Composite strategies
        override to include each wrapped model's metadata tagged with an
        ``origin`` label so a ``LeakageError`` names the exact component
        that drifted (strategy vs garch vs lstm vs classifier).
        """
        return collect_metadata(("strategy", self.training_metadata))

    def update(self, new_data: pd.DataFrame, **kwargs: object) -> None:
        """Incrementally update the trained strategy with a new window.

        Default: full retrain. Subclasses override by delegating to the
        internal model's ``update()`` (or, for ``PairsTradingStrategy``,
        re-testing cointegration on the extended window). Every override is
        transactional — ``_training_metadata`` is produced via
        ``extend_from(new_data)`` *before* any leaf mutation so a
        ``LeakageError`` on overlapping ``new_data`` leaves the strategy
        untouched.

        Post-update invariants: ``fit_timestamp`` stays frozen (provenance —
        only ``train()`` sets it); ``train_end`` and ``n_train_samples``
        advance.
        """
        self.train(new_data, **kwargs)

    def save(self, path: str | Path) -> None:
        """Persist the trained strategy to a directory at ``path``.

        Subclasses write ``metadata.json`` + ``config.json`` + any owned
        model subdirectories (e.g. ``<path>/garch/``). Must raise if called
        before ``train()`` and raise ``FileExistsError`` if ``path`` exists
        and is non-empty.

        The base implementation raises ``NotImplementedError`` so concrete
        strategies must override explicitly — a silent no-op would let tests
        pass without actually persisting anything.
        """
        raise NotImplementedError(f"{type(self).__name__}.save() not implemented")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a trained strategy from a directory at ``path``.

        Must return a fully-trained instance: ``generate_signals()`` works
        immediately, ``training_metadata`` is populated from the saved
        ``metadata.json``.
        """
        raise NotImplementedError(f"{cls.__name__}.load() not implemented")
