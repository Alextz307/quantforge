"""Strategy abstract interface with temporal contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import pandas as pd

from src.core.temporal import TrackedMetadata, TrainingMetadata, collect_metadata

if TYPE_CHECKING:
    import optuna


class IStrategy(ABC):
    """Every strategy MUST implement this interface.

    The train/generate split enforces that signal generation
    never accesses data that wasn't available at signal time.
    The engine will shift signals by 1 day automatically —
    strategies must NOT shift themselves.
    """

    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Learn parameters from historical data. Called once before backtesting.

        ``checkpoint_path`` is forwarded to NN-style leaves for best-state
        checkpointing during ``fit()``; strategies without such leaves
        accept the kwarg and ignore it. ``None`` disables checkpointing.
        """

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

    is_pairs_strategy: ClassVar[bool] = False
    """True for two-leg (cointegration / pairs) strategies.

    The walk-forward dispatcher reads this to decide whether to call
    ``engine.run`` (single-leg) or ``engine.run_pairs`` (two-leg, requires
    ``hedge_ratio`` + a wide-format bar frame).
    """

    # Subclasses inherit the unfitted state and must not re-declare these
    # in ``__init__``; :meth:`_set_fitted_with_metadata` is the only legal mutator.
    _fitted: bool = False
    _training_metadata: TrainingMetadata | None = None

    @property
    def hedge_ratio(self) -> float:
        """Cointegration hedge ratio for two-leg backtests.

        Only pairs strategies need to override; the default raises so a
        misconfigured single-leg strategy doesn't silently report a 0.0
        hedge to the pairs engine.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.hedge_ratio is only defined for pairs "
            f"strategies; fix by setting is_pairs_strategy=True and "
            f"overriding hedge_ratio if this strategy is two-legged."
        )

    @staticmethod
    @abstractmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
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

        ``save()`` overrides use this to collapse the
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

    def _set_fitted_with_metadata(self, metadata: TrainingMetadata) -> None:
        """Atomic write-side counterpart to :meth:`_assert_fitted_with_metadata`.

        ``train()`` and ``load()`` overrides call this as the very last step,
        instead of separately assigning ``self._training_metadata`` and
        ``self._fitted = True``. Two invariants the helper enforces that the
        previous two-line idiom did not:

        * **Metadata is non-None** — a ``None`` metadata with ``_fitted=True``
          would let walk-forward's deep leakage check silently skip a leaf
          (``get_all_training_metadata`` returns the strategy's own ``None``
          slot, ``validate_no_overlap`` short-circuits).
        * **Order is metadata-then-flag** — assigning ``_fitted=True`` before
          ``_training_metadata`` is set leaves a half-fitted state visible to
          any concurrent reader (or to a subclass whose own assignment raises
          between the two lines).

        Subclasses must use this helper rather than the two-line idiom so the
        atomic invariant is upheld at every callsite. The read side is still
        :meth:`_assert_fitted_with_metadata`; together the pair is the only
        legal way to commit / observe the fitted state.
        """
        if metadata is None:
            raise ValueError(
                f"{type(self).__name__}._set_fitted_with_metadata() requires a "
                "non-None TrainingMetadata; fix by passing the metadata produced "
                "by TrainingMetadata.from_fit(...) at the end of train()."
            )
        self._training_metadata = metadata
        self._fitted = True

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Every metadata object a fold's leakage check must validate.

        Default: just the strategy's own metadata. Composite strategies
        override to include each wrapped model's metadata tagged with an
        ``origin`` label so a ``LeakageError`` names the exact component
        that drifted (strategy vs garch vs lstm vs classifier).
        """
        return collect_metadata(("strategy", self.training_metadata))

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
        raise NotImplementedError(
            f"{type(self).__name__}.save() not implemented; fix by overriding "
            f"save() in the concrete strategy subclass."
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a trained strategy from a directory at ``path``.

        Must return a fully-trained instance: ``generate_signals()`` works
        immediately, ``training_metadata`` is populated from the saved
        ``metadata.json``.
        """
        raise NotImplementedError(
            f"{cls.__name__}.load() not implemented; fix by overriding load() "
            f"in the concrete strategy subclass."
        )
