"""
Strategy abstract interface with temporal contract.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Self

import pandas as pd

from src.core.temporal import TrackedMetadata, TrainingMetadata, collect_metadata

if TYPE_CHECKING:
    import optuna

RECURSIVE_LEAF_CONVERGENCE_MARGIN_BARS = 100


class IStrategy(ABC):
    """
    Every strategy MUST implement this interface.

    The train/generate split enforces that signal generation
    never accesses data that wasn't available at signal time.
    The engine will shift signals by 1 day automatically -
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
        """
        Learn parameters from historical data. Called once before backtesting.

        ``checkpoint_path`` is forwarded to NN-style leaves for best-state
        checkpointing during ``fit()``; strategies without such leaves
        accept the kwarg and ignore it. ``None`` disables checkpointing.
        """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate position signals for the given data.

        Returns a Series of position values. The engine shifts these
        by 1 bar automatically - do NOT shift inside strategy logic.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Strategy identifier.
        """

    @property
    @abstractmethod
    def required_warmup_bars(self) -> int:
        """
        Number of initial bars needed before signals are valid.
        """

    uses_xgboost: ClassVar[bool] = False
    """True for strategies whose ML leaf is XGBoost-backed (no MPS GPU path)."""

    convergence_margin_bars: ClassVar[int] = 0
    """Extra bars beyond :attr:`required_warmup_bars` to let recursive leaves converge.

    Pure rolling-window strategies (signals depend only on a bounded prefix of
    bars once each indicator has warmed up) keep the default ``0`` - their
    last-row signal is byte-identical for any input window longer than
    ``required_warmup_bars``. Strategies with GARCH or ARMA leaves carry
    recursive state whose forward iteration must converge out of the fitted
    backcast before the last-row signal stabilises; these strategies override
    to :data:`RECURSIVE_LEAF_CONVERGENCE_MARGIN_BARS` (geometric convergence
    of the recursion's effective memory is well within that bound for fitted
    equity-return models). The deployment layer reads this on auto-derive to
    size the live warmup window without a user-supplied constant.
    """

    is_pairs_strategy: ClassVar[bool] = False
    """True for two-leg (cointegration / pairs) strategies.

    The walk-forward dispatcher reads this to decide whether to call
    ``engine.run`` (single-leg) or ``engine.run_pairs`` (two-leg, requires
    ``hedge_ratio`` + a wide-format bar frame). Mutually exclusive with
    :attr:`is_multi_feature_strategy` - the validator rejects classes that
    set both to ``True``.
    """

    is_multi_feature_strategy: ClassVar[bool] = False
    """True for single-asset traded strategies that read N feature tickers.

    A multi-feature strategy trades exactly one asset (the
    :attr:`primary_ticker`) but its signal computation reads a wide-format
    frame whose columns follow the ``<ohlcv>_<TICKER>`` suffix convention
    (e.g. ``close_SPY``, ``close_QQQ``). The walk-forward dispatcher slices
    the primary asset's OHLCV out of the wide frame before calling
    ``engine.run`` - companion tickers never enter the engine's books.
    Mutually exclusive with :attr:`is_pairs_strategy`.
    """

    _training_metadata: TrainingMetadata | None = None

    @property
    def hedge_ratio(self) -> float:
        """
        Cointegration hedge ratio for two-leg backtests.

        Only pairs strategies need to override; the default raises so a
        misconfigured single-leg strategy doesn't silently report a 0.0
        hedge to the pairs engine.
        """

        raise NotImplementedError(
            f"{type(self).__name__}.hedge_ratio is only defined for pairs "
            f"strategies; fix by setting is_pairs_strategy=True and "
            f"overriding hedge_ratio if this strategy is two-legged."
        )

    @property
    def primary_ticker(self) -> str:
        """
        The single asset a multi-feature strategy trades and reports PnL against.

        Only multi-feature strategies need to override; the default raises so
        a misconfigured single-asset / pairs strategy doesn't silently route
        the wrong column through the slicer. The dispatcher uses this name to
        slice ``<ohlcv>_<primary_ticker>`` out of the wide frame; the value
        MUST appear in the experiment's ``data.tickers`` list (validated in
        :func:`src.orchestration.builder._validate_strategy_data_shape`).
        """

        raise NotImplementedError(
            f"{type(self).__name__}.primary_ticker is only defined for "
            f"multi-feature strategies; fix by setting "
            f"is_multi_feature_strategy=True and overriding primary_ticker "
            f"if this strategy reads a wide multi-ticker frame."
        )

    @staticmethod
    @abstractmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
        """
        Optuna search space for this strategy's ctor kwargs.

        Every strategy declares the joint feature / model / strategy
        hyperparameters it wants tuned - leaf knobs that pass through to
        wrapped models (e.g. ``arma_p_max`` on ReturnForecast) are
        flattened here, not resolved via a separate leaf ``suggest_params``
        call. The ``StrategyTuner`` merges the returned dict into
        ``ExperimentConfig.strategy.params`` per trial.
        """

    @property
    def training_metadata(self) -> TrainingMetadata | None:
        """
        Training period metadata, populated after train().
        """

        return getattr(self, "_training_metadata", None)

    def _assert_fitted_with_metadata(self, *, caller: str | None = None) -> TrainingMetadata:
        """
        Return ``self.training_metadata`` narrowed to non-None, raising otherwise.

        Canonical read-side guard for fitted state - every method that
        requires a completed ``train()`` (``generate_signals``, ``save``,
        ``hedge_ratio``) calls this.

        ``caller`` defaults to the calling frame's function name (via
        ``sys._getframe`` - CPython + PyPy supported, ~1us lookup paid
        only on the error path). Pass it explicitly if the helper is
        invoked from an inner closure or a wrapper whose name is not
        what you want surfaced in tracebacks.
        """

        meta = self.training_metadata
        if meta is None:
            actual_caller = caller or sys._getframe(1).f_code.co_name
            raise RuntimeError(
                f"{type(self).__name__}.{actual_caller}() called before train() "
                "completed; fix by calling strategy.train(df) first."
            )
        return meta

    def _set_fitted_with_metadata(self, metadata: TrainingMetadata) -> None:
        """
        Atomic write-side counterpart to :meth:`_assert_fitted_with_metadata`.

        ``train()`` and ``load()`` overrides call this as the very last step.
        Refusing ``None`` keeps walk-forward's deep leakage check honest:
        a ``None`` metadata combined with a "looks fitted" sentinel would let
        ``get_all_training_metadata`` return a ``None`` slot and the deep
        check would silently short-circuit.

        ``training_metadata is not None`` is the sole fitted-state signal -
        no separate boolean flag exists, so atomicity reduces to "the slot is
        either ``None`` or a complete :class:`TrainingMetadata`".
        """

        if metadata is None:
            raise ValueError(
                f"{type(self).__name__}._set_fitted_with_metadata() requires a "
                "non-None TrainingMetadata; fix by passing the metadata produced "
                "by TrainingMetadata.from_fit(...) at the end of train()."
            )
        self._training_metadata = metadata

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """
        Every metadata object a fold's leakage check must validate.

        Default: just the strategy's own metadata. Composite strategies
        override to include each wrapped model's metadata tagged with an
        ``origin`` label so a ``LeakageError`` names the exact component
        that drifted (strategy vs garch vs lstm vs classifier).
        """

        return collect_metadata(("strategy", self.training_metadata))

    def get_fold_diagnostics(self) -> Mapping[str, float]:
        """
        Per-fold diagnostic scalars surfaced from the most recent predict.

        Default: empty mapping. Walk-forward calls this once per fold
        AFTER ``generate_signals()`` and bundles the result into
        :class:`FoldResult.strategy_diagnostics` so the experiment
        manifest persists it. Composite strategies that track in-flight
        diagnostics (e.g. ``VolatilityTargeting``'s ``floor_bind_fraction``)
        override to return them.
        """

        return MappingProxyType({})

    def feature_columns(self) -> tuple[str, ...]:
        """
        Engineered feature columns this strategy's model consumes.

        Default empty: rule-based strategies (Bollinger, Pairs) consume no
        engineered features, so the feature-importance subsystem skips them.
        Feature-consuming strategies override to return the deterministic
        column names their leaf model was trained on - the set the permutation
        driver shuffles one column at a time. None of these names may be
        ``close`` (the permutation must leave the realised-target basis intact).
        """

        return ()

    def feature_importance_frame(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """
        Build the frame the importance driver permutes and scores.

        Must contain every column in :meth:`feature_columns` plus a raw
        ``close`` column (the basis for the realised target). Two shapes:

        * Strategies whose model already reads engineered columns straight
          from the input frame (the ARMA/GARCH hybrids) return ``data``
          unchanged - the features and raw OHLC are already present.
        * Strategies that compute features internally from raw OHLCV (the
          classifier strategies) materialise their feature matrix here and
          attach a raw ``close`` column, so the driver can shuffle the
          engineered columns the model actually consumes.

        Default ``None``: the strategy is skipped by the importance subsystem.
        """

        return None

    def feature_importance_score(self, frame: pd.DataFrame) -> float | None:
        """
        Higher-is-better score of the model's predictions on ``frame``.

        ``frame`` is a (possibly column-permuted) frame from
        :meth:`feature_importance_frame`. Out-of-sample permutation importance
        re-scores it with one feature column shuffled and attributes the score
        drop to that feature. The score MUST reflect what the model predicts
        (directional hit-rate for return / probability models; negative QLIKE
        for the volatility forecaster) and derive its realised target only
        from ``frame['close']`` (never a feature column), so it stays invariant
        to feature permutation. Default ``None``: the strategy is skipped.
        """

        return None

    def feature_gain(self) -> Mapping[str, float] | None:
        """
        Native booster gain per feature column, for tree-model strategies only.

        Default ``None``: non-tree strategies (hybrids, rule-based) have no
        gain map. XGBoost-backed strategies override to return one entry per
        feature column (zero-filled for features the booster never split on).
        """

        return None

    def save(self, path: str | Path) -> None:
        """
        Persist the trained strategy to a directory at ``path``.

        Subclasses write ``metadata.json`` + ``config.json`` + any owned
        model subdirectories (e.g. ``<path>/garch/``). Must raise if called
        before ``train()`` and raise ``FileExistsError`` if ``path`` exists
        and is non-empty.

        The base implementation raises ``NotImplementedError`` so concrete
        strategies must override explicitly - a silent no-op would let tests
        pass without actually persisting anything.
        """

        raise NotImplementedError(
            f"{type(self).__name__}.save() not implemented; fix by overriding "
            f"save() in the concrete strategy subclass."
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Reconstruct a trained strategy from a directory at ``path``.

        Must return a fully-trained instance: ``generate_signals()`` works
        immediately, ``training_metadata`` is populated from the saved
        ``metadata.json``.
        """

        raise NotImplementedError(
            f"{cls.__name__}.load() not implemented; fix by overriding load() "
            f"in the concrete strategy subclass."
        )
