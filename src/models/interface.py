"""Model abstract interfaces for predictors and classifiers."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import pandas as pd

from src.core.temporal import TrackedMetadata, TrainingMetadata, collect_metadata


class IPredictor(ABC):
    """Interface for all predictive models (volatility, price).

    Follows the same fit/transform pattern as IFeaturePipeline:
    fit on training data only, then predict on new data.
    """

    @abstractmethod
    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Train the model on training data only.

        ``checkpoint_path`` is honored by NN-style leaves that benefit from
        best-state checkpointing (LSTM, XGBoost); statistical leaves
        (GARCH, ARMA) accept the kwarg for interface uniformity but ignore
        it. ``None`` is always a valid value and disables checkpointing.
        """

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions for the given data."""

    @abstractmethod
    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Predict a single value from a recent data window."""

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to a directory at ``path``.

        Subclasses write ``metadata.json`` + ``config.json`` + model-specific
        weights. Must raise if called before ``fit()`` and raise
        ``FileExistsError`` if ``path`` exists and is non-empty.

        The base implementation raises ``NotImplementedError`` so concrete
        models must override explicitly — a silent no-op would let tests pass
        without actually persisting anything.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.save() not implemented; fix by overriding "
            f"save() in the concrete predictor subclass."
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted model from a directory at ``path``.

        Must return a fully-fitted instance: ``predict()`` works immediately,
        ``training_metadata`` is populated from the saved ``metadata.json``.
        """
        raise NotImplementedError(
            f"{cls.__name__}.load() not implemented; fix by overriding load() "
            f"in the concrete predictor subclass."
        )

    def fit_predict(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> pd.Series:
        """Convenience: fit + predict on training data."""
        self.fit(train_data, target, checkpoint_path=checkpoint_path, **kwargs)
        return self.predict(train_data)

    # Subclasses inherit the unfitted state and must not re-declare this in
    # ``__init__``; :meth:`_set_fitted_with_metadata` is the only legal mutator,
    # and ``training_metadata is not None`` is the only fitted-state signal.
    _training_metadata: TrainingMetadata | None = None

    @property
    def training_metadata(self) -> TrainingMetadata | None:
        """Training period metadata, populated after fit()."""
        return getattr(self, "_training_metadata", None)

    def _assert_fitted_with_metadata(self, *, caller: str | None = None) -> TrainingMetadata:
        """Return ``self.training_metadata`` narrowed to non-None, raising otherwise.

        Canonical read-side guard — every method that requires a completed
        ``fit()`` (``predict``, ``predict_single``, ``save``, ``update``)
        calls this. Composite predictors that also need a leaf-presence check
        (e.g. ``self._scaler is None`` for a Hybrid model) layer that as a
        separate statement on top; this helper deliberately speaks only to
        the metadata invariant.

        ``caller`` defaults to the calling frame's function name (via
        ``sys._getframe``); pass it explicitly only when invoked from an
        inner closure or a wrapper whose name is not the right traceback
        anchor.
        """
        meta = self.training_metadata
        if meta is None:
            actual_caller = caller or sys._getframe(1).f_code.co_name
            raise RuntimeError(
                f"{type(self).__name__}.{actual_caller}() called before fit() "
                "completed; fix by calling model.fit(...) first."
            )
        return meta

    def _set_fitted_with_metadata(self, metadata: TrainingMetadata) -> None:
        """Atomic write-side counterpart to :meth:`_assert_fitted_with_metadata`.

        ``fit()`` and ``load()`` overrides call this as the very last step.
        ``training_metadata is not None`` is the sole fitted-state signal —
        no separate boolean flag exists, so atomicity reduces to "the slot is
        either ``None`` or a complete :class:`TrainingMetadata`". Refusing
        ``None`` keeps walk-forward's deep leakage check honest: a ``None``
        metadata combined with a "looks fitted" sentinel would let
        ``get_all_training_metadata`` return a ``None`` slot and the deep
        check would silently short-circuit.
        """
        if metadata is None:
            raise ValueError(
                f"{type(self).__name__}._set_fitted_with_metadata() requires a "
                "non-None TrainingMetadata; fix by passing the metadata produced "
                "by TrainingMetadata.from_fit(...) at the end of fit()."
            )
        self._training_metadata = metadata

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Every metadata object this predictor and its owned leaves expose.

        Default: just the predictor's own metadata tagged with the class name
        (lower-cased). Composites (HybridVolatility, HybridReturn) override to
        include each wrapped leaf (GARCH / ARMA / LSTM) with an explicit
        origin label so a caller's deep leakage check surfaces the exact
        component that drifted.
        """
        return collect_metadata((type(self).__name__, self.training_metadata))


class IClassifier(ABC):
    """Interface for directional classifiers.

    Follows the same fit/predict pattern as IPredictor with
    consistent parameter naming: train_data, target.
    """

    @abstractmethod
    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Train the classifier on training data only.

        ``checkpoint_path`` is honored by classifiers that support best-state
        checkpointing (XGBoost). ``None`` disables checkpointing.
        """

    @abstractmethod
    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """Predict class probabilities."""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict class labels."""

    def save(self, path: str | Path) -> None:
        """Persist the fitted classifier to a directory at ``path``.

        The base implementation raises ``NotImplementedError``; concrete
        classifiers must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.save() not implemented; fix by overriding "
            f"save() in the concrete classifier subclass."
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted classifier from a directory at ``path``."""
        raise NotImplementedError(
            f"{cls.__name__}.load() not implemented; fix by overriding load() "
            f"in the concrete classifier subclass."
        )

    def fit_predict(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> pd.Series:
        """Convenience: fit + predict on training data."""
        self.fit(train_data, target, checkpoint_path=checkpoint_path, **kwargs)
        return self.predict(train_data)

    _training_metadata: TrainingMetadata | None = None

    @property
    def training_metadata(self) -> TrainingMetadata | None:
        """Training period metadata, populated after fit()."""
        return getattr(self, "_training_metadata", None)

    def _assert_fitted_with_metadata(self, *, caller: str | None = None) -> TrainingMetadata:
        """Mirror of :meth:`IPredictor._assert_fitted_with_metadata` for classifiers."""
        meta = self.training_metadata
        if meta is None:
            actual_caller = caller or sys._getframe(1).f_code.co_name
            raise RuntimeError(
                f"{type(self).__name__}.{actual_caller}() called before fit() "
                "completed; fix by calling classifier.fit(...) first."
            )
        return meta

    def _set_fitted_with_metadata(self, metadata: TrainingMetadata) -> None:
        """Mirror of :meth:`IPredictor._set_fitted_with_metadata` for classifiers."""
        if metadata is None:
            raise ValueError(
                f"{type(self).__name__}._set_fitted_with_metadata() requires a "
                "non-None TrainingMetadata; fix by passing the metadata produced "
                "by TrainingMetadata.from_fit(...) at the end of fit()."
            )
        self._training_metadata = metadata

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Every metadata object this classifier and its owned leaves expose."""
        return collect_metadata((type(self).__name__, self.training_metadata))
