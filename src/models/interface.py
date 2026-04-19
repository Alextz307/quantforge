"""Model abstract interfaces for predictors and classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd

if TYPE_CHECKING:
    from src.core.temporal import TrainingMetadata


class IPredictor(ABC):
    """Interface for all predictive models (volatility, price).

    Follows the same fit/transform pattern as IFeaturePipeline:
    fit on training data only, then predict on new data.
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, target: pd.Series, **kwargs: object) -> None:
        """Train the model on training data only."""

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
        raise NotImplementedError(f"{type(self).__name__}.save() not implemented")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted model from a directory at ``path``.

        Must return a fully-fitted instance: ``predict()`` works immediately,
        ``training_metadata`` is populated from the saved ``metadata.json``.
        """
        raise NotImplementedError(f"{cls.__name__}.load() not implemented")

    def fit_predict(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> pd.Series:
        """Convenience: fit + predict on training data."""
        self.fit(train_data, target, **kwargs)
        return self.predict(train_data)

    @property
    def training_metadata(self) -> TrainingMetadata | None:
        """Training period metadata, populated after fit()."""
        return getattr(self, "_training_metadata", None)

    def update(self, new_data: pd.DataFrame, target: pd.Series, **kwargs: object) -> None:
        """Incrementally update the fitted model with a new window.

        Default: full refit. Subclasses override with warm-start behavior
        (GARCH: skip AIC grid search; ARMA: fixed-order refit; LSTM: fine-tune;
        XGBoost: continue-boost). Every override is transactional —
        ``_training_metadata`` is produced via ``extend_from(new_data)`` *before*
        any mutation so a ``LeakageError`` on overlapping ``new_data`` leaves
        the model untouched.

        The ``new_data`` contract differs per leaf: GARCH/ARMA concatenate it
        with cached training targets and refit on the combined series, while
        LSTM/XGBoost consume only the new bars (fine-tune / continue-boost)
        without re-visiting earlier samples. See each override's docstring.

        Post-update invariants: ``fit_timestamp`` stays frozen (provenance —
        only ``fit()`` sets it); ``train_end`` and ``n_train_samples`` advance.
        """
        self.fit(new_data, target, **kwargs)


class IClassifier(ABC):
    """Interface for directional classifiers.

    Follows the same fit/predict pattern as IPredictor with
    consistent parameter naming: train_data, target.
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, target: pd.Series, **kwargs: object) -> None:
        """Train the classifier on training data only."""

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
        raise NotImplementedError(f"{type(self).__name__}.save() not implemented")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted classifier from a directory at ``path``."""
        raise NotImplementedError(f"{cls.__name__}.load() not implemented")

    def fit_predict(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> pd.Series:
        """Convenience: fit + predict on training data."""
        self.fit(train_data, target, **kwargs)
        return self.predict(train_data)

    @property
    def training_metadata(self) -> TrainingMetadata | None:
        """Training period metadata, populated after fit()."""
        return getattr(self, "_training_metadata", None)

    def update(self, new_data: pd.DataFrame, target: pd.Series, **kwargs: object) -> None:
        """Incrementally update the fitted classifier with a new window.

        Default: full refit. See :meth:`IPredictor.update` for the shared
        contract around transactional ``extend_from(new_data)`` validation
        and the ``fit_timestamp`` / ``train_end`` / ``n_train_samples``
        post-update invariants.
        """
        self.fit(new_data, target, **kwargs)
