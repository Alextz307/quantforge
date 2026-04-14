"""Model abstract interfaces for predictors and classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
        """Incrementally update model with new data. Default: full refit."""
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
        """Incrementally update classifier with new data. Default: full refit."""
        self.fit(new_data, target, **kwargs)
