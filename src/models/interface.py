"""Model abstract interfaces for predictors and classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class IPredictor(ABC):
    """Interface for all predictive models (volatility, price).

    Follows the same fit/transform pattern as IFeaturePipeline:
    fit on training data only, then predict on new data.
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, target: pd.Series) -> None:
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
    ) -> pd.Series:
        """Convenience: fit + predict on training data."""
        self.fit(train_data, target)
        return self.predict(train_data)


class IClassifier(ABC):
    """Interface for directional classifiers.

    Follows the same fit/predict pattern as IPredictor with
    consistent parameter naming: train_data, target.
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, target: pd.Series) -> None:
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
    ) -> pd.Series:
        """Convenience: fit + predict on training data."""
        self.fit(train_data, target)
        return self.predict(train_data)
