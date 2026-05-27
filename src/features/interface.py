"""Feature pipeline abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class IFeaturePipeline(ABC):
    """Every feature pipeline must respect temporal boundaries.

    Fit on training data only, then transform test data using
    learned parameters (e.g., scaler statistics).
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """Learn parameters from TRAINING data only."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply learned parameters. Must not peek beyond each row's timestamp."""

    def fit_transform(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit + transform on training data."""

        self.fit(train_data)
        return self.transform(train_data)
