"""Strategy abstract interface with temporal contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
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

    @property
    def training_metadata(self) -> TrainingMetadata | None:
        """Training period metadata, populated after train()."""
        return getattr(self, "_training_metadata", None)

    def update(self, new_data: pd.DataFrame, **kwargs: object) -> None:
        """Incrementally update strategy models with new data. Default: full retrain."""
        self.train(new_data, **kwargs)
