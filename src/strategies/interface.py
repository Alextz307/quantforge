"""Strategy abstract interface with temporal contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


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
