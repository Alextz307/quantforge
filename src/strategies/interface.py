"""Strategy abstract interface with temporal contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Self

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
