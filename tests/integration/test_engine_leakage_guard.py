"""Anti-leakage tripwire: ``evaluate_walk_forward`` raises ``LeakageError``
when a strategy reports ``training_metadata`` that overlaps the test fold.

Defense-in-depth: ``WalkForwardValidator`` already yields non-overlapping
TemporalSplits, but a buggy strategy could legitimately train on more
data than ``fold.train`` (e.g., look up an external panel during fit).
The tripwire catches that *behavior* via ``training_metadata`` —
strategies don't lie about the fact that they trained, they just have
to be honest about *which window* they trained on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.engine import (
    SLIPPAGE_SCENARIOS,
    CppBacktestEngine,
    SlippageScenario,
    evaluate_walk_forward,
)
from src.strategies.interface import IStrategy
from tests.conftest import make_synthetic_ohlcv_df, make_walk_forward_validator

LEAKAGE_N_ROWS = 400
N_SPLITS = 2
TEST_SIZE = 100
NORMAL_SCENARIO = SLIPPAGE_SCENARIOS[SlippageScenario.NORMAL]


class _LyingStrategy(IStrategy):
    """Reports a ``TrainingMetadata`` covering the full panel, not just
    ``fold.train`` — simulates a buggy strategy that touched out-of-fold
    data during ``train()``.
    """

    def __init__(self, full_panel_metadata: TrainingMetadata) -> None:
        self._training_metadata = full_panel_metadata

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        pass  # metadata is preset in __init__

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=data.index, name="lying_signal")

    @property
    def name(self) -> str:
        return "Lying"

    @property
    def required_warmup_bars(self) -> int:
        return 0


class _ForgetfulStrategy(IStrategy):
    """``train()`` runs but never populates ``_training_metadata``."""

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        pass

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=data.index)

    @property
    def name(self) -> str:
        return "Forgetful"

    @property
    def required_warmup_bars(self) -> int:
        return 0


def test_orchestrator_raises_on_overlapping_training_metadata() -> None:
    bars = make_synthetic_ohlcv_df(n_rows=LEAKAGE_N_ROWS)
    full_panel_metadata = TrainingMetadata.from_fit(bars, Interval.DAILY, ("close",))
    strategy = _LyingStrategy(full_panel_metadata=full_panel_metadata)

    with pytest.raises(LeakageError, match="Evaluation data starts at"):
        evaluate_walk_forward(
            strategy,
            bars,
            make_walk_forward_validator(N_SPLITS, TEST_SIZE),
            CppBacktestEngine(),
            NORMAL_SCENARIO,
            Interval.DAILY,
        )


def test_orchestrator_raises_when_training_metadata_missing() -> None:
    """Contract enforcement: train() must populate training_metadata."""
    bars = make_synthetic_ohlcv_df(n_rows=LEAKAGE_N_ROWS)

    with pytest.raises(RuntimeError, match="did not populate training_metadata"):
        evaluate_walk_forward(
            _ForgetfulStrategy(),
            bars,
            make_walk_forward_validator(N_SPLITS, TEST_SIZE),
            CppBacktestEngine(),
            NORMAL_SCENARIO,
            Interval.DAILY,
        )
