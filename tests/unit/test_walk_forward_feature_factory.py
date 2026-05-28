"""
Tests for the ``feature_pipeline_factory`` kwarg on ``evaluate_walk_forward``.

The factory shape is load-bearing: a single pre-fit instance would either
leak scaler stats across folds or trip the fit-once scaler guard. These
tests pin the per-fold-refit contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_engine import SlippageConfig, SlippageModel
from src.core.temporal import TrainingMetadata, WalkForwardValidator
from src.core.types import Interval
from src.engine.cpp_engine import CppBacktestEngine
from src.engine.walk_forward import evaluate_walk_forward
from src.features.interface import IFeaturePipeline
from src.strategies.interface import IStrategy
from tests.conftest import make_synthetic_ohlcv_df

_N_SPLITS = 2
_TEST_SIZE = 30
_GAP = 1


@dataclass
class _RecordingPipeline(IFeaturePipeline):
    """
    Pipeline that records how many times fit() was called on THIS instance.

    Lets the test distinguish between "single instance refit per fold" (wrong)
    and "fresh instance per fold" (right).
    """

    fit_call_count: int = 0
    transform_call_count: int = 0

    def fit(self, train_data: pd.DataFrame) -> None:
        self.fit_call_count += 1

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.transform_call_count += 1
        # No-op pipeline: pass the bars straight through so the strategy
        # downstream still sees OHLCV and can generate signals normally.
        return data


@dataclass
class _FactoryRecorder:
    """
    Closure over ``instances`` so the test can inspect every pipeline built.
    """

    instances: list[_RecordingPipeline] = field(default_factory=list)

    def __call__(self) -> _RecordingPipeline:
        p = _RecordingPipeline()
        self.instances.append(p)
        return p


class _PassThroughStrategy(IStrategy):
    """
    Strategy that emits zero signals — tests the wiring, not the logic.
    """

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(train_data, Interval.DAILY, ("close",))
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=data.index, name="signal")

    @property
    def name(self) -> str:
        return "PassThrough"

    @property
    def required_warmup_bars(self) -> int:
        return 0

    @staticmethod
    def suggest_params(trial: object) -> dict[str, object]:
        return {}


def _slippage_zero() -> SlippageConfig:
    return SlippageConfig(
        model=SlippageModel.NoSlippage,
        base_bps=0.0,
        volume_impact_coeff=0.0,
    )


class TestFeatureFactory:
    def test_factory_invoked_once_per_fold(self) -> None:
        bars = make_synthetic_ohlcv_df()
        validator = WalkForwardValidator(n_splits=_N_SPLITS, test_size=_TEST_SIZE, gap=_GAP)
        recorder = _FactoryRecorder()
        evaluate_walk_forward(
            strategy=_PassThroughStrategy(),
            bars=bars,
            validator=validator,
            engine=CppBacktestEngine(),
            slippage=_slippage_zero(),
            interval=Interval.DAILY,
            feature_pipeline_factory=recorder,
        )
        assert len(recorder.instances) == _N_SPLITS

    def test_each_instance_fit_exactly_once(self) -> None:
        """
        No fit-once guard trips because every instance is fresh.
        """

        bars = make_synthetic_ohlcv_df()
        validator = WalkForwardValidator(n_splits=_N_SPLITS, test_size=_TEST_SIZE, gap=_GAP)
        recorder = _FactoryRecorder()
        evaluate_walk_forward(
            strategy=_PassThroughStrategy(),
            bars=bars,
            validator=validator,
            engine=CppBacktestEngine(),
            slippage=_slippage_zero(),
            interval=Interval.DAILY,
            feature_pipeline_factory=recorder,
        )
        for p in recorder.instances:
            assert p.fit_call_count == 1

    def test_transform_called_for_train_and_test(self) -> None:
        bars = make_synthetic_ohlcv_df()
        validator = WalkForwardValidator(n_splits=_N_SPLITS, test_size=_TEST_SIZE, gap=_GAP)
        recorder = _FactoryRecorder()
        evaluate_walk_forward(
            strategy=_PassThroughStrategy(),
            bars=bars,
            validator=validator,
            engine=CppBacktestEngine(),
            slippage=_slippage_zero(),
            interval=Interval.DAILY,
            feature_pipeline_factory=recorder,
        )
        for p in recorder.instances:
            assert p.transform_call_count == 2

    def test_default_no_factory_produces_same_fold_count(self) -> None:
        """
        No factory means no feature application; strategy sees raw bars.
        Smoke check that the existing default path still works."""

        bars = make_synthetic_ohlcv_df()
        validator = WalkForwardValidator(n_splits=_N_SPLITS, test_size=_TEST_SIZE, gap=_GAP)
        results = evaluate_walk_forward(
            strategy=_PassThroughStrategy(),
            bars=bars,
            validator=validator,
            engine=CppBacktestEngine(),
            slippage=_slippage_zero(),
            interval=Interval.DAILY,
        )
        assert len(results) == _N_SPLITS
