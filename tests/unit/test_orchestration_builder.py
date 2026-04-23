"""Tests for :func:`build_experiment` — config → wired :class:`Experiment`."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from quant_engine import SlippageConfig
from src.core.config import ExperimentConfig
from src.core.temporal import WalkForwardValidator
from src.data.interface import IDataSource
from src.data.loader import YFinanceSource
from src.engine.cpp_engine import CppBacktestEngine
from src.features.interface import IFeaturePipeline
from src.features.pipeline import FeatureEngineeringPipeline
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import Experiment
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy

_START = datetime(2020, 1, 1)
_END = datetime(2023, 1, 1)
_N_SPLITS = 3
_TEST_SIZE = 100
_GAP = 3
_WINDOW = 25
_K = 1.8
_RSI_PERIOD = 11
# A k value the AdaptiveBollinger ctor rejects (k must be > 0) — used to prove
# that ctor params flow through the registry end-to-end rather than being
# silently dropped.
_INVALID_K = -1.0


def _cfg_dict(**overrides: Any) -> dict[str, Any]:
    base = {
        "name": "builder_test",
        "data": {
            "source": "yfinance",
            "tickers": ["SPY"],
            "start": _START,
            "end": _END,
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {"window": _WINDOW, "k": _K},
        },
        "validation": {
            "n_splits": _N_SPLITS,
            "test_size": _TEST_SIZE,
            "gap": _GAP,
        },
        "slippage": {"scenario": "normal"},
    }
    base.update(overrides)
    return base


class TestBuildExperiment:
    def test_returns_experiment(self) -> None:
        cfg = ExperimentConfig.model_validate(_cfg_dict())
        exp = build_experiment(cfg)

        assert isinstance(exp, Experiment)
        assert exp.config is cfg

    def test_data_source_instantiated(self) -> None:
        cfg = ExperimentConfig.model_validate(_cfg_dict())
        exp = build_experiment(cfg)

        assert isinstance(exp.data_source, IDataSource)
        assert isinstance(exp.data_source, YFinanceSource)

    def test_strategy_instantiated_by_name(self) -> None:
        cfg = ExperimentConfig.model_validate(_cfg_dict())
        exp = build_experiment(cfg)

        assert isinstance(exp.strategy, AdaptiveBollingerStrategy)

    def test_strategy_params_flow_through_ctor(self) -> None:
        """Garbage params must surface as the ctor's own ValueError, proving
        they travel through the registry dispatch untouched."""
        d = _cfg_dict()
        d["strategy"]["params"]["k"] = _INVALID_K
        cfg = ExperimentConfig.model_validate(d)
        with pytest.raises(ValueError, match="k must be > 0"):
            build_experiment(cfg)

    def test_validator_instantiated_with_params(self) -> None:
        cfg = ExperimentConfig.model_validate(_cfg_dict())
        exp = build_experiment(cfg)

        assert isinstance(exp.validator, WalkForwardValidator)
        assert exp.validator.n_splits == _N_SPLITS
        assert exp.validator.test_size == _TEST_SIZE
        assert exp.validator.gap == _GAP
        assert exp.validator.expanding is True

    def test_engine_instantiated(self) -> None:
        cfg = ExperimentConfig.model_validate(_cfg_dict())
        exp = build_experiment(cfg)

        assert isinstance(exp.engine, CppBacktestEngine)

    def test_slippage_resolves_to_config(self) -> None:
        cfg = ExperimentConfig.model_validate(_cfg_dict())
        exp = build_experiment(cfg)

        assert isinstance(exp.slippage, SlippageConfig)
        assert exp.slippage.base_bps == pytest.approx(1.0)

    def test_no_features_by_default(self) -> None:
        cfg = ExperimentConfig.model_validate(_cfg_dict())
        exp = build_experiment(cfg)
        assert exp.feature_pipeline_factory is None

    def test_feature_factory_wired_when_specified(self) -> None:
        d = _cfg_dict()
        d["features"] = {"name": "standard", "params": {"rsi_period": _RSI_PERIOD}}
        cfg = ExperimentConfig.model_validate(d)
        exp = build_experiment(cfg)

        assert exp.feature_pipeline_factory is not None
        pipeline = exp.feature_pipeline_factory()
        assert isinstance(pipeline, IFeaturePipeline)
        assert isinstance(pipeline, FeatureEngineeringPipeline)

    def test_feature_factory_yields_fresh_instance_each_call(self) -> None:
        """Per-fold fit_once contract: each factory call MUST return a new
        pipeline so the scaler fit_once guard doesn't fire across folds."""
        d = _cfg_dict()
        d["features"] = {"name": "standard", "params": {"rsi_period": _RSI_PERIOD}}
        cfg = ExperimentConfig.model_validate(d)
        exp = build_experiment(cfg)
        assert exp.feature_pipeline_factory is not None

        first = exp.feature_pipeline_factory()
        second = exp.feature_pipeline_factory()
        assert first is not second

    def test_run_is_callable(self) -> None:
        """``Experiment.run()`` behaviour is covered end-to-end by
        ``test_orchestration_experiment.py`` and the gated smoke test.
        This builder-level check only asserts the method isn't a leftover
        ``NotImplementedError`` stub.
        """
        cfg = ExperimentConfig.model_validate(_cfg_dict())
        exp = build_experiment(cfg)
        assert callable(exp.run)


class TestBuilderSurfacesRegistryErrors:
    def test_invalid_strategy_name_rejected_at_config_time(self) -> None:
        d = _cfg_dict()
        d["strategy"]["name"] = "Nonexistent"
        with pytest.raises(ValidationError, match="unknown strategy 'Nonexistent'"):
            ExperimentConfig.model_validate(d)
