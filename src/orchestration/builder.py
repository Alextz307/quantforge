"""Config → wired :class:`Experiment` factory.

Resolves every :class:`ComponentConfig` against its global registry and
instantiates the concrete validator, engine, and slippage scenario. Kept
deliberately thin: composite-wiring logic (strategies that own their own
leaf models or feature pipelines) is handled inside each strategy's own
constructor, not reinvented here.
"""

from __future__ import annotations

from collections.abc import Callable

from src.core.config import ComponentConfig, ExperimentConfig
from src.core.registry import (
    data_source_registry,
    feature_registry,
    strategy_registry,
)
from src.core.temporal import WalkForwardValidator
from src.engine.cpp_engine import CppBacktestEngine
from src.engine.scenarios import SLIPPAGE_SCENARIOS
from src.features.interface import IFeaturePipeline
from src.orchestration.experiment import Experiment


def _make_feature_pipeline_factory(
    features_cfg: ComponentConfig,
) -> Callable[[], IFeaturePipeline]:
    """Capture ``features_cfg`` in a closure so callers get a fresh instance per call.

    Split out so the closure binds ``features_cfg`` once, not a loop variable —
    avoids the late-binding trap if this ever ends up inside a loop.
    """
    return lambda: feature_registry.create_from_config(features_cfg)


def build_experiment(cfg: ExperimentConfig) -> Experiment:
    """Instantiate every component referenced by ``cfg`` and bundle into an :class:`Experiment`."""
    data_source = data_source_registry.create_from_config(cfg.data.source)
    strategy = strategy_registry.create_from_config(cfg.strategy)
    feature_pipeline_factory: Callable[[], IFeaturePipeline] | None = (
        _make_feature_pipeline_factory(cfg.features) if cfg.features is not None else None
    )
    validator = WalkForwardValidator(
        n_splits=cfg.validation.n_splits,
        test_size=cfg.validation.test_size,
        gap=cfg.validation.gap,
        expanding=cfg.validation.expanding,
        snap_to_day=cfg.validation.snap_to_day,
    )
    engine = CppBacktestEngine()
    slippage = SLIPPAGE_SCENARIOS[cfg.slippage.scenario]

    return Experiment(
        config=cfg,
        data_source=data_source,
        strategy=strategy,
        validator=validator,
        engine=engine,
        slippage=slippage,
        feature_pipeline_factory=feature_pipeline_factory,
    )
