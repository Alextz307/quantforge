"""The wired, ready-to-run experiment primitive.

``Experiment`` is a frozen bundle of every component that participates in a
single walk-forward run: data source, strategy, validator, engine, slippage,
and an optional feature-pipeline FACTORY. It is produced by
:func:`src.orchestration.builder.build_experiment` from a validated
:class:`ExperimentConfig`.

Why the feature pipeline is a factory, not an instance
------------------------------------------------------
Feature pipelines (e.g. :class:`FeatureEngineeringPipeline`) enforce a
``fit_once`` guard on their scaler — a second ``fit()`` raises
``LeakageError``. A walk-forward run needs to fit the scaler PER FOLD on
``fold.train`` only; fitting once on the full dev region would leak later
folds' test-window statistics into earlier folds' features. A single
instance cannot satisfy both constraints. A factory closure captures the
config-derived kwargs and produces a fresh instance whenever the caller
asks — one per fold.

The strategy stays as an instance because each ``IStrategy.train()``
implementation is contracted to reset its own fit state from scratch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from quant_engine import SlippageConfig
from src.core.config import ExperimentConfig
from src.core.temporal import WalkForwardValidator
from src.data.interface import IDataSource
from src.engine.interface import IBacktestEngine
from src.features.interface import IFeaturePipeline
from src.orchestration.types import ExperimentResult
from src.strategies.interface import IStrategy


@dataclass(frozen=True)
class Experiment:
    """A fully-wired walk-forward experiment.

    Prefer constructing via :func:`build_experiment` — direct instantiation
    is intentional for tests that want to inject mocks per component.
    """

    config: ExperimentConfig
    data_source: IDataSource
    strategy: IStrategy
    validator: WalkForwardValidator
    engine: IBacktestEngine
    slippage: SlippageConfig
    feature_pipeline_factory: Callable[[], IFeaturePipeline] | None = None

    def run(self) -> ExperimentResult:
        raise NotImplementedError(
            f"{type(self).__name__}.run() not implemented — "
            f"build_experiment currently only assembles components."
        )
