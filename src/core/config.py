"""Pydantic v2 config schema for end-to-end experiments.

``ExperimentConfig`` is the frozen, validated root object produced from a
YAML file via :func:`load_experiment_config`. ``build_experiment`` consumes
it to wire up a fully-instantiated ``Experiment``. Every field is strictly
typed; ``extra='forbid'`` across the tree prevents silent typos in user
YAML.

Registry name validators run at load time so a YAML referring to an
unregistered strategy / data source / feature pipeline fails with a pointed
error naming the available alternatives, not buried later under a stack
trace from ``KeyError`` in ``ComponentRegistry.get``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.registry import (
    data_source_registry,
    feature_registry,
    strategy_registry,
)
from src.core.types import Interval
from src.engine.scenarios import SlippageScenario


def _ensure_registries_populated() -> None:
    """Import the three component packages so their registry decorators run.

    Each package's ``__init__.py`` walks its own directory and imports every
    non-private, non-``interface`` module — so a new strategy / data source /
    feature pipeline file registers itself automatically, with no edits here.

    Deferred out of module scope because these package imports transitively
    pull in torch / xgboost / arch / statsmodels / pmdarima (~4 s on a cold
    interpreter). Consumers that only reference ``ExperimentConfig`` /
    ``FoldRecord`` as types pay zero ML-framework cost. Python caches module
    imports in ``sys.modules``, so repeat calls are effectively free.
    """
    import src.data  # noqa: F401
    import src.features  # noqa: F401
    import src.strategies  # noqa: F401


class ComponentConfig(BaseModel):
    """Generic pluggable-component spec: a registry name plus its kwargs.

    ``params`` values are typed as ``object`` because YAML values arrive
    untyped (str / int / float / bool / list / dict) and their concrete
    types are enforced downstream by each component's own ``__init__``
    validation. A stricter shape here would duplicate that check and
    reject legitimate per-component types (e.g. ``Interval`` enum values).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    params: dict[str, object] = Field(default_factory=dict)


class DataConfig(BaseModel):
    """Data-fetch spec: which source, tickers, date range, and bar interval.

    ``source`` is a :class:`ComponentConfig` so per-source kwargs (e.g.
    ``data_dir`` for the CSV source) flow through cleanly. A bare string in
    YAML is accepted and coerced to ``ComponentConfig(name=<str>, params={})``
    so the common-case ``source: yfinance`` stays one line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: ComponentConfig
    tickers: list[str] = Field(min_length=1)
    start: datetime
    end: datetime
    interval: Interval = Interval.DAILY
    cache_dir: Path | None = None

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, v: object) -> object:
        if isinstance(v, str):
            return {"name": v, "params": {}}
        return v

    @model_validator(mode="after")
    def _validate_range_and_source(self) -> Self:
        if self.start >= self.end:
            raise ValueError(
                f"data.start ({self.start.isoformat()}) must be strictly before "
                f"data.end ({self.end.isoformat()}); swap the dates or widen the range."
            )
        _ensure_registries_populated()
        if self.source.name not in data_source_registry:
            raise ValueError(
                f"unknown data source '{self.source.name}'; "
                f"available: {sorted(data_source_registry.list_all())}"
            )
        return self


class ValidationConfig(BaseModel):
    """Walk-forward validator knobs + holdout reservation contract.

    Fields ``n_splits`` / ``test_size`` / ``gap`` / ``expanding`` /
    ``snap_to_day`` map one-to-one to :class:`WalkForwardValidator.__init__`.

    The holdout contract
    --------------------
    The holdout region is the chunk of data at the END of ``(data.start,
    data.end)`` that MUST NOT be seen by any dev run, HPO trial, or model fit.
    It is reserved for a single post-thesis out-of-sample evaluation — the
    honest number Chapter 7 reports.

    Two mutually-exclusive ways to express the boundary:

    * ``holdout_pct``: declarative fraction (0.15 = reserve last 15% of bars).
      The runner derives the absolute timestamp at fetch time and persists it
      to the experiment manifest.
    * ``holdout_start``: explicit pinned timestamp. Use this to reproduce a
      prior run exactly, or to pin the boundary against data-vendor drift.

    Setting BOTH raises ``ValidationError`` — the boundary must have exactly
    one canonical source. Setting NEITHER disables holdout reservation
    entirely (the caller is responsible for having shortened ``data.end``
    themselves if they meant to reserve something).

    Leakage prevention (six layered tripwires)
    ------------------------------------------
    1. **Config**: ``_validate_holdout_exclusive`` rejects configs that set
       both ``holdout_pct`` and ``holdout_start``.
    2. **Resolution**: :func:`src.core.temporal.resolve_holdout_boundary`
       rejects a pinned ``holdout_start`` that is not present in the fetched
       data (data drift would otherwise shift the boundary silently).
    3. **Split**: the runner constructs ``TemporalSplit(train=dev,
       test=holdout, split_date=boundary)`` which raises ``LeakageError`` on
       any overlap.
    4. **Per-fold construction**: each walk-forward ``TemporalSplit``
       re-validates that train precedes test.
    5. **Per-fold training**: ``strategy.training_metadata.validate_no_overlap
       (fold.test)`` fires before signal generation if training somehow
       covered the test region.
    6. **Final holdout eval** (post-Phase-7 command): loads the saved model,
       reads ``holdout_start`` from the manifest (NOT the config — the
       manifest is the source of truth once written), and calls
       ``strategy.training_metadata.validate_no_overlap(holdout)`` — raises
       ``LeakageError`` if the loaded model's training window overlaps the
       holdout region for any reason.

    Enforcement of points 2-6 lands alongside ``Experiment.run()``; the
    fields are declared now so configs written today stay valid later. Point
    1 is active as of this change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_splits: int = Field(default=4, ge=1)
    test_size: int = Field(default=252, ge=1)
    gap: int = Field(default=5, ge=0)
    expanding: bool = True
    snap_to_day: bool = False
    holdout_pct: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Fraction of the fetched data to reserve as holdout, sliced off "
            "the END. 0.0 disables reservation. Mutually exclusive with "
            "holdout_start."
        ),
    )
    holdout_start: datetime | None = Field(
        default=None,
        description=(
            "Absolute timestamp at which holdout begins. Pinning this makes "
            "the split boundary reproducible across data-vendor updates. "
            "Mutually exclusive with holdout_pct."
        ),
    )

    @model_validator(mode="after")
    def _validate_holdout_exclusive(self) -> Self:
        if self.holdout_pct > 0.0 and self.holdout_start is not None:
            raise ValueError(
                "validation: set at most one of holdout_pct / holdout_start; "
                "they are two ways to express the same boundary and the "
                "manifest can only record one canonical timestamp."
            )
        return self


class SlippageConfigSpec(BaseModel):
    """Slippage scenario selector — indexes into ``SLIPPAGE_SCENARIOS``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario: SlippageScenario = SlippageScenario.NORMAL


class ExperimentConfig(BaseModel):
    """Root experiment config. Loaded from YAML, consumed by ``build_experiment``.

    ``features`` is optional: strategies that own their own feature engineering
    (every Phase 2.5 strategy except hybrids) don't need a separate pipeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    seed: int = 42
    data: DataConfig
    features: ComponentConfig | None = None
    strategy: ComponentConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    slippage: SlippageConfigSpec = Field(default_factory=SlippageConfigSpec)
    risk_free_rate: float = 0.0

    @model_validator(mode="after")
    def _validate_component_names(self) -> Self:
        _ensure_registries_populated()
        if self.strategy.name not in strategy_registry:
            raise ValueError(
                f"unknown strategy '{self.strategy.name}'; "
                f"available: {sorted(strategy_registry.list_all())}"
            )
        if self.features is not None and self.features.name not in feature_registry:
            raise ValueError(
                f"unknown feature pipeline '{self.features.name}'; "
                f"available: {sorted(feature_registry.list_all())}"
            )
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from YAML.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty or pydantic validation fails.
    """
    config_path = Path(path)
    try:
        with open(config_path) as f:
            raw: dict[str, object] | None = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Experiment config not found: {config_path}; "
            f"check the --config path or create the file first."
        ) from None

    if raw is None:
        raise ValueError(
            f"Empty experiment config: {config_path}; "
            f"populate it with at minimum `name`, `data`, and `strategy` sections."
        )

    return ExperimentConfig.model_validate(raw)
