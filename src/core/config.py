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

from collections import Counter
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
    """Import the component packages so their registry decorators run.

    Each package's ``__init__.py`` walks its own directory and imports every
    non-private, non-``interface`` module — so a new strategy / data source /
    feature pipeline / model file registers itself automatically, with no
    edits here.

    Deferred out of module scope because these package imports transitively
    pull in torch / xgboost / arch / statsmodels / pmdarima (~4 s on a cold
    interpreter). Consumers that only reference ``ExperimentConfig`` /
    ``FoldRecord`` as types pay zero ML-framework cost. Python caches module
    imports in ``sys.modules``, so repeat calls are effectively free.
    """
    import src.data  # noqa: F401
    import src.features  # noqa: F401
    import src.models  # noqa: F401
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

    Leakage prevention (five layered tripwires)
    -------------------------------------------
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
    don't need a separate pipeline.
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


def _find_duplicates[T](items: list[T]) -> list[T]:
    """Return items that occur more than once, in first-occurrence order."""
    return [item for item, count in Counter(items).items() if count > 1]


class UniverseProfile(BaseModel):
    """Reusable ``data:`` + ``validation:`` block deep-merged onto a strategy YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class StudyLeg(BaseModel):
    """One (strategy × set-of-universes) leg of an empirical study."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = Field(min_length=1)
    strategy_config: Path
    hpo_config: Path
    universes: list[str] = Field(min_length=1)

    @field_validator("universes")
    @classmethod
    def _no_duplicate_universes(cls, v: list[str]) -> list[str]:
        dupes = _find_duplicates(v)
        if dupes:
            raise ValueError(
                f"duplicate universe name(s) in leg: {dupes!r}; "
                f"each universe must appear at most once per leg."
            )
        return v


class StudySpec(BaseModel):
    """Declarative enumeration of every (strategy × universe) leg of a study.

    Path fields are typed ``Path`` but not checked for existence at schema
    validation time so the schema stays pure. The orchestrator and the
    test suite open the files.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    seed: int = 42
    output_dir: Path
    legs: list[StudyLeg] = Field(min_length=1)

    @field_validator("legs")
    @classmethod
    def _no_duplicate_strategies(cls, v: list[StudyLeg]) -> list[StudyLeg]:
        dupes = _find_duplicates([leg.strategy for leg in v])
        if dupes:
            raise ValueError(
                f"duplicate strategy name(s) across legs: {dupes!r}; "
                f"merge their universe lists into a single leg."
            )
        return v


def write_frozen_yaml(path: str | Path, cfg: BaseModel, *, sort_keys: bool = True) -> None:
    """Dump a validated pydantic config to YAML at ``path``.

    ``mode="json"`` coerces ``datetime`` / ``Path`` / enum values to
    JSON-safe primitives ``yaml.safe_dump`` accepts. Used by the experiment
    runner to write the frozen ``config.yaml`` alongside ``manifest.json``.
    """
    payload = cfg.model_dump(mode="json")
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=sort_keys)


def load_yaml_config[T: BaseModel](path: str | Path, cls: type[T], kind: str) -> T:
    """Shared YAML-load pipeline for :class:`ExperimentConfig` and siblings.

    ``experiment run`` / ``experiment tune`` want identical error framing
    for missing / empty / invalid config files — extracting the common
    logic avoids drift in the error messages users actually see.
    """
    config_path = Path(path)
    try:
        with open(config_path) as f:
            raw: dict[str, object] | None = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{kind} config not found: {config_path}; "
            f"check the --config path or create the file first."
        ) from None

    if raw is None:
        raise ValueError(
            f"Empty {kind} config: {config_path}; populate it with at minimum "
            f"the fields required by {cls.__name__}."
        )

    return cls.model_validate(raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from YAML.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty or pydantic validation fails.
    """
    return load_yaml_config(path, ExperimentConfig, "experiment")


def load_universe_profile(path: str | Path) -> UniverseProfile:
    """Load and validate a :class:`UniverseProfile` from YAML.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty or pydantic validation fails.
    """
    return load_yaml_config(path, UniverseProfile, "universe profile")


def load_study_spec(path: str | Path) -> StudySpec:
    """Load and validate a :class:`StudySpec` from YAML.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty or pydantic validation fails.
    """
    return load_yaml_config(path, StudySpec, "study spec")
