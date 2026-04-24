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

from src.core.leaf_keys import describe_supported_leaf_keys
from src.core.registry import (
    classifier_registry,
    data_source_registry,
    feature_registry,
    model_registry,
    strategy_registry,
)
from src.core.types import Interval, ModelKind
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


# Per-strategy list of ctor kwargs that a pretrained leaf OWNS: when the
# leaf is frozen-injected, these hyperparameters are pinned by the artifact
# and the user shouldn't also be setting them in strategy.params (the
# artifact wins silently, which makes HPO vs. pretrained collisions a
# silent-misfit debugging nightmare). ``feature_columns``, ``interval``,
# and ``lstm_lookback`` are deliberately excluded — the strategy still
# needs those even when the leaf is frozen, and ``validate_pretrained_leaf``
# catches mismatches vs. the artifact at injection time.
_LEAF_KEY_OWNED_PARAMS: dict[str, dict[str, tuple[str, ...]]] = {
    "ReturnForecast": {
        "return_model": (
            "arma_p_max",
            "arma_q_max",
            "arma_information_criterion",
            "lstm_hidden_dim",
            "lstm_num_layers",
            "lstm_dropout",
            "lstm_lr",
            "lstm_epochs",
            "lstm_loss_fn",
            "lstm_patience",
            "lstm_batch_size",
            "lstm_val_split_ratio",
            "lstm_device",
        ),
    },
    "VolatilityTargeting": {
        "vol_model": (
            "garch_p_max",
            "garch_q_max",
            "lstm_hidden_dim",
            "lstm_num_layers",
            "lstm_dropout",
            "lstm_lr",
            "lstm_epochs",
            "lstm_loss_fn",
            "lstm_patience",
            "lstm_batch_size",
            "lstm_val_split_ratio",
            "lstm_device",
            "min_vol",
        ),
    },
}


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


class StandaloneModelConfig(BaseModel):
    """Root config for ``experiment train-model`` — train one leaf standalone.

    A sibling of :class:`ExperimentConfig`: same ``data`` / ``features`` /
    ``seed`` fields, different output. Runs one ``model.fit()`` on the
    data slice ``[train_start, train_end]`` (inclusive) and persists a
    model artifact at ``experiment_results/models/<name>/`` that a later
    ``experiment run`` can inject via ``pretrained_leaves``.

    ``model_kind`` disambiguates the two model registries (predictor vs
    classifier). Getting it wrong at config-load time fails loud with an
    actionable error; getting it wrong at load-time silently dispatches
    to the wrong registry, which has been a source of real bugs.

    ``train_start`` / ``train_end`` default to ``None`` (use the full
    fetched window). When set, they must bracket ``data.start`` /
    ``data.end`` respectively — the data fetch fails otherwise and we
    surface that at config-validate time instead of after fetching.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    seed: int = 42
    data: DataConfig
    features: ComponentConfig | None = None
    model: ComponentConfig
    model_kind: ModelKind = ModelKind.PREDICTOR
    train_start: datetime | None = None
    train_end: datetime | None = None

    @model_validator(mode="after")
    def _validate_model_registered_and_range(self) -> Self:
        _ensure_registries_populated()
        registry = model_registry if self.model_kind == ModelKind.PREDICTOR else classifier_registry
        if self.model.name not in registry:
            raise ValueError(
                f"unknown {self.model_kind.value} '{self.model.name}'; "
                f"available: {sorted(registry.list_all())}"
            )
        if self.features is not None and self.features.name not in feature_registry:
            raise ValueError(
                f"unknown feature pipeline '{self.features.name}'; "
                f"available: {sorted(feature_registry.list_all())}"
            )
        if (
            self.train_start is not None
            and self.train_end is not None
            and self.train_start >= self.train_end
        ):
            raise ValueError(
                f"train_start ({self.train_start.isoformat()}) must be strictly "
                f"before train_end ({self.train_end.isoformat()}); swap or widen."
            )
        # ``data.interval`` is the source of truth — the standalone trainer
        # injects it into model.params before ctor. Accepting the same key
        # under model.params silently lets a user's value get stomped. Reject
        # at config time so the mismatch is visible at the boundary.
        if "interval" in self.model.params:
            raise ValueError(
                "model.params must not set 'interval'; data.interval is the "
                "canonical source and the standalone trainer forwards it to "
                "the model ctor automatically. Remove 'interval' from "
                "model.params."
            )
        return self


class ExperimentConfig(BaseModel):
    """Root experiment config. Loaded from YAML, consumed by ``build_experiment``.

    ``features`` is optional: strategies that own their own feature engineering
    don't need a separate pipeline.

    ``pretrained_leaves`` maps a strategy-specific leaf key (e.g.
    ``"return_model"`` for ReturnForecast, ``"vol_model"`` for
    VolatilityTargeting) to the directory of a previously-saved standalone
    model artifact. The builder loads each artifact and injects it into
    the strategy via its ``pretrained_leaves`` ctor kwarg; the leaf stays
    frozen across folds while the strategy's own state re-fits each fold.

    Config-layer validation (see ``_validate_pretrained_leaves_config``):

    * Every key must appear in the target strategy's ``_leaf_keys``
      ClassVar. Unknown keys raise with the supported set listed.
    * Every path must exist as a directory (presence of ``manifest.json``
      is left to load-time; config validation is file-system shallow).
    * Leaf-specific params in ``strategy.params`` collide with a frozen
      leaf — if ``pretrained_leaves`` pins ``"return_model"``, passing
      e.g. ``arma_p_max`` in ``strategy.params`` is rejected: the artifact
      owns those hyperparameters and silently ignoring user-supplied
      values would confuse debugging. The per-leaf-key conflict set is
      declared as a ClassVar ``_LEAF_KEY_OWNED_PARAMS`` below.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    seed: int = 42
    data: DataConfig
    features: ComponentConfig | None = None
    strategy: ComponentConfig
    pretrained_leaves: dict[str, Path] = Field(default_factory=dict)
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

    @model_validator(mode="after")
    def _validate_pretrained_leaves_config(self) -> Self:
        if not self.pretrained_leaves:
            return self

        _ensure_registries_populated()
        # Pydantic v2 does not guarantee ordering between model_validator
        # callbacks: if this runs before ``_validate_component_names`` on a
        # config with an unknown strategy name, ``strategy_registry.get``
        # raises ``KeyError`` that pydantic wraps into an unhelpful error.
        # Defer to the other validator for the canonical "unknown strategy"
        # message; the unknown-strategy case still fails validation.
        if self.strategy.name not in strategy_registry:
            return self
        strategy_cls = strategy_registry.get(self.strategy.name)
        supported_keys: frozenset[str] = getattr(strategy_cls, "_leaf_keys", frozenset())

        unknown = set(self.pretrained_leaves) - supported_keys
        if unknown:
            raise ValueError(
                f"pretrained_leaves contains unknown key(s) {sorted(unknown)!r} "
                f"for strategy '{self.strategy.name}'; "
                f"{describe_supported_leaf_keys(supported_keys, self.strategy.name)}."
            )

        for key, path in self.pretrained_leaves.items():
            if not path.is_dir():
                raise ValueError(
                    f"pretrained_leaves['{key}'] path does not exist or is not a "
                    f"directory: {path}; fix by running `experiment train-model` "
                    f"first, or by pointing at an existing artifact directory."
                )

        collision_map = _LEAF_KEY_OWNED_PARAMS.get(self.strategy.name, {})
        collisions: dict[str, list[str]] = {}
        for key in self.pretrained_leaves:
            owned = collision_map.get(key, ())
            overlap = sorted(set(owned) & set(self.strategy.params))
            if overlap:
                collisions[key] = overlap
        if collisions:
            raise ValueError(
                f"pretrained_leaves frozen leaf owns hyperparameters that are also "
                f"set in strategy.params — artifact would win silently: {collisions}. "
                f"Fix by removing those keys from strategy.params (the leaf already "
                f"pins them)."
            )
        return self


def write_frozen_yaml(path: str | Path, cfg: BaseModel, *, sort_keys: bool = True) -> None:
    """Dump a validated pydantic config to YAML at ``path``.

    ``mode="json"`` coerces ``datetime`` / ``Path`` / enum values to
    JSON-safe primitives ``yaml.safe_dump`` accepts. Shared between the
    experiment runner (frozen ``config.yaml`` alongside ``manifest.json``)
    and the standalone-model artifact writer.
    """
    payload = cfg.model_dump(mode="json")
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=sort_keys)


def _load_yaml_config[T: BaseModel](path: str | Path, cls: type[T], kind: str) -> T:
    """Shared YAML-load pipeline for :class:`ExperimentConfig` and siblings.

    Both ``experiment run`` and ``experiment train-model`` want identical
    error framing for missing / empty / invalid config files — extracting
    the common logic avoids drift in the error messages users actually
    see.
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
    return _load_yaml_config(path, ExperimentConfig, "experiment")


def load_standalone_model_config(path: str | Path) -> StandaloneModelConfig:
    """Load and validate a :class:`StandaloneModelConfig` from YAML.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty or pydantic validation fails.
    """
    return _load_yaml_config(path, StandaloneModelConfig, "standalone-model")
