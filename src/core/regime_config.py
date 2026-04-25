"""Typed config for ``experiment regime`` — selects a regime detector.

Kept separate from :class:`ExperimentConfig` for the same reason
:class:`HPOConfig` is: regime analysis is post-hoc, runs against an
already-persisted experiment, and a single experiment is reusable
across multiple regime cuts (covid split, trend, vol quintile) without
rewriting the run YAML.

The detector spec is a :class:`ComponentConfig` so per-detector kwargs
flow cleanly (``period`` needs a list of boundaries; ``trend`` /
``volatility`` take small numeric knobs). A bare string in YAML is
coerced to ``ComponentConfig(name=<str>, params={})`` for the
parameterless default case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.core.config import ComponentConfig, load_yaml_config


class RegimeConfig(BaseModel):
    """Wraps a :class:`ComponentConfig` whose name lives in ``regime_registry``.

    Validation runs in two layers:

    1. ``detector.name`` is checked against ``regime_registry.list_all()``
       at validate-time so an unknown name fails the YAML load with a
       pointed message.
    2. The detector's own ``__init__`` validates the params dict — so
       e.g. ``trend`` rejects ``window=1`` immediately when the
       ``regime`` subcommand instantiates it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: ComponentConfig

    @field_validator("detector", mode="before")
    @classmethod
    def _coerce_detector(cls, v: object) -> object:
        if isinstance(v, str):
            return {"name": v, "params": {}}
        return v

    @model_validator(mode="after")
    def _validate_detector(self) -> Self:
        # Lazy import to avoid pulling pandas / numpy on a YAML schema-
        # check pass that doesn't actually instantiate a detector.
        from src.orchestration.regime import regime_registry

        if self.detector.name not in regime_registry:
            raise ValueError(
                f"unknown regime detector '{self.detector.name}'; "
                f"available: {sorted(regime_registry.list_all())}"
            )
        return self


def load_regime_config(path: str | Path) -> RegimeConfig:
    """Read a YAML file and validate it as a :class:`RegimeConfig`."""
    return load_yaml_config(path, RegimeConfig, "regime")
