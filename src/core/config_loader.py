"""YAML configuration loader for pipeline assembly.

Parses and validates pipeline configuration. Full pipeline assembly
is a planned extension that will wire the validated config into the
orchestrator; today the loader only round-trips + validates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.core.types import Interval


class ValidationConfig(BaseModel):
    """Validation section of pipeline config."""

    model_config = ConfigDict(frozen=True)

    method: str = "walk_forward"
    n_splits: int = Field(default=4, ge=1)
    test_size: int = Field(default=252, ge=1)
    gap: int = Field(default=5, ge=0)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration parsed from YAML."""

    model_config = ConfigDict(frozen=True)

    data_source: str = "yfinance"
    tickers: list[str]
    interval: Interval = Interval.DAILY
    # Any is required here: YAML values are untyped at parse time,
    # and Pydantic handles runtime validation/coercion.
    features: list[dict[str, Any]] = Field(default_factory=list)
    strategies: list[dict[str, Any]] = Field(default_factory=list)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    slippage_scenarios: list[str] = Field(
        default_factory=lambda: ["zero", "normal", "adverse", "extreme"]
    )
    metrics: list[str] = Field(default_factory=lambda: ["sharpe", "sortino", "max_drawdown"])


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline configuration from YAML.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated PipelineConfig instance.
    """
    config_path = Path(path)
    try:
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}") from None

    if raw is None:
        raise ValueError(f"Empty config file: {config_path}")

    return PipelineConfig(**raw)


def build_pipeline_from_config(path: str | Path) -> None:
    """Build a complete pipeline from YAML config.

    Currently a skeleton — full assembly is a planned extension that
    will instantiate the orchestrator from the validated config.

    Args:
        path: Path to the YAML configuration file.

    Raises:
        NotImplementedError: Always — the orchestration wiring is not in place yet.
    """
    config = load_pipeline_config(path)

    raise NotImplementedError(
        f"Pipeline assembly not yet implemented. "
        f"Config loaded successfully with {len(config.tickers)} tickers, "
        f"{len(config.strategies)} strategies."
    )
