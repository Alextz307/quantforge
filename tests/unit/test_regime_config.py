"""Unit tests for ``RegimeConfig`` YAML loading + validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.regime_config import RegimeConfig, load_regime_config


def test_period_yaml_loads_with_inline_boundaries(tmp_path: Path) -> None:
    yaml_content = """
detector:
  name: period
  params:
    boundaries:
      - label: pre
        start: "2019-01-01"
        end: "2020-01-01"
      - label: post
        start: "2020-01-01"
        end: "2021-01-01"
"""
    path = tmp_path / "period.yaml"
    path.write_text(yaml_content)
    cfg = load_regime_config(path)
    assert cfg.detector.name == "period"
    assert isinstance(cfg.detector.params["boundaries"], list)


def test_bare_string_detector_coerces_to_componentconfig() -> None:
    cfg = RegimeConfig.model_validate({"detector": "trend"})
    assert cfg.detector.name == "trend"
    assert cfg.detector.params == {}


def test_unknown_detector_name_raises_with_alternatives() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RegimeConfig.model_validate({"detector": {"name": "not_a_real_detector", "params": {}}})
    msg = str(exc_info.value)
    assert "not_a_real_detector" in msg
    # Available detectors are listed in the error
    for name in ("period", "trend", "volatility"):
        assert name in msg


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        RegimeConfig.model_validate({"detector": "trend", "unknown_field": "should be rejected"})


def test_shipped_yamls_load_clean() -> None:
    """The three example YAMLs under config/regimes/ must parse + validate."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    for filename in ("covid_split.yaml", "bull_bear_200ma.yaml", "vol_quintile.yaml"):
        path = repo_root / "config" / "regimes" / filename
        cfg = load_regime_config(path)
        assert cfg.detector.name in {"period", "trend", "volatility"}
