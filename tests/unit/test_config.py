"""
Tests for :class:`ExperimentConfig` and :func:`load_experiment_config`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src.core.config import (
    ComponentConfig,
    DataConfig,
    ExperimentConfig,
    SlippageConfigSpec,
    ValidationConfig,
    load_experiment_config,
    load_universe_profile,
)
from src.core.types import Interval
from src.engine.scenarios import SlippageScenario
from tests.conftest import REPO_ROOT

_START = datetime(2020, 1, 1)
_END = datetime(2023, 1, 1)
_SEED = 42
_N_SPLITS = 4
_TEST_SIZE = 252
_GAP = 5
_RISK_FREE = 0.01


def _minimal_config_dict() -> dict[str, Any]:
    """
    Smallest valid config dict - every required field set, defaults elsewhere.
    """

    return {
        "name": "test_run",
        "seed": _SEED,
        "data": {
            "source": "yfinance",
            "tickers": ["SPY"],
            "start": _START,
            "end": _END,
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {"window": 20, "k": 2.0},
        },
        "validation": {
            "n_splits": _N_SPLITS,
            "test_size": _TEST_SIZE,
            "gap": _GAP,
        },
        "slippage": {"scenario": "normal"},
        "risk_free_rate": _RISK_FREE,
    }


class TestExperimentConfig:
    def test_minimal_roundtrip(self) -> None:
        cfg = ExperimentConfig.model_validate(_minimal_config_dict())

        assert cfg.name == "test_run"
        assert cfg.seed == _SEED
        assert cfg.data.source == ComponentConfig(name="yfinance")
        assert cfg.data.tickers == ["SPY"]
        assert cfg.data.interval == Interval.DAILY
        assert cfg.strategy.name == "AdaptiveBollinger"
        assert cfg.strategy.params["window"] == 20
        assert cfg.validation.n_splits == _N_SPLITS
        assert cfg.slippage.scenario == SlippageScenario.NORMAL
        assert cfg.risk_free_rate == _RISK_FREE
        assert cfg.features is None

    def test_features_optional(self) -> None:
        d = _minimal_config_dict()
        d["features"] = {"name": "standard", "params": {"rsi_period": 10}}
        cfg = ExperimentConfig.model_validate(d)
        assert cfg.features is not None
        assert cfg.features.name == "standard"
        assert cfg.features.params["rsi_period"] == 10

    def test_string_source_coerces_to_component(self) -> None:
        cfg = ExperimentConfig.model_validate(_minimal_config_dict())
        assert isinstance(cfg.data.source, ComponentConfig)
        assert cfg.data.source.name == "yfinance"
        assert cfg.data.source.params == {}

    def test_explicit_component_source_accepted(self) -> None:
        d = _minimal_config_dict()
        d["data"]["source"] = {"name": "csv", "params": {"data_dir": "/tmp"}}
        cfg = ExperimentConfig.model_validate(d)
        assert cfg.data.source.name == "csv"
        assert cfg.data.source.params == {"data_dir": "/tmp"}

    def test_unknown_strategy_raises(self) -> None:
        d = _minimal_config_dict()
        d["strategy"]["name"] = "NoSuchStrategy"
        with pytest.raises(ValidationError, match="unknown strategy 'NoSuchStrategy'"):
            ExperimentConfig.model_validate(d)

    def test_unknown_data_source_raises(self) -> None:
        d = _minimal_config_dict()
        d["data"]["source"] = "no_such_source"
        with pytest.raises(ValidationError, match="unknown data source 'no_such_source'"):
            ExperimentConfig.model_validate(d)

    def test_unknown_feature_pipeline_raises(self) -> None:
        d = _minimal_config_dict()
        d["features"] = {"name": "nope", "params": {}}
        with pytest.raises(ValidationError, match="unknown feature pipeline 'nope'"):
            ExperimentConfig.model_validate(d)

    def test_start_after_end_raises(self) -> None:
        d = _minimal_config_dict()
        d["data"]["start"] = _END
        d["data"]["end"] = _START
        with pytest.raises(ValidationError, match="must be strictly before"):
            ExperimentConfig.model_validate(d)

    def test_empty_tickers_raises(self) -> None:
        d = _minimal_config_dict()
        d["data"]["tickers"] = []
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(d)

    def test_extra_fields_forbidden(self) -> None:
        d = _minimal_config_dict()
        d["not_a_real_field"] = "value"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ExperimentConfig.model_validate(d)

    def test_extra_fields_forbidden_in_data(self) -> None:
        d = _minimal_config_dict()
        d["data"]["extra"] = "oops"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ExperimentConfig.model_validate(d)

    def test_frozen_instances(self) -> None:
        cfg = ExperimentConfig.model_validate(_minimal_config_dict())
        with pytest.raises(ValidationError):
            cfg.name = "changed"


class TestLoadExperimentConfig:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(yaml.safe_dump(_minimal_config_dict(), default_flow_style=False))

        cfg = load_experiment_config(yaml_path)
        assert cfg.name == "test_run"
        assert cfg.strategy.name == "AdaptiveBollinger"

    def test_missing_file_raises_with_remediation(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="check the --config path"):
            load_experiment_config(tmp_path / "missing.yaml")

    def test_empty_file_raises_with_remediation(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        with pytest.raises(ValueError, match="Empty experiment config"):
            load_experiment_config(yaml_path)


class TestReferenceConfigLoads:
    """
    The committed reference ``config/example.yaml`` must validate.
    """

    def test_example_yaml_validates(self) -> None:
        cfg = load_experiment_config(REPO_ROOT / "config/example.yaml")
        assert cfg.name
        assert cfg.data.tickers
        assert isinstance(cfg.validation, ValidationConfig)
        assert isinstance(cfg.slippage, SlippageConfigSpec)


_UNIVERSE_DIR = REPO_ROOT / "config" / "universes"


class TestUniverseProfilesLoad:
    @pytest.mark.parametrize("profile_path", sorted(_UNIVERSE_DIR.glob("*.yaml")))
    def test_universe_yaml_parses_as_profile(self, profile_path: Path) -> None:
        profile = load_universe_profile(profile_path)
        assert profile.data.tickers
        assert profile.data.start < profile.data.end
        assert isinstance(profile.validation, ValidationConfig)
        assert profile.validation.holdout_pct >= 0.0


_HOLDOUT_PCT = 0.15
_HOLDOUT_START_ISO = "2023-01-03T00:00:00"


class TestValidationConfigHoldout:
    """
    Holdout contract (ValidationConfig docstring, tripwire #1).
    """

    def test_holdout_fields_default_to_unset(self) -> None:
        cfg = ExperimentConfig.model_validate(_minimal_config_dict())
        assert cfg.validation.holdout_pct == 0.0
        assert cfg.validation.holdout_start is None

    def test_holdout_pct_accepted_in_range(self) -> None:
        d = _minimal_config_dict()
        d["validation"]["holdout_pct"] = _HOLDOUT_PCT
        cfg = ExperimentConfig.model_validate(d)
        assert cfg.validation.holdout_pct == pytest.approx(_HOLDOUT_PCT)
        assert cfg.validation.holdout_start is None

    def test_holdout_pct_rejects_negative(self) -> None:
        d = _minimal_config_dict()
        d["validation"]["holdout_pct"] = -0.01
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(d)

    def test_holdout_pct_rejects_one_or_more(self) -> None:
        d = _minimal_config_dict()
        d["validation"]["holdout_pct"] = 1.0
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(d)

    def test_holdout_start_accepted_as_iso_string(self) -> None:
        d = _minimal_config_dict()
        d["validation"]["holdout_start"] = _HOLDOUT_START_ISO
        cfg = ExperimentConfig.model_validate(d)
        assert cfg.validation.holdout_start == datetime.fromisoformat(_HOLDOUT_START_ISO)
        assert cfg.validation.holdout_pct == 0.0

    def test_both_holdout_fields_raises_exclusivity(self) -> None:
        """
        Tripwire #1: config-level exclusivity between pct and start.
        """

        d = _minimal_config_dict()
        d["validation"]["holdout_pct"] = _HOLDOUT_PCT
        d["validation"]["holdout_start"] = _HOLDOUT_START_ISO
        with pytest.raises(ValidationError, match="at most one of holdout_pct / holdout_start"):
            ExperimentConfig.model_validate(d)


class TestDataConfigEdges:
    def test_interval_accepts_string(self) -> None:
        d = _minimal_config_dict()
        d["data"]["interval"] = "hour"
        cfg = ExperimentConfig.model_validate(d)
        assert cfg.data.interval == Interval.HOUR

    def test_cache_dir_optional(self) -> None:
        cfg = ExperimentConfig.model_validate(_minimal_config_dict())
        assert cfg.data.cache_dir is None

    def test_cache_dir_coerced_to_path(self) -> None:
        d = _minimal_config_dict()
        d["data"]["cache_dir"] = "/tmp/cache"
        cfg = ExperimentConfig.model_validate(d)
        assert cfg.data.cache_dir == Path("/tmp/cache")


class TestComponentConfigDefaults:
    def test_params_default_empty(self) -> None:
        c = ComponentConfig(name="x")
        assert c.params == {}

    def test_params_accept_heterogeneous_values(self) -> None:
        c = ComponentConfig(
            name="x",
            params={"window": 20, "k": 2.0, "trending": True, "tickers": ["A", "B"]},
        )
        assert c.params["window"] == 20
        assert c.params["tickers"] == ["A", "B"]


class TestStandaloneModels:
    def test_data_config_direct(self) -> None:
        dc = DataConfig(
            source=ComponentConfig(name="yfinance"),
            tickers=["AAPL"],
            start=_START,
            end=_END,
        )
        assert dc.interval == Interval.DAILY
