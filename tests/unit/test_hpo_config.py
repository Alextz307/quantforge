"""
Validation + YAML round-trip for :class:`HPOConfig`.

Kept decoupled from :class:`ExperimentConfig`: the tuner takes both as
separate positional args, so this test file doesn't reach into
experiment-config validators - that surface is covered elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.core.hpo_config import (
    HPOConfig,
    PrunerKind,
    SamplerKind,
    load_hpo_config,
)

_MIN_STUDY_NAME = "spy_bollinger"


class TestHPOConfigDefaults:
    def test_minimal_payload_applies_defaults(self) -> None:
        cfg = HPOConfig.model_validate({"study_name": _MIN_STUDY_NAME})
        assert cfg.study_name == _MIN_STUDY_NAME
        assert cfg.n_trials == 100
        assert cfg.n_jobs == 1
        assert cfg.sampler == SamplerKind.TPE
        assert cfg.pruner == PrunerKind.MEDIAN
        assert cfg.timeout_s is None
        assert cfg.seed == 42

    def test_enum_strings_coerce_from_yaml(self) -> None:
        cfg = HPOConfig.model_validate(
            {
                "study_name": _MIN_STUDY_NAME,
                "sampler": "random",
                "pruner": "hyperband",
            }
        )
        assert cfg.sampler == SamplerKind.RANDOM
        assert cfg.pruner == PrunerKind.HYPERBAND

    def test_frozen_config(self) -> None:
        cfg = HPOConfig.model_validate({"study_name": _MIN_STUDY_NAME})
        with pytest.raises(ValidationError):
            cfg.n_trials = 100


class TestHPOConfigValidation:
    def test_missing_study_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            HPOConfig.model_validate({})

    def test_study_name_with_slash_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path separator"):
            HPOConfig.model_validate({"study_name": "some/nested/path"})

    def test_study_name_with_backslash_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path separator"):
            HPOConfig.model_validate({"study_name": "win\\path"})

    def test_zero_trials_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HPOConfig.model_validate({"study_name": _MIN_STUDY_NAME, "n_trials": 0})

    def test_zero_jobs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HPOConfig.model_validate({"study_name": _MIN_STUDY_NAME, "n_jobs": 0})

    def test_negative_jobs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HPOConfig.model_validate({"study_name": _MIN_STUDY_NAME, "n_jobs": -1})

    def test_non_positive_timeout_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HPOConfig.model_validate({"study_name": _MIN_STUDY_NAME, "timeout_s": 0.0})

    def test_unknown_sampler_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HPOConfig.model_validate({"study_name": _MIN_STUDY_NAME, "sampler": "grid"})

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HPOConfig.model_validate({"study_name": _MIN_STUDY_NAME, "unknown_knob": True})


class TestLoadHPOConfig:
    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        payload = {
            "study_name": _MIN_STUDY_NAME,
            "n_trials": 5,
            "n_jobs": 2,
            "sampler": "tpe",
            "pruner": "none",
            "timeout_s": 60.0,
            "seed": 1,
        }
        config_path = tmp_path / "hpo.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(payload, f)

        cfg = load_hpo_config(config_path)
        assert cfg.study_name == _MIN_STUDY_NAME
        assert cfg.n_trials == 5
        assert cfg.n_jobs == 2
        assert cfg.pruner == PrunerKind.NONE

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="hpo config not found"):
            load_hpo_config(tmp_path / "missing.yaml")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        with pytest.raises(ValueError, match="Empty hpo config"):
            load_hpo_config(config_path)
