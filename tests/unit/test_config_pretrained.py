"""Tests for :class:`ExperimentConfig.pretrained_leaves` validation.

Covers the three config-layer rejection classes documented on
``ExperimentConfig._validate_pretrained_leaves_config``:

* unknown leaf key for the target strategy
* non-existent or non-directory path
* leaf-owned hyperparameter in ``strategy.params`` (collision)

Plus a happy-path round-trip through ``model_validate``.

Also covers :class:`StandaloneModelConfig` validation: unknown model
name, unknown feature pipeline, invalid train_start/end range.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import ExperimentConfig, StandaloneModelConfig
from src.core.types import ModelKind

_SPY_DATA = {
    "source": "csv",
    "tickers": ["SPY"],
    "start": "2020-01-01",
    "end": "2024-01-01",
    "interval": "daily",
}
_FEATURE_COLUMNS = ["sma_20", "rsi_14", "volume_z"]


def _base_payload(tmp_path: Path, strategy_name: str = "ReturnForecast") -> dict[str, object]:
    return {
        "name": "t",
        "seed": 42,
        "data": _SPY_DATA,
        "strategy": {
            "name": strategy_name,
            "params": {"feature_columns": _FEATURE_COLUMNS},
        },
    }


class TestExperimentConfigPretrainedLeaves:
    def test_empty_dict_is_accepted(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig.model_validate(_base_payload(tmp_path))
        assert cfg.pretrained_leaves == {}

    def test_valid_key_and_existing_path_passes(self, tmp_path: Path) -> None:
        leaf_dir = tmp_path / "hybrid_ret"
        leaf_dir.mkdir()
        payload = _base_payload(tmp_path)
        payload["pretrained_leaves"] = {"return_model": str(leaf_dir)}
        cfg = ExperimentConfig.model_validate(payload)
        assert cfg.pretrained_leaves == {"return_model": leaf_dir}

    def test_unknown_leaf_key_raises(self, tmp_path: Path) -> None:
        leaf_dir = tmp_path / "model"
        leaf_dir.mkdir()
        payload = _base_payload(tmp_path)
        payload["pretrained_leaves"] = {"vol_model": str(leaf_dir)}
        with pytest.raises(ValueError, match="unknown key"):
            ExperimentConfig.model_validate(payload)

    def test_non_ml_strategy_rejects_any_pretrained_leaf(self, tmp_path: Path) -> None:
        leaf_dir = tmp_path / "model"
        leaf_dir.mkdir()
        payload = _base_payload(tmp_path, strategy_name="AdaptiveBollinger")
        payload["strategy"] = {"name": "AdaptiveBollinger", "params": {}}
        payload["pretrained_leaves"] = {"garch": str(leaf_dir)}
        with pytest.raises(ValueError, match="owns no ML leaves"):
            ExperimentConfig.model_validate(payload)

    def test_non_existent_path_raises(self, tmp_path: Path) -> None:
        payload = _base_payload(tmp_path)
        payload["pretrained_leaves"] = {"return_model": str(tmp_path / "nonexistent")}
        with pytest.raises(ValueError, match="does not exist"):
            ExperimentConfig.model_validate(payload)

    def test_file_not_directory_raises(self, tmp_path: Path) -> None:
        leaf_file = tmp_path / "leaf.json"
        leaf_file.write_text("{}")
        payload = _base_payload(tmp_path)
        payload["pretrained_leaves"] = {"return_model": str(leaf_file)}
        with pytest.raises(ValueError, match="not a\\s+directory"):
            ExperimentConfig.model_validate(payload)

    def test_leaf_owned_param_collision_raises(self, tmp_path: Path) -> None:
        """Setting a leaf-owned hyperparameter in ``strategy.params`` collides
        with a pinned pretrained leaf — the artifact owns that value."""
        leaf_dir = tmp_path / "model"
        leaf_dir.mkdir()
        payload = _base_payload(tmp_path)
        payload["strategy"]["params"]["arma_p_max"] = 3  # type: ignore[index]
        payload["pretrained_leaves"] = {"return_model": str(leaf_dir)}
        with pytest.raises(ValueError, match="frozen leaf owns hyperparameters"):
            ExperimentConfig.model_validate(payload)

    def test_non_leaf_owned_param_is_fine(self, tmp_path: Path) -> None:
        """Strategy-level params (``position_scale``, ``max_leverage``) stay
        user-tunable even with a frozen leaf."""
        leaf_dir = tmp_path / "model"
        leaf_dir.mkdir()
        payload = _base_payload(tmp_path)
        payload["strategy"]["params"]["position_scale"] = 15.0  # type: ignore[index]
        payload["pretrained_leaves"] = {"return_model": str(leaf_dir)}
        ExperimentConfig.model_validate(payload)

    def test_unknown_strategy_surfaces_canonical_error(self, tmp_path: Path) -> None:
        """Pydantic v2 does not order model_validator callbacks; if the
        pretrained-leaves validator runs first on an unknown strategy, it
        must defer to ``_validate_component_names`` (canonical "unknown
        strategy" error) rather than raising ``KeyError`` from the
        registry lookup.
        """
        leaf_dir = tmp_path / "model"
        leaf_dir.mkdir()
        payload = _base_payload(tmp_path, strategy_name="NotARealStrategy")
        payload["pretrained_leaves"] = {"return_model": str(leaf_dir)}
        with pytest.raises(ValueError, match="unknown strategy"):
            ExperimentConfig.model_validate(payload)

    def test_momentum_gatekeeper_directional_classifier_key_accepted(self, tmp_path: Path) -> None:
        leaf_dir = tmp_path / "xgb_dir"
        leaf_dir.mkdir()
        payload = _base_payload(tmp_path, strategy_name="MomentumGatekeeper")
        payload["strategy"] = {"name": "MomentumGatekeeper", "params": {}}
        payload["pretrained_leaves"] = {"directional_classifier": str(leaf_dir)}
        cfg = ExperimentConfig.model_validate(payload)
        assert cfg.pretrained_leaves == {"directional_classifier": leaf_dir}

    def test_momentum_gatekeeper_xgb_param_collision_raises(self, tmp_path: Path) -> None:
        """Setting ``n_estimators`` in ``strategy.params`` collides with a
        frozen DirectionalClassifier — the artifact owns the booster size."""
        leaf_dir = tmp_path / "xgb_dir"
        leaf_dir.mkdir()
        payload = _base_payload(tmp_path, strategy_name="MomentumGatekeeper")
        payload["strategy"] = {"name": "MomentumGatekeeper", "params": {"n_estimators": 50}}
        payload["pretrained_leaves"] = {"directional_classifier": str(leaf_dir)}
        with pytest.raises(ValueError, match="frozen leaf owns hyperparameters"):
            ExperimentConfig.model_validate(payload)


class TestStandaloneModelConfig:
    def _base(self) -> dict[str, object]:
        return {
            "name": "spy_hybrid_ret_2024q4",
            "seed": 42,
            "data": _SPY_DATA,
            "model": {"name": "hybrid_return", "params": {"feature_columns": _FEATURE_COLUMNS}},
            "model_kind": "predictor",
        }

    def test_happy_path(self) -> None:
        cfg = StandaloneModelConfig.model_validate(self._base())
        assert cfg.model.name == "hybrid_return"
        assert cfg.model_kind == ModelKind.PREDICTOR

    def test_unknown_model_raises(self) -> None:
        payload = self._base()
        payload["model"] = {"name": "nonexistent_model", "params": {}}
        with pytest.raises(ValueError, match="unknown predictor"):
            StandaloneModelConfig.model_validate(payload)

    def test_unknown_classifier_raises_targeted_message(self) -> None:
        payload = self._base()
        payload["model_kind"] = "classifier"
        payload["model"] = {"name": "hybrid_return", "params": {}}
        # "hybrid_return" is a predictor — not a classifier. Wrong kind.
        with pytest.raises(ValueError, match="unknown classifier"):
            StandaloneModelConfig.model_validate(payload)

    def test_inverted_train_window_raises(self) -> None:
        payload = self._base()
        payload["train_start"] = "2023-01-01"
        payload["train_end"] = "2020-01-01"
        with pytest.raises(ValueError, match="train_start.*before.*train_end"):
            StandaloneModelConfig.model_validate(payload)

    def test_interval_in_model_params_rejected(self) -> None:
        """``data.interval`` is the canonical source; the standalone trainer
        injects it into ``model.params`` automatically. Accepting an
        ``interval`` in model.params silently overwrites the user's value
        — reject at config time so the mismatch is visible at the boundary.
        """
        payload = self._base()
        payload["model"] = {
            "name": "hybrid_return",
            "params": {"feature_columns": _FEATURE_COLUMNS, "interval": "hourly"},
        }
        with pytest.raises(ValueError, match="model.params must not set 'interval'"):
            StandaloneModelConfig.model_validate(payload)
