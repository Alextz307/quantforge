"""Unit tests for `webapp.backend.app.services.config_service`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from webapp.backend.app.schemas.configs import ConfigKind
from webapp.backend.app.services.config_service import (
    ConfigNotFoundError,
    list_configs,
    read_config,
    validate,
)
from webapp.backend.tests.conftest import make_valid_experiment_payload

VALID_UNIVERSE_PAYLOAD: dict[str, object] = {
    "data": {
        "source": "yfinance",
        "tickers": ["SPY"],
        "start": "2020-01-01",
        "end": "2024-12-31",
        "interval": "daily",
    },
    "validation": {"holdout_pct": 0.2},
}


@pytest.fixture
def config_root(tmp_path: Path) -> Path:
    """Synthetic ``config/`` tree mirroring the real layout's plurals."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "universes").mkdir()
    (tmp_path / "hpo").mkdir()
    (tmp_path / "study").mkdir()
    (tmp_path / "regimes").mkdir()
    (tmp_path / "models").mkdir()

    (tmp_path / "strategies" / "ab.yaml").write_text(
        yaml.safe_dump({"name": "AdaptiveBollinger", "params": {}}),
        encoding="utf-8",
    )
    (tmp_path / "universes" / "spy.yaml").write_text(
        yaml.safe_dump(VALID_UNIVERSE_PAYLOAD),
        encoding="utf-8",
    )
    (tmp_path / "universes" / "qqq.yaml").write_text(
        yaml.safe_dump(VALID_UNIVERSE_PAYLOAD),
        encoding="utf-8",
    )
    return tmp_path


def test_validate_experiment_happy_path() -> None:
    response = validate(ConfigKind.EXPERIMENT, make_valid_experiment_payload())

    assert response.valid is True
    assert response.errors == []


def test_validate_universe_happy_path() -> None:
    response = validate(ConfigKind.UNIVERSE, VALID_UNIVERSE_PAYLOAD)

    assert response.valid is True


def test_validate_loose_kinds_always_pass() -> None:
    # strategy / regime have no Pydantic counterpart — HPO is validated
    # against HPOConfig and has its own dedicated tests below.
    for kind in (ConfigKind.STRATEGY, ConfigKind.REGIME):
        response = validate(kind, {"arbitrary": "shape"})
        assert response.valid is True
        assert response.errors == []


def test_validate_hpo_happy_path() -> None:
    payload = {"study_name": "demo", "n_trials": 5}
    response = validate(ConfigKind.HPO, payload)

    assert response.valid is True
    assert response.errors == []


def test_validate_hpo_rejects_missing_study_name() -> None:
    response = validate(ConfigKind.HPO, {"n_trials": 5})

    assert response.valid is False
    locs = [tuple(err.loc) for err in response.errors]
    assert ("study_name",) in locs


def test_validate_hpo_rejects_extra_fields() -> None:
    response = validate(ConfigKind.HPO, {"study_name": "demo", "garbage": 1})

    assert response.valid is False
    locs = [tuple(err.loc) for err in response.errors]
    assert ("garbage",) in locs


def test_validate_missing_required_field() -> None:
    payload = make_valid_experiment_payload()
    del payload["data"]

    response = validate(ConfigKind.EXPERIMENT, payload)

    assert response.valid is False
    locs = [tuple(err.loc) for err in response.errors]
    assert ("data",) in locs


def test_validate_bad_type() -> None:
    payload = make_valid_experiment_payload()
    payload["seed"] = "not-an-int"

    response = validate(ConfigKind.EXPERIMENT, payload)

    assert response.valid is False
    assert any(err.loc == ["seed"] for err in response.errors)


def test_validate_invalid_enum() -> None:
    payload = make_valid_experiment_payload()
    data_field = payload["data"]
    assert isinstance(data_field, dict)
    data_field["interval"] = "fortnightly"

    response = validate(ConfigKind.EXPERIMENT, payload)

    assert response.valid is False
    assert any("interval" in err.loc for err in response.errors)


def test_validate_experiment_missing_required_strategy_param() -> None:
    """Pydantic accepts ``strategy.params`` as a generic dict; the post-pass
    pre-flight (via ``describe_strategy``) catches missing required ctor
    kwargs so a doomed config never spawns a subprocess."""
    payload = make_valid_experiment_payload()
    # ``ReturnForecast`` ctor declares ``feature_columns: list[str]`` (required).
    payload["strategy"] = {"name": "ReturnForecast", "params": {}}

    response = validate(ConfigKind.EXPERIMENT, payload)

    assert response.valid is False
    locs = [tuple(err.loc) for err in response.errors]
    assert ("strategy", "params", "feature_columns") in locs
    assert all(err.type == "missing" for err in response.errors)


def test_validate_experiment_required_strategy_param_filled_passes() -> None:
    payload = make_valid_experiment_payload()
    payload["strategy"] = {
        "name": "ReturnForecast",
        "params": {"feature_columns": ["rsi_14", "macd_12_26_9"]},
    }

    response = validate(ConfigKind.EXPERIMENT, payload)

    assert response.valid is True


def test_validate_experiment_extra_field_rejected_before_strategy_check() -> None:
    """A Pydantic-level error short-circuits the strategy-completeness pre-flight."""
    payload = make_valid_experiment_payload()
    payload["typo_field"] = "oops"

    response = validate(ConfigKind.EXPERIMENT, payload)

    assert response.valid is False
    assert any("typo_field" in err.loc for err in response.errors)


def test_validate_extra_field_rejected() -> None:
    payload = make_valid_experiment_payload()
    payload["typo_field"] = "oops"

    response = validate(ConfigKind.EXPERIMENT, payload)

    assert response.valid is False
    assert any("typo_field" in err.loc for err in response.errors)


def test_validate_unknown_strategy_name() -> None:
    payload = make_valid_experiment_payload()
    payload["strategy"] = {"name": "NoSuchStrategy", "params": {}}

    response = validate(ConfigKind.EXPERIMENT, payload)

    assert response.valid is False


def test_list_configs_returns_sorted_stems(config_root: Path) -> None:
    entries = list_configs(config_root, ConfigKind.UNIVERSE)

    assert [e.name for e in entries] == ["qqq", "spy"]


def test_list_configs_missing_dir_returns_empty(config_root: Path) -> None:
    # `experiment` has no top-level dir under config/.
    entries = list_configs(config_root, ConfigKind.EXPERIMENT)

    assert entries == []


def test_read_config_round_trip(config_root: Path) -> None:
    detail = read_config(config_root, ConfigKind.UNIVERSE, "spy")

    assert detail.name == "spy"
    assert detail.parsed == VALID_UNIVERSE_PAYLOAD
    assert detail.parse_error is None
    assert "tickers" in detail.raw


def test_read_config_missing_raises(config_root: Path) -> None:
    with pytest.raises(ConfigNotFoundError):
        read_config(config_root, ConfigKind.UNIVERSE, "nope")


def test_read_config_malformed_yaml_returns_parse_error(config_root: Path) -> None:
    (config_root / "universes" / "broken.yaml").write_text(
        "tickers: [SPY\n  unterminated", encoding="utf-8"
    )

    detail = read_config(config_root, ConfigKind.UNIVERSE, "broken")

    assert detail.parsed is None
    assert detail.parse_error is not None


def test_read_config_non_mapping_top_level(config_root: Path) -> None:
    (config_root / "universes" / "scalar.yaml").write_text("just_a_string\n", encoding="utf-8")

    detail = read_config(config_root, ConfigKind.UNIVERSE, "scalar")

    assert detail.parsed is None
    assert detail.parse_error is not None
