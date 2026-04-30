"""Unit tests for :func:`src.core.config_overrides.apply_overrides`.

Behavioural surface:
* Replace an existing scalar (top-level + nested).
* Replace an existing list (full list, not per-index).
* YAML-typed value coercion (int, bool, date-as-string, quoted-as-string).
* Multiple overrides apply in order, last-wins on collision.
* Values may contain ``=`` (split on first ``=`` only).
* Typo guard: missing or non-dict intermediate key raises ValueError.
* Malformed input (no ``=``, empty key) raises ValueError.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from src.core.config_overrides import apply_overrides


def _base_payload() -> dict[str, object]:
    return {
        "name": "demo",
        "seed": 42,
        "data": {
            "source": {"name": "yfinance", "params": {"cache_dir": "data/cache"}},
            "tickers": ["SPY"],
            "start": "2018-01-02",
            "end": "2024-12-31",
        },
        "validation": {"n_splits": 4, "expanding": True},
    }


def test_replace_top_level_scalar() -> None:
    out = apply_overrides(_base_payload(), ["name=other"])
    assert out["name"] == "other"


def test_replace_nested_scalar() -> None:
    out = apply_overrides(_base_payload(), ["data.start=2020-01-01"])
    assert out["data"]["start"] == _dt.date(2020, 1, 1)  # type: ignore[index]


def test_replace_list() -> None:
    out = apply_overrides(_base_payload(), ["data.tickers=[QQQ, IWM]"])
    assert out["data"]["tickers"] == ["QQQ", "IWM"]  # type: ignore[index]


def test_int_coercion() -> None:
    out = apply_overrides(_base_payload(), ["seed=99"])
    assert out["seed"] == 99
    assert isinstance(out["seed"], int)


def test_bool_coercion() -> None:
    out = apply_overrides(_base_payload(), ["validation.expanding=false"])
    assert out["validation"]["expanding"] is False  # type: ignore[index]


def test_quoted_string_stays_string() -> None:
    out = apply_overrides(_base_payload(), ['name="123"'])
    assert out["name"] == "123"
    assert isinstance(out["name"], str)


def test_multiple_overrides_apply_in_order() -> None:
    out = apply_overrides(
        _base_payload(),
        ["seed=1", "data.tickers=[QQQ]", "seed=2"],
    )
    assert out["seed"] == 2
    assert out["data"]["tickers"] == ["QQQ"]  # type: ignore[index]


def test_value_may_contain_equals() -> None:
    out = apply_overrides(_base_payload(), ["name=key=value"])
    assert out["name"] == "key=value"


def test_missing_intermediate_raises() -> None:
    with pytest.raises(ValueError, match=r"'dat'.*missing or not a dict"):
        apply_overrides(_base_payload(), ["dat.tickers=[QQQ]"])


def test_intermediate_is_list_raises() -> None:
    with pytest.raises(ValueError, match=r"'tickers'.*missing or not a dict"):
        apply_overrides(_base_payload(), ["data.tickers.0=QQQ"])


def test_no_equals_raises() -> None:
    with pytest.raises(ValueError, match=r"missing '='"):
        apply_overrides(_base_payload(), ["data.tickers"])


def test_empty_key_raises() -> None:
    with pytest.raises(ValueError, match=r"empty key"):
        apply_overrides(_base_payload(), ["=foo"])


def test_deep_nested_replace() -> None:
    out = apply_overrides(_base_payload(), ["data.source.params.cache_dir=tests/fixtures"])
    assert out["data"]["source"]["params"]["cache_dir"] == "tests/fixtures"  # type: ignore[index]


def test_pydantic_round_trip_via_overrides() -> None:
    """Overrides survive pydantic re-validation when types match the schema."""
    from src.core.config import ExperimentConfig, load_experiment_config

    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    payload = cfg.model_dump(mode="json")
    overridden = apply_overrides(
        payload,
        ["data.tickers=[QQQ]", "seed=999"],
    )
    re_validated = ExperimentConfig.model_validate(overridden)
    assert re_validated.data.tickers == ["QQQ"]
    assert re_validated.seed == 999
