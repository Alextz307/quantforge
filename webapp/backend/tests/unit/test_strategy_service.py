"""Unit tests for `webapp.backend.app.services.strategy_service`."""

from __future__ import annotations

import pytest

from src.core.registry import strategy_registry
from webapp.backend.app.schemas.strategies import ParamKind
from webapp.backend.app.services.strategy_service import describe_strategy


def test_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError):
        describe_strategy("NoSuchStrategy")


def test_adaptive_bollinger_classification() -> None:
    schema = describe_strategy("AdaptiveBollinger")

    by_name = {p.name: p for p in schema.params}

    assert by_name["window"].kind is ParamKind.INT
    assert by_name["window"].default == 20
    assert by_name["window"].required is False

    assert by_name["k"].kind is ParamKind.FLOAT
    assert by_name["k"].default == 2.0

    assert by_name["interval"].kind is ParamKind.ENUM
    assert by_name["interval"].choices is not None
    assert "daily" in by_name["interval"].choices
    assert by_name["interval"].default == "daily"

    # ``pretrained_leaves`` is composite-injected by the experiment builder
    # — the form must never expose it.
    assert "pretrained_leaves" not in by_name


def test_complex_types_fall_back_to_complex_kind() -> None:
    schema = describe_strategy("ReturnForecast")

    by_name = {p.name: p for p in schema.params}

    # ``feature_columns: list[str]`` — required, complex.
    assert by_name["feature_columns"].kind is ParamKind.COMPLEX
    assert by_name["feature_columns"].required is True

    # ``lstm_device: Device | None`` — Optional, complex.
    assert by_name["lstm_device"].kind is ParamKind.COMPLEX


def test_every_registered_strategy_describes_cleanly() -> None:
    for name in strategy_registry.list_all():
        schema = describe_strategy(name)
        assert schema.name == name
        assert schema.qualname.startswith("src.strategies.")
        assert all(p.name not in {"self", "pretrained_leaves"} for p in schema.params)
