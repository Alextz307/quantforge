"""Unit tests for `webapp.backend.app.services.strategy_service`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.device import available_devices
from src.core.registry import strategy_registry
from webapp.backend.app.schemas.strategies import ParamKind
from webapp.backend.app.services.strategy_service import (
    describe_strategy,
    get_canonical_strategy_params,
)


@pytest.fixture(autouse=True)
def _reset_device_cache() -> None:
    available_devices.cache_clear()


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

    # ``interval`` is framework-managed (duplicates ``data.interval``);
    # showing it would let the user desync the two. Never surfaces as a form field.
    assert "interval" not in by_name


def test_str_list_annotations_classify_as_str_list() -> None:
    """``list[str]`` / ``tuple[str, ...]`` / ``Sequence[str]`` render as a
    comma-or-space separated text input — far better UX than hand-writing JSON."""
    schema = describe_strategy("ReturnForecast")
    by_name = {p.name: p for p in schema.params}

    # ``feature_columns: list[str]`` — required, list-of-str.
    assert by_name["feature_columns"].kind is ParamKind.STR_LIST
    assert by_name["feature_columns"].required is True
    assert by_name["feature_columns"].nullable is False

    # CrossAssetMomentum has ``feature_tickers: Sequence[str]``.
    cam = describe_strategy("CrossAssetMomentum")
    cam_params = {p.name: p for p in cam.params}
    assert cam_params["feature_tickers"].kind is ParamKind.STR_LIST


def test_optional_enum_unwraps_to_typed_dropdown() -> None:
    """``T | None`` peels to T's kind with ``nullable=True`` so the form can
    render Device etc. as a typed dropdown instead of a JSON textarea."""
    schema = describe_strategy("ReturnForecast")

    by_name = {p.name: p for p in schema.params}

    # ``lstm_device: Device | None = None`` — Optional[Enum] → ENUM + nullable.
    assert by_name["lstm_device"].kind is ParamKind.ENUM
    assert by_name["lstm_device"].nullable is True
    assert by_name["lstm_device"].required is False
    assert by_name["lstm_device"].choices is not None
    assert "cpu" in by_name["lstm_device"].choices


def test_device_choices_pruned_to_host_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a CPU-only host the device dropdown only offers ``auto`` + ``cpu``,
    so a user can't pick ``cuda`` / ``mps`` and hit a runtime RuntimeError."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("src.core.device._mps_available", lambda: False)

    schema = describe_strategy("ReturnForecast")
    by_name = {p.name: p for p in schema.params}

    assert by_name["lstm_device"].choices == ["auto", "cpu"]


def test_device_choices_include_cuda_when_host_has_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.core.device._mps_available", lambda: False)

    schema = describe_strategy("ReturnForecast")
    by_name = {p.name: p for p in schema.params}

    assert "cuda" in (by_name["lstm_device"].choices or [])
    assert "mps" not in (by_name["lstm_device"].choices or [])


def test_xgboost_strategies_drop_mps_from_device_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XGBoost-backed strategies (``uses_xgboost = True``) must not offer
    ``mps`` even on Apple Silicon — XGBoost's GPU path is NVIDIA-only,
    so picking MPS would raise ``ValueError`` at runtime."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("src.core.device._mps_available", lambda: True)

    xgb_schema = describe_strategy("MomentumGatekeeper")
    xgb_params = {p.name: p for p in xgb_schema.params}
    assert "mps" not in (xgb_params["device"].choices or [])
    assert "cpu" in (xgb_params["device"].choices or [])

    # Sanity: torch-backed strategy on the same host still offers MPS.
    torch_schema = describe_strategy("ReturnForecast")
    torch_params = {p.name: p for p in torch_schema.params}
    assert "mps" in (torch_params["lstm_device"].choices or [])


def test_canonical_params_loads_from_yaml(tmp_path: Path) -> None:
    """``get_canonical_strategy_params`` reads ``strategy.params`` from the
    canonical YAML so the form can pre-fill working defaults. Hidden params
    (``interval``) are filtered out."""
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    (strategies_dir / "adaptive_bollinger.yaml").write_text(
        "strategy:\n  name: AdaptiveBollinger\n  params:\n"
        "    window: 25\n    k: 1.5\n    interval: hour\n",
        encoding="utf-8",
    )
    params = get_canonical_strategy_params(tmp_path, "AdaptiveBollinger")
    assert params == {"window": 25, "k": 1.5}


def test_canonical_params_returns_none_when_yaml_missing(tmp_path: Path) -> None:
    assert get_canonical_strategy_params(tmp_path, "AdaptiveBollinger") is None


def test_canonical_params_handles_camel_case_strategy_name(tmp_path: Path) -> None:
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    (strategies_dir / "cross_asset_momentum.yaml").write_text(
        "strategy:\n  name: CrossAssetMomentum\n  params:\n"
        "    primary_ticker: SPY\n    feature_tickers: [QQQ, TLT]\n",
        encoding="utf-8",
    )
    params = get_canonical_strategy_params(tmp_path, "CrossAssetMomentum")
    assert params is not None
    assert params["primary_ticker"] == "SPY"
    assert params["feature_tickers"] == ["QQQ", "TLT"]


def test_non_optional_params_have_nullable_false() -> None:
    schema = describe_strategy("AdaptiveBollinger")
    by_name = {p.name: p for p in schema.params}
    assert by_name["window"].nullable is False
    assert by_name["k"].nullable is False


def test_every_registered_strategy_describes_cleanly() -> None:
    # Iterate ``list_public()`` so test stubs (``_``-prefix) registered by
    # ``tests/_strategy_stubs.py`` don't leak into webapp test pollution.
    for name in strategy_registry.list_public():
        schema = describe_strategy(name)
        assert schema.name == name
        assert schema.qualname.startswith("src.strategies.")
        assert all(p.name not in {"self", "interval"} for p in schema.params)
