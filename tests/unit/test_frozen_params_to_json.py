"""
Tests for :func:`frozen_params_to_json` — composite ctor-kwargs serializer.

Pins the three conversions every composite relies on: tuple→list,
Enum→.value, and ``omit`` drop. A regression here silently corrupts every
composite's ``config.json`` on save.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pytest

from src.core.persistence import frozen_params_to_json


class _SampleEnum(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"


@dataclass(frozen=True)
class _SampleParams:
    name: str
    count: int
    weights: tuple[float, ...]
    mode: _SampleEnum
    device: str | None = None


def _make_params() -> _SampleParams:
    return _SampleParams(
        name="x",
        count=7,
        weights=(1.0, 2.0, 3.0),
        mode=_SampleEnum.BETA,
        device="cpu",
    )


class TestFrozenParamsToJson:
    def test_tuple_becomes_list(self) -> None:
        d = frozen_params_to_json(_make_params())
        assert d["weights"] == [1.0, 2.0, 3.0]

    def test_enum_becomes_value_string(self) -> None:
        d = frozen_params_to_json(_make_params())
        assert d["mode"] == "beta"

    def test_omit_drops_listed_keys(self) -> None:
        d = frozen_params_to_json(_make_params(), omit=("device",))
        assert "device" not in d

    def test_omit_of_missing_key_is_noop(self) -> None:
        d = frozen_params_to_json(_make_params(), omit=("nonexistent",))
        assert d["name"] == "x"

    def test_primitives_preserved(self) -> None:
        d = frozen_params_to_json(_make_params())
        assert d["name"] == "x"
        assert d["count"] == 7

    def test_rejects_non_dataclass(self) -> None:
        with pytest.raises(TypeError, match="dataclass INSTANCE"):
            frozen_params_to_json({"name": "x"})

    def test_rejects_dataclass_class(self) -> None:
        """
        Passing the CLASS instead of an instance is a common mistake — catch it.
        """

        with pytest.raises(TypeError, match="dataclass INSTANCE"):
            frozen_params_to_json(_SampleParams)
