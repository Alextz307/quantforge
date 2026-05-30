"""
Tests for ComponentRegistry + package auto-discovery completeness.
"""

from __future__ import annotations

from enum import StrEnum
from pkgutil import iter_modules
from types import ModuleType

import pytest

from src.core.registry import (
    ComponentRegistry,
    data_source_registry,
    feature_registry,
    strategy_registry,
)

SAMPLE_VALUE = 42
DIRECT_INIT_VALUE = 10


class DummyBase:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class _Mode(StrEnum):
    FAST = "fast"
    SLOW = "slow"


class _WithMode:
    def __init__(self, mode: _Mode = _Mode.FAST) -> None:
        self.mode = mode


class _WithOptionalMode:
    def __init__(self, mode: _Mode | None = None) -> None:
        self.mode = mode


class _WithText:
    def __init__(self, label: str = "x") -> None:
        self.label = label


class _StrictWithMode:
    """
    Mirrors a leaf ctor that itself rejects non-Enum types - used to verify
    that bad-string values fall through to the ctor's own error path rather
    than being silently swallowed by the coercion helper.
    """

    def __init__(self, mode: _Mode = _Mode.FAST) -> None:
        if not isinstance(mode, _Mode):
            raise ValueError(f"mode must be a _Mode, got {type(mode).__name__}")
        self.mode = mode


class TestComponentRegistry:
    def test_register_and_get(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()

        @registry.register("test_component")
        class TestComponent(DummyBase):
            pass

        assert registry.get("test_component") is TestComponent

    def test_create_instance(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()

        @registry.register("test_component")
        class TestComponent(DummyBase):
            pass

        instance = registry.create("test_component", value=SAMPLE_VALUE)
        assert isinstance(instance, TestComponent)
        assert instance.value == SAMPLE_VALUE

    def test_list_all(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()

        @registry.register("alpha")
        class Alpha(DummyBase):
            pass

        @registry.register("beta")
        class Beta(DummyBase):
            pass

        names = registry.list_all()
        assert set(names) == {"alpha", "beta"}

    def test_get_nonexistent_raises_key_error(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_create_nonexistent_raises_key_error(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.create("nonexistent")

    def test_duplicate_registration_raises(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()

        @registry.register("dup")
        class First(DummyBase):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("dup")
            class Second(DummyBase):
                pass

    def test_contains(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()

        @registry.register("exists")
        class Exists(DummyBase):
            pass

        assert "exists" in registry
        assert "missing" not in registry

    def test_len(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()
        assert len(registry) == 0

        @registry.register("one")
        class One(DummyBase):
            pass

        assert len(registry) == 1

        @registry.register("two")
        class Two(DummyBase):
            pass

        assert len(registry) == 2

    def test_list_public_excludes_underscore_prefixed_test_stubs(self) -> None:
        """
        ``_``-prefix is the project-wide convention for "test-only / not
        user-visible". Production-facing introspection (webapp APIs, form
        dropdowns) must use ``list_public()`` so registered stubs from
        ``tests/_strategy_stubs.py`` never leak into the surface.
        """

        registry: ComponentRegistry[DummyBase] = ComponentRegistry()

        @registry.register("alpha")
        class Alpha(DummyBase):
            pass

        @registry.register("_TestStub")
        class _Stub(DummyBase):
            pass

        assert set(registry.list_all()) == {"alpha", "_TestStub"}
        assert registry.list_public() == ["alpha"]

    def test_decorator_returns_original_class(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()

        @registry.register("test")
        class Original(DummyBase):
            pass

        assert Original.__name__ == "Original"
        direct = Original(value=DIRECT_INIT_VALUE)
        assert direct.value == DIRECT_INIT_VALUE


class TestEnumCoercion:
    """
    ``create()`` coerces raw-string kwargs to Enum members when the ctor
    annotation is an Enum (or ``Enum | None`` / ``Union[Enum, ...]``).

    This is the boundary fix that lets dict-typed ``ComponentConfig.params``
    feed ``Strategy(interval="daily", ...)``-style ctors without forcing leaf
    classes to accept ``Enum | str`` unions.
    """

    def test_string_value_is_coerced_to_enum_member(self) -> None:
        registry: ComponentRegistry[_WithMode] = ComponentRegistry()
        registry.register("modal")(_WithMode)
        instance = registry.create("modal", mode="slow")
        assert isinstance(instance.mode, _Mode)
        assert instance.mode is _Mode.SLOW

    def test_optional_enum_string_is_coerced(self) -> None:
        registry: ComponentRegistry[_WithOptionalMode] = ComponentRegistry()
        registry.register("opt")(_WithOptionalMode)
        instance = registry.create("opt", mode="fast")
        assert instance.mode is _Mode.FAST

    def test_optional_enum_none_passes_through(self) -> None:
        registry: ComponentRegistry[_WithOptionalMode] = ComponentRegistry()
        registry.register("opt")(_WithOptionalMode)
        instance = registry.create("opt", mode=None)
        assert instance.mode is None

    def test_already_enum_instance_passes_through(self) -> None:
        registry: ComponentRegistry[_WithMode] = ComponentRegistry()
        registry.register("modal")(_WithMode)
        instance = registry.create("modal", mode=_Mode.SLOW)
        assert instance.mode is _Mode.SLOW

    def test_non_enum_annotation_string_passes_through(self) -> None:
        registry: ComponentRegistry[_WithText] = ComponentRegistry()
        registry.register("text")(_WithText)
        instance = registry.create("text", label="anything")
        assert instance.label == "anything"

    def test_invalid_enum_value_falls_through_to_ctor(self) -> None:
        registry: ComponentRegistry[_StrictWithMode] = ComponentRegistry()
        registry.register("modal")(_StrictWithMode)
        with pytest.raises(ValueError, match="must be a _Mode"):
            registry.create("modal", mode="bogus")


def _count_non_interface_modules(pkg: ModuleType) -> int:
    """
    Non-private modules in ``pkg`` excluding ``interface``.
    """

    return sum(
        1
        for info in iter_modules(pkg.__path__)
        if not info.name.startswith("_") and info.name != "interface"
    )


class TestPackageAutoDiscovery:
    """
    Importing ``src.strategies`` / ``src.data`` / ``src.features`` populates
    the matching registry. For ``src.strategies`` this is an exact-match check:
    every non-interface module IS a concrete strategy, so the count of modules
    must equal the registry size - a new strategy file without the
    ``@strategy_registry.register`` decorator trips this immediately rather
    than surfacing later as a cryptic ``ValidationError`` on config load.

    For ``src.data`` / ``src.features`` the match is looser because those
    packages mix concrete-source modules with stateless utilities (cache,
    normalizer, validator). The loose check only asserts that auto-discovery
    runs at all - which catches an empty ``__init__.py`` regression.
    """

    def test_strategies_registry_matches_package_contents_exactly(self) -> None:
        import src.strategies

        # ``list_public()`` filters underscore-prefixed test stubs; production
        # strategies use neither underscored filenames nor registry names.
        assert len(strategy_registry.list_public()) == _count_non_interface_modules(src.strategies)

    def test_data_sources_populated_after_package_import(self) -> None:
        import src.data  # noqa: F401

        assert len(data_source_registry) >= 1

    def test_feature_pipelines_populated_after_package_import(self) -> None:
        import src.features  # noqa: F401

        assert len(feature_registry) >= 1
