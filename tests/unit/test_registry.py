"""Tests for ComponentRegistry."""

from __future__ import annotations

import pytest

from src.core.registry import ComponentRegistry


class DummyBase:
    def __init__(self, value: int = 0) -> None:
        self.value = value


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

        instance = registry.create("test_component", value=42)
        assert isinstance(instance, TestComponent)
        assert instance.value == 42

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

    def test_decorator_returns_original_class(self) -> None:
        registry: ComponentRegistry[DummyBase] = ComponentRegistry()

        @registry.register("test")
        class Original(DummyBase):
            pass

        # The decorator should return the class unchanged
        assert Original.__name__ == "Original"
        direct = Original(value=10)
        assert direct.value == 10
