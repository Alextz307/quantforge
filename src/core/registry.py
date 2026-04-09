"""Generic typed component registry for pluggable components.

Uses TYPE_CHECKING imports to avoid circular dependencies at runtime.
The type parameter T provides compile-time safety via mypy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

__all__ = [
    "ComponentRegistry",
    "strategy_registry",
    "model_registry",
    "classifier_registry",
    "data_source_registry",
    "feature_registry",
]

if TYPE_CHECKING:
    from src.data.interface import IDataSource
    from src.features.interface import IFeaturePipeline
    from src.models.interface import IClassifier, IPredictor
    from src.strategies.interface import IStrategy


class ComponentRegistry[T]:
    """Generic typed registry for pluggable components.

    Components are registered via decorator and can be created by name.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator that registers a component class."""

        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(f"Component '{name}' is already registered")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Get a registered component class by name."""
        if name not in self._registry:
            raise KeyError(
                f"Component '{name}' not found. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list_all(self) -> list[str]:
        """List all registered component names."""
        return list(self._registry.keys())

    def create(self, name: str, **kwargs: object) -> T:
        """Create an instance of a registered component."""
        cls = self.get(name)
        return cls(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# Global registries — typed via TYPE_CHECKING imports (no runtime circular deps)
strategy_registry: ComponentRegistry[IStrategy] = ComponentRegistry()
model_registry: ComponentRegistry[IPredictor] = ComponentRegistry()
classifier_registry: ComponentRegistry[IClassifier] = ComponentRegistry()
data_source_registry: ComponentRegistry[IDataSource] = ComponentRegistry()
feature_registry: ComponentRegistry[IFeaturePipeline] = ComponentRegistry()
