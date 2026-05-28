"""
Generic typed component registry for pluggable components.

Uses TYPE_CHECKING imports to avoid circular dependencies at runtime.
The type parameter T provides compile-time safety via mypy.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from functools import cache
from importlib import import_module
from pkgutil import iter_modules
from types import UnionType
from typing import TYPE_CHECKING, Union, get_args, get_origin, get_type_hints

__all__ = [
    "ComponentRegistry",
    "autoload_package",
    "strategy_registry",
    "model_registry",
    "classifier_registry",
    "data_source_registry",
    "feature_registry",
]

if TYPE_CHECKING:
    from src.core.config import ComponentConfig
    from src.data.interface import IDataSource
    from src.features.interface import IFeaturePipeline
    from src.models.interface import IClassifier, IPredictor
    from src.strategies.interface import IStrategy


def _enum_type_in_annotation(annotation: object) -> type[Enum] | None:
    """
    Return the first Enum subclass in ``annotation`` (walks unions).

    Handles direct ``E``, ``Optional[E]``, ``E | None``, and ``Union[E, ...]``.
    """

    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            found = _enum_type_in_annotation(arg)
            if found is not None:
                return found
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


@cache
def _ctor_hints(cls: type) -> dict[str, object]:
    try:
        return dict(get_type_hints(cls.__init__))  # type: ignore[misc]
    except (NameError, TypeError):
        return {}


def _coerce_enum_kwargs(cls: type, kwargs: dict[str, object]) -> dict[str, object]:
    """
    Coerce string kwargs to Enum members when the ctor annotation expects them.

    The registry is the YAML/dict→ctor boundary: ``ComponentConfig.params``
    is dict-typed, so Enum coercion lives here rather than in leaf ctors.
    """

    hints = _ctor_hints(cls)
    out: dict[str, object] = {}
    for name, value in kwargs.items():
        enum_cls = _enum_type_in_annotation(hints.get(name))
        if enum_cls is not None and isinstance(value, str) and not isinstance(value, enum_cls):
            try:
                out[name] = enum_cls(value)
            except ValueError:
                out[name] = value
        else:
            out[name] = value
    return out


class ComponentRegistry[T]:
    """
    Generic typed registry for pluggable components.

    Components are registered via decorator and can be created by name.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """
        Decorator that registers a component class.
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(
                    f"Component '{name}' is already registered; fix by choosing a "
                    f"distinct registry name or removing the duplicate decorator."
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """
        Get a registered component class by name.
        """

        if name not in self._registry:
            raise KeyError(
                f"Component '{name}' not found; available: "
                f"{list(self._registry.keys())}. Fix by registering the component "
                f"(import its module so the @register decorator fires) or by "
                f"choosing one of the listed names."
            )
        return self._registry[name]

    def list_all(self) -> list[str]:
        """
        List all registered component names — including test stubs.

        Use ``list_public()`` for the production-facing surface.
        """

        return list(self._registry.keys())

    def list_public(self) -> list[str]:
        """
        List registered components excluding ``_``-prefixed test stubs.

        Test fixtures register stubs (``_MultiFeatureTestStub``,
        ``_BothFlagsStub``) into the same global registries that production
        loads. The ``_`` prefix is the project-wide convention for
        "internal / not user-visible" — webapp APIs and any other
        production-facing introspector should use this method so a stub
        never leaks into a list, dropdown, or external response.
        """

        return [name for name in self._registry if not name.startswith("_")]

    def create(self, name: str, **kwargs: object) -> T:
        """
        Create an instance of a registered component.

        String kwargs whose ctor annotation is an Enum (or Enum union) are
        coerced to Enum members before the call so YAML/dict params don't
        force leaf ctors to accept ``Enum | str`` unions.
        """

        cls = self.get(name)
        return cls(**_coerce_enum_kwargs(cls, kwargs))

    def create_from_config(self, config: ComponentConfig) -> T:
        """
        Create an instance from a :class:`ComponentConfig`.

        Equivalent to ``self.create(config.name, **config.params)``, but gives
        callers a single uniform entry point when dispatching pluggable
        components from validated YAML.
        """

        return self.create(config.name, **config.params)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


def autoload_package(
    pkg_path: list[str],
    pkg_name: str,
    *,
    skip: tuple[str, ...] = ("interface",),
) -> None:
    """
    Import every non-private, non-``skip`` module in a package.

    Fires each module's ``@..._registry.register`` decorator side-effects so
    registry-populating packages (``src.strategies`` / ``src.data`` /
    ``src.features``) can keep their ``__init__.py`` to a single call —
    dropping a new file in the package auto-registers its component.

    ``skip`` excludes module names that live alongside concrete components
    but are not themselves registrable (typically ``interface`` holding an
    ABC / Protocol).
    """

    for info in iter_modules(pkg_path):
        if not info.name.startswith("_") and info.name not in skip:
            import_module(f"{pkg_name}.{info.name}")


strategy_registry: ComponentRegistry[IStrategy] = ComponentRegistry()
model_registry: ComponentRegistry[IPredictor] = ComponentRegistry()
classifier_registry: ComponentRegistry[IClassifier] = ComponentRegistry()
data_source_registry: ComponentRegistry[IDataSource] = ComponentRegistry()
feature_registry: ComponentRegistry[IFeaturePipeline] = ComponentRegistry()
