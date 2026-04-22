"""Auto-imports every concrete feature pipeline so ``@feature_registry.register``
decorators fire at package-import time. Drop a new pipeline file here and
it registers automatically."""

from __future__ import annotations

from src.core.registry import autoload_package

autoload_package(__path__, __name__)
