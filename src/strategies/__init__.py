"""Auto-imports every concrete strategy so ``@strategy_registry.register``
decorators fire at package-import time. Drop a new strategy file here and
it registers automatically."""

from __future__ import annotations

from src.core.registry import autoload_package

autoload_package(__path__, __name__)
