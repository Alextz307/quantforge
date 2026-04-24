"""Auto-imports every concrete model / classifier so ``@model_registry`` and
``@classifier_registry`` decorators fire at package-import time. Drop a new
model file here and it registers automatically."""

from __future__ import annotations

from src.core.registry import autoload_package

autoload_package(__path__, __name__)
