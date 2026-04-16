"""
Python wrapper around the compiled ``quant_engine`` C++ extension.
"""

from __future__ import annotations

from quant_engine.quant_engine import hello

__all__: list = ["hello"]
