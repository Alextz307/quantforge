"""Domain-specific exceptions for the quant trading framework."""

from __future__ import annotations


class LeakageError(RuntimeError):
    """Raised when temporal data leakage (lookahead bias) is detected.

    This indicates a programming error where future data has leaked
    into a computation that should only use past/present data.
    """


class DataQualityError(ValueError):
    """Raised when input data fails quality validation.

    Examples: OHLCV ordering violations, excessive gaps, price outliers.
    """
