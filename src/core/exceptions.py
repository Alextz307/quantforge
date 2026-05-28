"""
Domain-specific exceptions for the quant trading framework.
"""

from __future__ import annotations


class LeakageError(RuntimeError):
    """
    Raised when temporal data leakage (lookahead bias) is detected.

    This indicates a programming error where future data has leaked
    into a computation that should only use past/present data.
    """


class DataQualityError(ValueError):
    """
    Raised when input data fails ingestion-time quality validation.

    Covers the cases checked by ``src.data.validator.validate_bars``: NaN in
    OHLCV columns, non-positive OHLC prices, negative volume, OHLC ordering
    violations (``high >= max(open, close)``, ``low <= min(open, close)``,
    ``high >= low``), duplicate timestamps, and empty / malformed frames.
    Also raised by data-source adapters on structural failures (e.g.
    ``YFinanceSource`` on an empty response).
    """


def guard_scaler_fit_once(scaler: object | None, component: str) -> None:
    """
    Raise ``LeakageError`` if ``scaler`` has already been fitted.

    Centralizes the fit-once guard pattern shared by feature pipelines and
    composite models so a second ``fit()`` call on training data is caught
    instead of silently re-fitting on (potentially) test data.
    """

    if scaler is not None:
        raise LeakageError(
            f"{component}.fit() called twice; the scaler must only be fit on "
            f"training data. Fix by reconstructing a fresh instance per fold "
            f"(the typical walk-forward pattern) instead of refitting."
        )
