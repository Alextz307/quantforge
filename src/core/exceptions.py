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


def guard_scaler_fit_once(scaler: object | None, component: str) -> None:
    """Raise ``LeakageError`` if ``scaler`` has already been fitted.

    Centralizes the fit-once guard pattern shared by feature pipelines and
    composite models so a second ``fit()`` call on training data is caught
    instead of silently re-fitting on (potentially) test data.
    """
    if scaler is not None:
        raise LeakageError(
            f"{component}.fit() called twice. Scaler must only be fit on training data."
        )
