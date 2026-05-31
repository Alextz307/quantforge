"""
Shared domain utilities for the quant trading framework.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import quant_engine
from src.core.constants import TRADING_DAYS_PER_YEAR
from src.core.types import Interval


def compute_log_returns(close: pd.Series[float]) -> pd.Series[float]:
    """
    Compute log returns from a close-price series.

    Equivalent to ``log(close[t] / close[t-1])``.  The first value is
    NaN (no prior close).  Callers should ``.dropna()`` when needed.
    """

    result: pd.Series[float] = np.log1p(close.pct_change())  # type: ignore[assignment]
    return result


def validate_open_unit_interval(value: float, name: str) -> None:
    """
    Ensure ``value`` lies in the open interval ``(0, 1)``; raise ``ValueError`` otherwise.
    """

    if not (0.0 < value < 1.0):
        raise ValueError(
            f"{name} must be in (0, 1), got {value}; fix by passing a strictly "
            f"positive fraction below 1.0 (typical: 0.2 for a 20% split)."
        )


def annualized_garman_klass(
    bars: pd.DataFrame, *, window: int, interval: Interval
) -> pd.Series[float]:
    """
    Annualized Garman-Klass realized volatility at the caller's interval.

    The underlying C++ estimator annualizes assuming daily bars; we rescale
    so ``Interval.HOUR`` and friends land on the same annualized horizon as
    the rest of the framework. Shared between ``VolatilityTargetingStrategy``
    and the standalone training dispatcher so the two callsites cannot drift.

    ``bars`` must carry ``open`` / ``high`` / ``low`` / ``close`` columns.
    Leading ``window-1`` values are NaN (warmup); callers should ``.dropna()``
    before using the result as a training target.
    """

    gk = quant_engine.GarmanKlass(window).compute(
        bars["open"].to_numpy(dtype=float, copy=False),
        bars["high"].to_numpy(dtype=float, copy=False),
        bars["low"].to_numpy(dtype=float, copy=False),
        bars["close"].to_numpy(dtype=float, copy=False),
    )
    interval_scale = math.sqrt(interval.annualization_factor() / TRADING_DAYS_PER_YEAR)
    return pd.Series(gk * interval_scale, index=bars.index)


def next_bar_direction(close: pd.Series[float]) -> pd.Series[int]:
    """
    Binary next-bar-up target (1 = up, 0 = down); final row excluded.

    Shared between ``DirectionalClassifier`` training targets (standalone
    dispatcher and MomentumGatekeeperStrategy's in-strategy batch builder)
    so the target formula cannot drift across call sites. The last row has
    no ``t+1`` close - returning its comparison would be a leakage hazard
    - so it's dropped, not filled.
    """

    direction: pd.Series[int] = (close.shift(-1) > close).astype(int).iloc[:-1]
    return direction


def align_features_for_directional_target(
    features: pd.DataFrame,
    close: pd.Series[float],
) -> tuple[pd.DataFrame, pd.Series[int]]:
    """
    Align ``features`` to ``next_bar_direction(close)`` and drop NaN rows.

    Shared training-batch builder for directional-classifier strategies
    (``MomentumGatekeeperStrategy``, ``CrossAssetMomentumStrategy``). The
    target drops the final row (no ``t+1`` close); the returned features
    are sliced to the target's index and then masked to rows where every
    feature column is non-NaN. Callers that need a column subset should
    pre-subset ``features`` before calling.
    """

    target = next_bar_direction(close)
    features_aligned = features.loc[target.index]
    valid_mask = features_aligned.notna().all(axis=1)
    return features_aligned.loc[valid_mask], target.loc[valid_mask]


def directional_accuracy(
    prediction: pd.Series[float],
    close: pd.Series[float],
    *,
    threshold: float = 0.0,
) -> float:
    """
    Hit-rate of a directional forecast against the realised next-bar move.

    ``prediction`` is the model's continuous output at bar ``t`` (a forecast
    return, or an up-probability); a value above ``threshold`` is read as
    "up". It is aligned against ``next_bar_direction(close)`` and masked to
    rows where both are present BEFORE the threshold is applied - so warmup
    rows (NaN prediction) are dropped rather than silently coerced to
    "down", and the final bar (no ``t+1`` close, hence NaN realised) never
    contributes. The realised target is derived only from ``close``, so it
    is invariant to permuting any engineered feature. Returns ``nan`` when no
    bar survives alignment.

    ``threshold`` is ``0.0`` for signed-return forecasts and ``0.5`` for
    probability forecasts. ``prediction`` must be indexed within ``close``'s
    index (it may be a subset - warmup rows legitimately drop out); a stray
    label raises rather than letting the index-aligned join silently discard a
    mismatched forecast as NaN.
    """

    stray_labels = prediction.index.difference(close.index)
    if len(stray_labels) > 0:
        raise ValueError(
            f"directional_accuracy: prediction carries {len(stray_labels)} index "
            f"label(s) absent from close (e.g. {stray_labels[0]!r}); align the "
            f"forecast to the close series so no row is silently dropped."
        )
    realised = next_bar_direction(close)
    aligned = pd.DataFrame({"prediction": prediction, "realised": realised}).dropna()
    if aligned.empty:
        return float("nan")
    predicted_up = (aligned["prediction"] > threshold).astype(int)
    return float((predicted_up == aligned["realised"]).mean())


def negative_qlike(
    forecast_vol: pd.Series[float],
    realised_vol: pd.Series[float],
) -> float:
    """
    Negative mean QLIKE of a volatility forecast (higher is better).

    Both inputs are annualized volatilities (standard deviations); they are
    squared to variances and scored with the QLIKE loss
    ``log(sigma2_hat) + sigma2_realised / sigma2_hat`` (Patton, 2011), which
    is robust to the noisy realised proxy and consistent under an imperfect
    variance estimator. The sign is flipped so a smaller forecast error
    yields a larger score, matching the higher-is-better convention of the
    permutation-importance driver. Inputs are aligned on their shared index
    and masked to rows where both are present and the forecast is strictly
    positive (QLIKE is undefined at ``sigma2_hat = 0``). Both series are
    contemporaneous at bar ``t`` and use only bars ``<= t``. Returns ``nan``
    when no bar survives.
    """

    aligned = pd.DataFrame({"forecast": forecast_vol, "realised": realised_vol}).dropna()
    aligned = aligned[aligned["forecast"] > 0.0]
    if aligned.empty:
        return float("nan")
    var_hat = aligned["forecast"] ** 2
    var_realised = aligned["realised"] ** 2
    qlike = np.log(var_hat) + var_realised / var_hat
    return float(-qlike.mean())


def next_bar_log_return(close: pd.Series[float]) -> pd.Series[float]:
    """
    Realised next-bar log return aligned at bar ``t`` (return ``t`` -> ``t+1``); final row excluded.

    The regression counterpart to :func:`next_bar_direction`: a return forecast
    at bar ``t`` predicts this quantity. Derived only from ``close``, so it is
    invariant to permuting any engineered feature. The last bar has no ``t+1``
    close, so it is dropped (not filled) to avoid a leakage hazard.
    """

    realised: pd.Series[float] = compute_log_returns(close).shift(-1).iloc[:-1]
    return realised


def negative_return_mse(forecast: pd.Series[float], close: pd.Series[float]) -> float:
    """
    Negative MSE of a return forecast vs the realised next-bar return (higher is better).

    The continuous counterpart to :func:`directional_accuracy` for a mean
    forecast. A return hybrid sets the forecast SIGN with a feature-invariant
    term (ARMA from ``close``) and only nudges its MAGNITUDE through the
    feature-fed residual, so a discrete hit-rate is blind to the features (it
    moves only when a sign flips); squared error tracks the magnitude, so the
    residual's feature dependence stays visible. The sign is flipped to match
    the higher-is-better convention of the permutation driver. Aligned on the
    shared index; the realised target comes only from ``close``, so it is
    invariant to feature permutation. Returns ``nan`` when no bar survives.
    """

    stray_labels = forecast.index.difference(close.index)
    if len(stray_labels) > 0:
        raise ValueError(
            f"negative_return_mse: forecast carries {len(stray_labels)} index "
            f"label(s) absent from close (e.g. {stray_labels[0]!r}); align the "
            f"forecast to the close series so no row is silently dropped."
        )
    aligned = pd.DataFrame({"forecast": forecast, "realised": next_bar_log_return(close)}).dropna()
    if aligned.empty:
        return float("nan")
    error = aligned["forecast"] - aligned["realised"]
    return float(-(error**2).mean())


# Clip probabilities off {0, 1} before the log so a confident miss yields a
# large-but-finite loss instead of -inf.
_LOG_LOSS_CLIP_EPS = 1e-15


def negative_log_loss(prob_up: pd.Series[float], close: pd.Series[float]) -> float:
    """
    Negative binary log-loss of an up-probability vs the realised next-bar move (higher is better).

    The continuous counterpart to :func:`directional_accuracy` for a directional
    classifier. A thresholded hit-rate ignores any probability shift that does
    not cross the decision boundary, so it under-resolves features that move
    ``P(up)`` without flipping the call; log-loss tracks the full probability, so
    every feature's contribution registers. The sign is flipped to match the
    higher-is-better convention of the permutation driver. ``prob_up`` is aligned
    against :func:`next_bar_direction`; the realised target comes only from
    ``close``, so it is invariant to feature permutation. Returns ``nan`` when no
    bar survives.
    """

    stray_labels = prob_up.index.difference(close.index)
    if len(stray_labels) > 0:
        raise ValueError(
            f"negative_log_loss: prob_up carries {len(stray_labels)} index "
            f"label(s) absent from close (e.g. {stray_labels[0]!r}); align the "
            f"forecast to the close series so no row is silently dropped."
        )
    aligned = pd.DataFrame({"prob": prob_up, "realised": next_bar_direction(close)}).dropna()
    if aligned.empty:
        return float("nan")
    prob = aligned["prob"].clip(_LOG_LOSS_CLIP_EPS, 1.0 - _LOG_LOSS_CLIP_EPS)
    realised = aligned["realised"]
    loss = -(realised * np.log(prob) + (1.0 - realised) * np.log(1.0 - prob))
    return float(-loss.mean())
