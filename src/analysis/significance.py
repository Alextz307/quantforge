"""Pairwise-strategy significance tests.

Two families of test, each appropriate to a different workload:

* **Stationary bootstrap** (Politis & Romano, 1994) — autocorrelation-aware
  percentile bootstrap on daily return series. Used for:

  - :func:`bootstrap_sharpe_ci` — 95% CI on a single strategy's Sharpe ratio.
  - :func:`paired_bootstrap_sharpe_differential` — 95% CI on the Sharpe
    differential between two aligned strategies; a CI that excludes zero
    indicates a significant difference. This is the fallback for
    non-forecaster strategies (Bollinger, Pairs, Momentum) where the
    Diebold-Mariano framing does not apply.

* **Diebold-Mariano** (1995, with Harvey-Leybourne-Newbold 1997
  small-sample correction) — parametric test on aligned point forecasts
  plus realised values. Used by :func:`diebold_mariano_test` for
  forecaster strategies (ARMA / LSTM / hybrids) that expose predicted
  returns or variances.

Block-size default
------------------
Both bootstrap functions default ``block_size`` to
``max(1, round(2 * sqrt(n)))`` — a widely used heuristic that gives a
sensible effective block length for autocorrelated series in the
hundreds-to-thousands-of-bars range. Users who know their data's
autocorrelation structure can pass an explicit block size; the
Politis-White (2004) optimal formula is not implemented here because
scipy does not expose it and the heuristic is within the 10% band of
optimal for typical daily-return series.

Determinism
-----------
Every bootstrap takes ``rng: np.random.Generator | None``. ``None``
defaults to a fixed-seed generator so report output is reproducible
across invocations. Callers (e.g. the comparison orchestrator) can pass
a seeded generator to vary the bootstrap across runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.stats as scipy_stats

_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.int64]

# Fixed seed for bootstrap determinism when no RNG is passed. Different
# from the aggregator's seed so a joint invocation (aggregate → bootstrap)
# does not share a draw sequence by accident.
_DEFAULT_RNG_SEED = 17

# Number of resamples for the bootstrap-based tests. 10k is the
# widely-accepted default for 2-decimal stability of 2.5 / 97.5
# percentiles; trivial on any modern CPU for n <= a few thousand bars.
_DEFAULT_N_RESAMPLES = 10_000


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile-bootstrap confidence interval.

    ``point_estimate`` is computed on the full (non-resampled) sample.
    ``lower`` / ``upper`` are the ``(1-confidence)/2`` and
    ``1-(1-confidence)/2`` percentiles of the bootstrap distribution.
    """

    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    block_size: int

    def excludes(self, value: float) -> bool:
        """``True`` if ``value`` falls outside ``[lower, upper]`` — common
        significance check for a differential CI against zero.
        """
        return value < self.lower or value > self.upper


@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano test outcome.

    ``direction`` reports the sign of the mean loss differential
    (``d = L(e_a) - L(e_b)``): ``"b"`` if ``mean(d) > 0`` (b's loss is
    smaller, so b forecasts better), ``"a"`` if ``mean(d) < 0``,
    ``"tie"`` on an exact zero. Significance is up to the caller —
    compare ``p_value`` against the desired alpha.
    """

    statistic: float
    p_value: float
    direction: Literal["a", "b", "tie"]
    h: int
    loss: Literal["mse", "mae"]


def bootstrap_sharpe_ci(
    returns: _FloatArray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    block_size: int | None = None,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Stationary-bootstrap 95% CI for the Sharpe ratio of ``returns``.

    ``returns`` is a 1-D array of periodic (per-bar) returns; annualisation
    is deliberately NOT applied here — feed pre-annualised returns if
    the CI should be in annualised Sharpe units. The bootstrap resamples
    the raw series, so whatever scale goes in, the CI is in the same
    scale out.
    """
    arr = _as_1d_float(returns, "returns")
    if len(arr) < 2:
        raise ValueError(
            f"bootstrap_sharpe_ci needs at least 2 returns to estimate Sharpe, "
            f"got {len(arr)}; fix by passing a longer return series."
        )
    _validate_confidence(confidence)
    rng_actual = rng if rng is not None else np.random.default_rng(_DEFAULT_RNG_SEED)
    block = _resolve_block_size(block_size, len(arr))

    point = _sharpe(arr)
    sharpes = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        idx = _stationary_bootstrap_indices(len(arr), block, rng_actual)
        sharpes[r] = _sharpe(arr[idx])

    lo, hi = _percentile_ci(sharpes, confidence)
    return BootstrapCI(
        point_estimate=point,
        lower=lo,
        upper=hi,
        confidence=confidence,
        n_resamples=n_resamples,
        block_size=block,
    )


def paired_bootstrap_sharpe_differential(
    returns_a: _FloatArray,
    returns_b: _FloatArray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    block_size: int | None = None,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Stationary-bootstrap 95% CI on ``sharpe(a) - sharpe(b)``.

    The two return series must be aligned in time — same bars, same
    length — so a single set of bootstrap indices can be applied to both,
    preserving the per-bar pairing that makes this a paired (not
    independent) test. The CI is on the Sharpe DIFFERENTIAL; it excludes
    zero when the two strategies' Sharpe ratios differ significantly.
    """
    a = _as_1d_float(returns_a, "returns_a")
    b = _as_1d_float(returns_b, "returns_b")
    if a.shape != b.shape:
        raise ValueError(
            f"paired_bootstrap requires aligned return series, got shapes "
            f"{a.shape} vs {b.shape}; fix by passing same-length aligned arrays."
        )
    if len(a) < 2:
        raise ValueError(
            f"paired_bootstrap needs at least 2 aligned returns per series, got {len(a)}."
        )
    _validate_confidence(confidence)
    rng_actual = rng if rng is not None else np.random.default_rng(_DEFAULT_RNG_SEED)
    block = _resolve_block_size(block_size, len(a))

    point = _sharpe(a) - _sharpe(b)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        idx = _stationary_bootstrap_indices(len(a), block, rng_actual)
        diffs[r] = _sharpe(a[idx]) - _sharpe(b[idx])

    lo, hi = _percentile_ci(diffs, confidence)
    return BootstrapCI(
        point_estimate=point,
        lower=lo,
        upper=hi,
        confidence=confidence,
        n_resamples=n_resamples,
        block_size=block,
    )


def diebold_mariano_test(
    forecast_a: _FloatArray,
    forecast_b: _FloatArray,
    actual: _FloatArray,
    *,
    h: int = 1,
    loss: Literal["mse", "mae"] = "mse",
) -> DMResult:
    """Harvey-Leybourne-Newbold-corrected Diebold-Mariano (1995) test.

    Null hypothesis: the two forecasts have equal expected loss.
    Alternative (two-sided): unequal expected loss. ``h`` is the forecast
    horizon in bars and controls the Bartlett-kernel bandwidth for the
    long-run variance estimator (bandwidth ``h-1``); ``h=1`` collapses to
    the lag-0 variance, appropriate for one-step-ahead forecasts. ``loss``
    selects the per-observation loss (``mse`` = squared error, ``mae`` =
    absolute error).

    The HLN small-sample correction scales the raw DM statistic by
    ``sqrt((n + 1 - 2h + h(h-1)/n) / n)`` and reads the p-value off a
    Student-t with ``n-1`` dof — both standard for empirical forecasting
    work where ``n`` is in the hundreds. Degenerate case: identical
    forecasts (``L(e_a) == L(e_b)`` everywhere) produce zero long-run
    variance; we return ``statistic=0, p_value=1.0, direction="tie"``
    rather than a divide-by-zero.
    """
    a = _as_1d_float(forecast_a, "forecast_a")
    b = _as_1d_float(forecast_b, "forecast_b")
    y = _as_1d_float(actual, "actual")
    if not (a.shape == b.shape == y.shape):
        raise ValueError(
            f"diebold_mariano_test requires aligned forecasts + actuals; got "
            f"forecast_a={a.shape}, forecast_b={b.shape}, actual={y.shape}; fix "
            f"by aligning all three on the same test index."
        )
    if h < 1:
        raise ValueError(f"h must be >= 1 (forecast horizon in bars), got {h}.")
    if loss not in {"mse", "mae"}:
        raise ValueError(f"loss must be 'mse' or 'mae', got {loss!r}.")
    n = len(a)
    if n < 2:
        raise ValueError(f"DM test needs n >= 2 aligned observations, got {n}.")

    errors_a = y - a
    errors_b = y - b
    loss_a = errors_a * errors_a if loss == "mse" else np.abs(errors_a)
    loss_b = errors_b * errors_b if loss == "mse" else np.abs(errors_b)
    d = loss_a - loss_b
    mean_d = float(np.mean(d))

    long_run_var = _newey_west_long_run_var(d, mean_d, lag=h - 1)
    if long_run_var <= 0.0:
        # Identical-loss sequences: DM stat undefined (0/0). Report tie.
        return DMResult(statistic=0.0, p_value=1.0, direction="tie", h=h, loss=loss)

    dm_stat = mean_d / math.sqrt(long_run_var / n)
    hln_factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_factor
    p_value = 2.0 * (1.0 - float(scipy_stats.t.cdf(abs(dm_stat_corrected), df=n - 1)))

    if mean_d > 0:
        direction: Literal["a", "b", "tie"] = "b"
    elif mean_d < 0:
        direction = "a"
    else:
        direction = "tie"

    return DMResult(
        statistic=float(dm_stat_corrected),
        p_value=float(p_value),
        direction=direction,
        h=h,
        loss=loss,
    )


def _stationary_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> _IntArray:
    """Draw a length-``n`` index vector per Politis-Romano (1994).

    Block lengths are geometrically distributed with mean ``block_size``;
    block starts are uniform on ``[0, n)``; indices wrap modulo ``n`` so
    a block that runs past the end continues from the start. This is the
    stationarity condition that distinguishes the Politis-Romano bootstrap
    from the fixed-block variant (which breaks stationarity at the seams).
    """
    p = 1.0 / block_size
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        take = min(length, n - i)
        # Wrap around via two slices when the block crosses the boundary.
        end = start + take
        if end <= n:
            out[i : i + take] = np.arange(start, end)
        else:
            first_chunk = n - start
            out[i : i + first_chunk] = np.arange(start, n)
            out[i + first_chunk : i + take] = np.arange(0, take - first_chunk)
        i += take
    return out


def _sharpe(returns: _FloatArray) -> float:
    """Unannualised Sharpe: ``mean / std_ddof1``, zero when std vanishes.

    Zero-std short-circuits to ``0.0`` instead of raising or returning
    ``inf`` — a flat-return resample produces an ill-defined Sharpe and
    ``0.0`` is the only value that lets the bootstrap distribution
    remain finite. The aggregate CI still correctly widens when many
    resamples hit this branch.
    """
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(returns) / std)


def _newey_west_long_run_var(d: _FloatArray, mean_d: float, *, lag: int) -> float:
    """Bartlett-kernel HAC long-run variance estimator.

    ``lag == 0`` collapses to the plain sample variance (what DM with
    ``h=1`` wants); ``lag > 0`` sums weighted autocovariances up to ``lag``
    with Bartlett weights ``1 - k/(lag+1)``.
    """
    centered = d - mean_d
    gamma_0 = float(np.mean(centered * centered))
    long_run = gamma_0
    for k in range(1, lag + 1):
        gamma_k = float(np.mean(centered[k:] * centered[:-k]))
        weight = 1.0 - k / (lag + 1)
        long_run += 2.0 * weight * gamma_k
    return long_run


def _percentile_ci(samples: _FloatArray, confidence: float) -> tuple[float, float]:
    """Symmetric percentile CI from a bootstrap sample distribution."""
    alpha = 1.0 - confidence
    lo = float(np.percentile(samples, 100 * alpha / 2.0))
    hi = float(np.percentile(samples, 100 * (1.0 - alpha / 2.0)))
    return lo, hi


def _resolve_block_size(block_size: int | None, n: int) -> int:
    """Heuristic default: ``max(1, round(2 * sqrt(n)))``."""
    if block_size is not None:
        if block_size < 1:
            raise ValueError(
                f"block_size must be >= 1, got {block_size}; fix by passing a "
                f"positive integer or None for the 2*sqrt(n) default."
            )
        return block_size
    return max(1, int(round(2.0 * math.sqrt(n))))


def _as_1d_float(arr: _FloatArray, name: str) -> _FloatArray:
    """Coerce to contiguous 1-D float64, raising on wrong shape."""
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got ndim={out.ndim} shape={out.shape}; fix by "
            f"flattening before passing."
        )
    return out


def _validate_confidence(confidence: float) -> None:
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}; typical values are 0.9, 0.95, 0.99."
        )
