"""
Pairwise-strategy significance tests.

Three families of test, each appropriate to a different workload:

* **Stationary bootstrap** (Politis & Romano, 1994) - autocorrelation-aware
  percentile bootstrap on daily return series. Used for:

  - :func:`bootstrap_sharpe_ci` - 95% CI on a single strategy's Sharpe ratio.
  - :func:`paired_bootstrap_sharpe_differential` - 95% CI on the Sharpe
    differential between two aligned strategies; a CI that excludes zero
    indicates a significant difference. This is the fallback for
    non-forecaster strategies (Bollinger, Pairs, Momentum) where the
    Diebold-Mariano framing does not apply.

* **Diebold-Mariano** (1995, with Harvey-Leybourne-Newbold 1997
  small-sample correction) - parametric test on aligned point forecasts
  plus realised values. Used by :func:`diebold_mariano_test` for
  forecaster strategies (ARMA / LSTM / hybrids) that expose predicted
  returns or variances.

* **Deflated Sharpe ratio** (Bailey & López de Prado, 2014) -
  multiple-testing-adjusted significance for an HPO study's best Sharpe.
  Used by :func:`deflated_sharpe_ratio` after a tuning run completes;
  inputs are the per-trial Sharpes from the Optuna study (no per-trial
  return series required).

Block-size default
------------------
Both bootstrap functions default ``block_size`` to
``max(1, round(2 * sqrt(n)))`` - a widely used heuristic that gives a
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import scipy.stats as scipy_stats

from quant_engine import MetricsCalculator
from src.core import json_io

_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.int64]

# Fixed seed for bootstrap determinism when no RNG is passed. Different
# from the aggregator's seed so a joint invocation (aggregate -> bootstrap)
# does not share a draw sequence by accident.
_DEFAULT_RNG_SEED = 17

_DEFAULT_N_RESAMPLES = 10_000


class DMLoss(StrEnum):
    """
    Per-observation loss for the Diebold-Mariano test.
    """

    MSE = "mse"
    MAE = "mae"


class DMDirection(StrEnum):
    """
    Sign of the Diebold-Mariano mean loss differential.

    ``A`` / ``B`` indicate which of the two forecasts has the smaller
    expected loss; ``TIE`` is the exact-zero degenerate case.
    """

    A = "a"
    B = "b"
    TIE = "tie"


@dataclass(frozen=True)
class BootstrapCI:
    """
    Percentile-bootstrap confidence interval.

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
        """
        ``True`` if ``value`` falls outside ``[lower, upper]`` - common
        significance check for a differential CI against zero.
        """

        return value < self.lower or value > self.upper

    def to_dict(self) -> dict[str, object]:
        return {
            "point_estimate": self.point_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "block_size": self.block_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> BootstrapCI:
        return cls(
            point_estimate=json_io.get_float(d, "point_estimate"),
            lower=json_io.get_float(d, "lower"),
            upper=json_io.get_float(d, "upper"),
            confidence=json_io.get_float(d, "confidence"),
            n_resamples=json_io.get_int(d, "n_resamples"),
            block_size=json_io.get_int(d, "block_size"),
        )


_EULER_MASCHERONI = 0.5772156649015329
_TRIAL_VARIANCE_EPS = 1e-12


@dataclass(frozen=True)
class DeflatedSharpe:
    """
    Bailey-López de Prado (2014) deflated Sharpe ratio.

    Quantifies how much of an HPO study's best Sharpe is plausibly
    chance, given that the search tried many configurations. The
    deflated value is a probability in ``[0, 1]`` - values close to
    ``1`` indicate the best Sharpe is unlikely to be a multiple-testing
    artefact; values near ``0.5`` indicate it is consistent with noise.

    Implementation follows the practical post-hoc form used by López de
    Prado's MlFinLab and similar libraries: the moments of the trial
    Sharpe distribution stand in for the unobserved selected-strategy
    return moments in BLP eq.(9). This is the only form that works
    when only the Optuna ``study.db`` (not per-trial return series) is
    available downstream.
    """

    observed_sharpe: float
    expected_max_sharpe: float
    deflated_sharpe: float
    p_value: float
    n_trials: int
    sample_length: int
    trial_sharpe_variance: float
    trial_sharpe_skew: float
    trial_sharpe_kurtosis: float

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_sharpe": self.observed_sharpe,
            "expected_max_sharpe": self.expected_max_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "p_value": self.p_value,
            "n_trials": self.n_trials,
            "sample_length": self.sample_length,
            "trial_sharpe_variance": self.trial_sharpe_variance,
            "trial_sharpe_skew": self.trial_sharpe_skew,
            "trial_sharpe_kurtosis": self.trial_sharpe_kurtosis,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> DeflatedSharpe:
        return cls(
            observed_sharpe=json_io.get_float(d, "observed_sharpe"),
            expected_max_sharpe=json_io.get_float(d, "expected_max_sharpe"),
            deflated_sharpe=json_io.get_float(d, "deflated_sharpe"),
            p_value=json_io.get_float(d, "p_value"),
            n_trials=json_io.get_int(d, "n_trials"),
            sample_length=json_io.get_int(d, "sample_length"),
            trial_sharpe_variance=json_io.get_float(d, "trial_sharpe_variance"),
            trial_sharpe_skew=json_io.get_float(d, "trial_sharpe_skew"),
            trial_sharpe_kurtosis=json_io.get_float(d, "trial_sharpe_kurtosis"),
        )


@dataclass(frozen=True)
class PooledSharpe:
    """
    Pooled out-of-sample Sharpe + Probabilistic Sharpe Ratio for one leg.

    Computed from the concatenation of every walk-forward fold's
    within-fold OOS returns (the seam return between two folds is not
    tradeable and is dropped by the producer). Unlike the mean-of-folds
    Sharpe, this is the realised end-to-end track record - observation
    weighted, so long folds count more than short ones.

    ``psr`` is the Bailey-Lopez de Prado Probabilistic Sharpe Ratio
    against a zero benchmark: the probability the true Sharpe exceeds 0
    given the return distribution's ``skew`` / ``kurtosis`` and sample
    length ``n_obs``. The cross-leg *deflated* Sharpe (selection over many
    legs) is produced separately by :func:`deflate_pooled_across_legs`,
    which needs every leg at once. ``kurtosis`` is non-excess (Normal = 3).

    A leg with fewer than two pooled OOS returns yields all-NaN fields;
    downstream code discriminates on ``math.isnan(sharpe)``.
    """

    sharpe: float
    psr: float
    n_obs: int
    skew: float
    kurtosis: float


@dataclass(frozen=True)
class DMResult:
    """
    Diebold-Mariano test outcome.

    ``direction`` reports the sign of the mean loss differential
    (``d = L(e_a) - L(e_b)``): :attr:`DMDirection.B` if ``mean(d) > 0``
    (b's loss is smaller, so b forecasts better), :attr:`DMDirection.A`
    if ``mean(d) < 0``, :attr:`DMDirection.TIE` on an exact zero.
    Significance is up to the caller - compare ``p_value`` against the
    desired alpha.
    """

    statistic: float
    p_value: float
    direction: DMDirection
    h: int
    loss: DMLoss


def bootstrap_sharpe_ci(
    returns: _FloatArray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    block_size: int | None = None,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """
    Stationary-bootstrap 95% CI for the Sharpe ratio of ``returns``.

    ``returns`` is a 1-D array of periodic (per-bar) returns; annualisation
    is deliberately NOT applied here - feed pre-annualised returns if
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
    sharpes = _run_block_bootstrap(
        lambda idx: _sharpe(arr[idx]),
        len(arr),
        n_resamples=n_resamples,
        block=block,
        rng=rng_actual,
    )

    lo, hi = percentile_ci(sharpes, confidence)
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
    """
    Stationary-bootstrap 95% CI on ``sharpe(a) - sharpe(b)``.

    The two return series must be aligned in time - same bars, same
    length - so a single set of bootstrap indices can be applied to both,
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
    diffs = _run_block_bootstrap(
        lambda idx: _sharpe(a[idx]) - _sharpe(b[idx]),
        len(a),
        n_resamples=n_resamples,
        block=block,
        rng=rng_actual,
    )

    lo, hi = percentile_ci(diffs, confidence)
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
    loss: DMLoss = DMLoss.MSE,
) -> DMResult:
    """
    Harvey-Leybourne-Newbold-corrected Diebold-Mariano (1995) test.

    Null hypothesis: the two forecasts have equal expected loss.
    Alternative (two-sided): unequal expected loss. ``h`` is the forecast
    horizon in bars and controls the Bartlett-kernel bandwidth for the
    long-run variance estimator (bandwidth ``h-1``); ``h=1`` collapses to
    the lag-0 variance, appropriate for one-step-ahead forecasts. ``loss``
    selects the per-observation loss (``mse`` = squared error, ``mae`` =
    absolute error).

    The HLN small-sample correction scales the raw DM statistic by
    ``sqrt((n + 1 - 2h + h(h-1)/n) / n)`` and reads the p-value off a
    Student-t with ``n-1`` dof - both standard for empirical forecasting
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
        raise ValueError(
            f"h must be >= 1 (forecast horizon in bars), got {h}; fix by "
            f"passing h=1 for one-step-ahead forecasts."
        )
    n = len(a)
    if n < 2:
        raise ValueError(
            f"DM test needs n >= 2 aligned observations, got {n}; fix by "
            f"passing a longer aligned forecast/actual series."
        )

    errors_a = y - a
    errors_b = y - b
    loss_a = errors_a * errors_a if loss is DMLoss.MSE else np.abs(errors_a)
    loss_b = errors_b * errors_b if loss is DMLoss.MSE else np.abs(errors_b)
    d = loss_a - loss_b
    mean_d = float(np.mean(d))

    long_run_var = _newey_west_long_run_var(d, mean_d, lag=h - 1)
    if long_run_var <= 0.0:
        # Identical-loss sequences: DM stat undefined (0/0). Report tie.
        return DMResult(statistic=0.0, p_value=1.0, direction=DMDirection.TIE, h=h, loss=loss)

    dm_stat = mean_d / math.sqrt(long_run_var / n)
    hln_factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_factor
    p_value = 2.0 * (1.0 - float(scipy_stats.t.cdf(abs(dm_stat_corrected), df=n - 1)))

    if mean_d > 0:
        direction = DMDirection.B
    elif mean_d < 0:
        direction = DMDirection.A
    else:
        direction = DMDirection.TIE

    return DMResult(
        statistic=float(dm_stat_corrected),
        p_value=float(p_value),
        direction=direction,
        h=h,
        loss=loss,
    )


def deflated_sharpe_ratio(
    trial_sharpes: Sequence[float],
    *,
    sample_length: int,
) -> DeflatedSharpe:
    """
    Compute the Bailey-López de Prado (2014) deflated Sharpe ratio.

    The deflated Sharpe asks: given that an HPO search tried N
    candidate configurations and selected the best one, how much of
    the observed peak Sharpe is plausibly genuine vs. selection bias?
    Returns a probability - values near ``1`` are evidence of a real
    signal, values near ``0.5`` are indistinguishable from noise.

    Inputs:
        trial_sharpes: per-trial Sharpe ratios from the completed
            Optuna study. The maximum across trials is the observed
            (selected) Sharpe. Length must be >= 2.
        sample_length: number of return observations the selected
            strategy was evaluated on (typically ``n_dev_bars``).
            Must be >= 2.

    Formula (BLP eq. 9, post-hoc trial-based form):

    ``E[max{Sh_n}] ~= sqrt(V[Sh]) * ((1-gamma)*Z^-1(1-1/N) + gamma*Z^-1(1-1/(N*e)))``
    where gamma is Euler-Mascheroni and ``Z^-1`` is the standard-normal quantile.

    ``psi(Sh*) = (Sh* - E[max{Sh_n}]) * sqrt(T-1) /
                sqrt(1 - gamma_hat_3*Sh* + ((gamma_hat_4 - 1)/4)*Sh*^2)``

    ``DSR = Phi(psi)`` - standard-normal CDF of the test statistic.

    Degenerate cases:
        - Single trial (``N=1``): ``E[max] = 0`` (no selection); DSR
          collapses to the plain Sharpe's one-sided p-value.
        - Zero trial variance: ``E[max] = 0``; DSR computes as if
          there were no multiple-testing penalty.
        - Denominator <= 0 (extreme negative skew x large Sharpe):
          DSR = 0.5 (treat as undefined / non-significant).
    """

    if sample_length < 2:
        raise ValueError(
            f"sample_length must be >= 2, got {sample_length}; fix by passing "
            f"the dev-region bar count."
        )
    arr = _as_1d_float(np.asarray(trial_sharpes, dtype=np.float64), "trial_sharpes")
    n_trials = len(arr)
    if n_trials < 1:
        raise ValueError(
            "trial_sharpes must be non-empty; got 0 trials. Fix by passing the "
            "completed-trial Sharpes from the Optuna study."
        )

    observed = float(np.max(arr))
    if n_trials >= 2:
        trial_var = float(np.var(arr, ddof=1))
        # Catastrophic-cancellation guard: scipy.stats.skew/kurtosis warn
        # and return nonsensical moments when the input is numerically
        # constant. Treat near-zero variance as "all trials identical"
        # and fall back to Normal-baseline moments (skew=0, kurtosis=3),
        # which collapses the BLP eq.(9) penalty cleanly.
        if trial_var > _TRIAL_VARIANCE_EPS:
            trial_skew = float(scipy_stats.skew(arr, bias=False))
            trial_kurt = float(scipy_stats.kurtosis(arr, fisher=False, bias=False))
        else:
            trial_var = 0.0
            trial_skew = 0.0
            trial_kurt = 3.0
    else:
        trial_var = 0.0
        trial_skew = 0.0
        trial_kurt = 3.0

    expected_max = _expected_max_sharpe(n_trials=n_trials, trial_variance=trial_var)

    deflated = _psr_probability(
        observed_sharpe=observed,
        benchmark_sharpe=expected_max,
        skew=trial_skew,
        kurtosis=trial_kurt,
        sample_length=sample_length,
    )
    p_value = 1.0 - deflated

    return DeflatedSharpe(
        observed_sharpe=observed,
        expected_max_sharpe=expected_max,
        deflated_sharpe=deflated,
        p_value=p_value,
        n_trials=n_trials,
        sample_length=sample_length,
        trial_sharpe_variance=trial_var,
        trial_sharpe_skew=trial_skew,
        trial_sharpe_kurtosis=trial_kurt,
    )


def _expected_max_sharpe(*, n_trials: int, trial_variance: float) -> float:
    """
    Expected maximum of ``n_trials`` iid normal Sharpes (BLP 2014 appendix).

    ``E[max] ~= sqrt(V) * ((1-gamma)*Z^-1(1 - 1/N) + gamma*Z^-1(1 - 1/(N*e)))``
    where ``gamma`` is Euler-Mascheroni and ``Z^-1`` is the standard-normal
    quantile function. Collapses to ``0`` for ``N=1`` (no selection) or
    ``trial_variance == 0`` (flat trial distribution).
    """

    if n_trials < 2 or trial_variance <= 0.0:
        return 0.0
    std = math.sqrt(trial_variance)
    q1 = float(scipy_stats.norm.ppf(1.0 - 1.0 / n_trials))
    q2 = float(scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return std * ((1.0 - _EULER_MASCHERONI) * q1 + _EULER_MASCHERONI * q2)


def _psr_probability(
    *,
    observed_sharpe: float,
    benchmark_sharpe: float,
    skew: float,
    kurtosis: float,
    sample_length: int,
) -> float:
    """
    Bailey-Lopez de Prado eq.(9) test statistic mapped to a probability.

    Shared core of both the trial-based deflated Sharpe and the pooled-OOS
    PSR. Returns ``Phi(psi)`` where

    ``psi = (observed - benchmark) * sqrt(T - 1) /
            sqrt(1 - skew*observed + ((kurtosis - 1)/4)*observed^2)``

    The denominator is the Sharpe-estimator standard error corrected for
    non-normal returns (``kurtosis`` non-excess, Normal = 3). A non-positive
    radicand (extreme negative skew x large Sharpe) leaves the statistic
    undefined; we return ``0.5`` (non-significant) rather than raise.
    ``benchmark_sharpe`` is ``0`` for a plain PSR - P[true Sharpe > 0] - or
    the expected-max over N candidates for the selection-deflated form.
    """

    denom_sq = (
        1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe
    )
    if denom_sq <= 0.0:
        return 0.5
    psi = (observed_sharpe - benchmark_sharpe) * math.sqrt(sample_length - 1) / math.sqrt(denom_sq)
    return float(scipy_stats.norm.cdf(psi))


def compute_pooled_sharpe(
    returns: _FloatArray, *, annualization_factor: int, risk_free_rate: float = 0.0
) -> PooledSharpe:
    """
    Pooled OOS Sharpe + zero-benchmark PSR for one leg's stitched returns.

    ``returns`` is the concatenation of each fold's within-fold OOS returns
    (the caller drops the non-tradeable seam between folds). The Sharpe is
    annualised through the same C++ ``MetricsCalculator.sharpe_ratio`` the
    per-fold metrics use, and ``risk_free_rate`` is the same rate those
    per-fold metrics subtract, so pooled and per-fold Sharpes are on one
    scale. The PSR uses the per-bar return skew / kurtosis and
    ``T = len(returns)``.

    Fewer than two returns -> all-NaN (Sharpe undefined). A numerically
    constant series falls back to Normal-baseline moments (skew 0,
    kurtosis 3) to dodge scipy's constant-input warnings, matching
    :func:`deflated_sharpe_ratio`.
    """

    arr = _as_1d_float(returns, "returns")
    n_obs = len(arr)
    if n_obs < 2:
        nan = float("nan")
        return PooledSharpe(sharpe=nan, psr=nan, n_obs=n_obs, skew=nan, kurtosis=nan)

    sharpe = float(
        MetricsCalculator.sharpe_ratio(
            np.ascontiguousarray(arr), annualization_factor, risk_free_rate
        )
    )
    if float(np.var(arr, ddof=1)) > _TRIAL_VARIANCE_EPS:
        skew = float(scipy_stats.skew(arr, bias=False))
        kurtosis = float(scipy_stats.kurtosis(arr, fisher=False, bias=False))
    else:
        skew = 0.0
        kurtosis = 3.0

    psr = _psr_probability(
        observed_sharpe=sharpe,
        benchmark_sharpe=0.0,
        skew=skew,
        kurtosis=kurtosis,
        sample_length=n_obs,
    )
    return PooledSharpe(sharpe=sharpe, psr=psr, n_obs=n_obs, skew=skew, kurtosis=kurtosis)


def deflate_pooled_across_legs(legs: Sequence[PooledSharpe]) -> tuple[float, ...]:
    """
    Selection-deflated pooled Sharpe probability per leg, aligned to input order.

    Guards the "we kept the best of many strategy x universe pairs"
    selection bias: each leg's PSR benchmark is the expected maximum Sharpe
    across all legs (BLP expected-max of ``N`` iid-normal Sharpes with the
    legs' observed Sharpe variance), not zero. A leg whose pooled Sharpe
    clears that selection bar scores high; one below it scores low.

    Legs with a NaN pooled Sharpe (too few OOS bars) are excluded from the
    variance + expected-max estimate and map to NaN in the output, so the
    return tuple stays aligned with ``legs``.
    """

    finite = [leg for leg in legs if not math.isnan(leg.sharpe)]
    if not finite:
        return tuple(float("nan") for _ in legs)

    n_legs = len(finite)
    sharpes = np.array([leg.sharpe for leg in finite], dtype=np.float64)
    variance = float(np.var(sharpes, ddof=1)) if n_legs >= 2 else 0.0
    expected_max = _expected_max_sharpe(n_trials=n_legs, trial_variance=variance)

    return tuple(
        float("nan")
        if math.isnan(leg.sharpe)
        else _psr_probability(
            observed_sharpe=leg.sharpe,
            benchmark_sharpe=expected_max,
            skew=leg.skew,
            kurtosis=leg.kurtosis,
            sample_length=leg.n_obs,
        )
        for leg in legs
    )


def _run_block_bootstrap(
    statistic_at_idx: Callable[[_IntArray], float],
    n: int,
    *,
    n_resamples: int,
    block: int,
    rng: np.random.Generator,
) -> _FloatArray:
    """
    Drive ``n_resamples`` stationary-bootstrap draws of a scalar statistic.

    Single source of truth for both the single-series and paired Sharpe
    bootstraps: each iteration draws a length-``n`` Politis-Romano index
    vector and feeds it to ``statistic_at_idx`` (which closes over the
    underlying return arrays). Returning a 1-D float array lets callers
    pipe straight into :func:`percentile_ci`.
    """

    out = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        idx = _stationary_bootstrap_indices(n, block, rng)
        out[r] = statistic_at_idx(idx)
    return out


def _stationary_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> _IntArray:
    """
    Draw a length-``n`` index vector per Politis-Romano (1994).

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
    """
    Unannualised Sharpe: ``mean / std_ddof1``, zero when std vanishes.

    Zero-std short-circuits to ``0.0`` instead of raising or returning
    ``inf`` - a flat-return resample produces an ill-defined Sharpe and
    ``0.0`` is the only value that lets the bootstrap distribution
    remain finite. The aggregate CI still correctly widens when many
    resamples hit this branch.
    """

    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(returns) / std)


def _newey_west_long_run_var(d: _FloatArray, mean_d: float, *, lag: int) -> float:
    """
    Bartlett-kernel HAC long-run variance estimator.

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


def percentile_ci(samples: _FloatArray, confidence: float) -> tuple[float, float]:
    """
    Symmetric percentile CI from a bootstrap sample distribution.
    """

    alpha = 1.0 - confidence
    lo = float(np.percentile(samples, 100 * alpha / 2.0))
    hi = float(np.percentile(samples, 100 * (1.0 - alpha / 2.0)))
    return lo, hi


def _resolve_block_size(block_size: int | None, n: int) -> int:
    """
    Heuristic default: ``max(1, round(2 * sqrt(n)))``.
    """

    if block_size is not None:
        if block_size < 1:
            raise ValueError(
                f"block_size must be >= 1, got {block_size}; fix by passing a "
                f"positive integer or None for the 2*sqrt(n) default."
            )
        return block_size
    return max(1, int(round(2.0 * math.sqrt(n))))


def _as_1d_float(arr: _FloatArray, name: str) -> _FloatArray:
    """
    Coerce to contiguous 1-D float64, raising on wrong shape.
    """

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
