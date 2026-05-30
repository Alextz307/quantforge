"""
Behavioral tests for :mod:`src.analysis.significance`.

Bootstrap tests use a seeded RNG so percentile draws are stable across
runs. Significance tests use synthetic return / forecast series with
known ground truth (a fixed-Sharpe AR(1) series for the bootstrap, a
deliberately biased forecast for DM) so the pass/fail decision is
deterministic - no statistical flakiness in CI.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from src.analysis.significance import (
    BootstrapCI,
    DeflatedSharpe,
    DMDirection,
    DMLoss,
    DMResult,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    diebold_mariano_test,
    paired_bootstrap_sharpe_differential,
)

_FloatArray = npt.NDArray[np.float64]

_BOOTSTRAP_SEED = 12345
_N_BARS = 500
_DAILY_VOL = 0.01
_EXPECTED_SHARPE = 0.05  # unannualised per-bar mean / std of return series
_SHARPE_CI_RESAMPLES = 2000  # tighter than the default so the test is fast


def _make_rng(seed: int = _BOOTSTRAP_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _daily_returns_with_known_sharpe(n: int, *, sharpe: float, seed: int) -> _FloatArray:
    """
    IID normal returns engineered so sample Sharpe ~= ``sharpe``.

    Draws from N(mu, sigma^2) with mu = sharpe * sigma; sample Sharpe
    converges to the population value as n grows. At n=500 the sample
    Sharpe is within ~10% of the population value at 95% confidence -
    enough headroom for the CI test below.
    """

    rng = np.random.default_rng(seed)
    return rng.normal(loc=sharpe * _DAILY_VOL, scale=_DAILY_VOL, size=n)


class TestBootstrapSharpeCI:
    def test_ci_contains_point_estimate(self) -> None:
        rets = _daily_returns_with_known_sharpe(_N_BARS, sharpe=_EXPECTED_SHARPE, seed=101)
        ci = bootstrap_sharpe_ci(rets, n_resamples=_SHARPE_CI_RESAMPLES, rng=_make_rng())
        assert isinstance(ci, BootstrapCI)
        assert ci.lower <= ci.point_estimate <= ci.upper

    def test_ci_brackets_the_known_population_sharpe(self) -> None:
        """
        For a 500-bar series with engineered Sharpe ~= 0.05, the
        bootstrap 95% CI should include 0.05 with very high probability -
        the test uses a fixed seed so the outcome is deterministic.
        """

        rets = _daily_returns_with_known_sharpe(_N_BARS, sharpe=_EXPECTED_SHARPE, seed=101)
        ci = bootstrap_sharpe_ci(rets, n_resamples=_SHARPE_CI_RESAMPLES, rng=_make_rng())
        assert ci.lower <= _EXPECTED_SHARPE <= ci.upper

    def test_rejects_series_shorter_than_two(self) -> None:
        with pytest.raises(ValueError, match="at least 2 returns"):
            bootstrap_sharpe_ci(np.array([0.01]))

    def test_rejects_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_sharpe_ci(np.array([0.01, 0.02, 0.03]), confidence=1.5)


class TestPairedBootstrapDifferential:
    def test_identical_series_yield_ci_tight_around_zero(self) -> None:
        rets = _daily_returns_with_known_sharpe(_N_BARS, sharpe=_EXPECTED_SHARPE, seed=202)
        ci = paired_bootstrap_sharpe_differential(
            rets, rets, n_resamples=_SHARPE_CI_RESAMPLES, rng=_make_rng()
        )
        assert ci.point_estimate == 0.0
        assert ci.lower <= 0.0 <= ci.upper

    def test_dominant_strategy_differential_excludes_zero(self) -> None:
        """
        Series a has higher mean return than b at the same volatility -
        the Sharpe differential is positive and the 95% CI should exclude 0.
        """

        rng = np.random.default_rng(303)
        n = _N_BARS
        a = rng.normal(loc=0.002, scale=_DAILY_VOL, size=n)
        b = rng.normal(loc=-0.001, scale=_DAILY_VOL, size=n)
        ci = paired_bootstrap_sharpe_differential(
            a, b, n_resamples=_SHARPE_CI_RESAMPLES, rng=_make_rng()
        )
        assert ci.excludes(0.0)
        assert ci.point_estimate > 0.0

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            paired_bootstrap_sharpe_differential(
                np.array([0.01, 0.02, 0.03]), np.array([0.01, 0.02])
            )


class TestDieboldMariano:
    def test_identical_forecasts_return_tie_and_p_one(self) -> None:
        rng = np.random.default_rng(404)
        y = rng.normal(size=_N_BARS)
        result = diebold_mariano_test(y, y, y)
        assert isinstance(result, DMResult)
        assert result.direction is DMDirection.TIE
        assert result.p_value == 1.0

    def test_biased_forecast_a_detected_as_worse(self) -> None:
        """
        Forecaster a has a large additive bias; b has zero error.
        DM should reject equality with direction=B (b forecasts better).
        """

        rng = np.random.default_rng(505)
        y = rng.normal(size=_N_BARS)
        b = y.copy()  # perfect forecaster
        a = y + 0.5  # constant bias
        result = diebold_mariano_test(a, b, y, h=1, loss=DMLoss.MSE)
        assert result.direction is DMDirection.B
        assert result.p_value < 0.01

    def test_rejects_unaligned_shapes(self) -> None:
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.0, 0.0])
        y = np.array([0.1, 0.1, 0.1])
        with pytest.raises(ValueError, match="aligned"):
            diebold_mariano_test(a, b, y)

    def test_rejects_non_positive_horizon(self) -> None:
        y = np.zeros(_N_BARS)
        with pytest.raises(ValueError, match="h"):
            diebold_mariano_test(y, y, y, h=0)

    def test_mae_loss_also_detects_biased_forecast(self) -> None:
        """
        MAE and MSE should agree on which forecaster is better when
        the bias is large - the HLN-corrected statistic flips sign with
        the mean loss differential, independent of the loss choice.
        """

        rng = np.random.default_rng(606)
        y = rng.normal(size=_N_BARS)
        a = y + 0.5
        b = y.copy()
        mae_result = diebold_mariano_test(a, b, y, loss=DMLoss.MAE)
        assert mae_result.direction is DMDirection.B
        assert mae_result.p_value < 0.01


_DSR_N_TRIALS = 50
_DSR_SAMPLE_LENGTH = 1000
_DSR_SIGNIFICANT_THRESHOLD = 0.95
_DSR_NOISE_UPPER_BAND = 0.6
_DSR_NOISE_SCALE = 0.1
_DSR_OUTLIER_SHARPE = 2.0
_DSR_NOISE_POOL_SCALE = 0.2
_DSR_PERFECT_SHARPE_PLATEAU = 0.3


class TestDeflatedSharpeRatio:
    """
    The DSR is high when the best Sharpe is far above the trial-pool's
    expected maximum, low when the best Sharpe is consistent with what
    a noisy search would produce by chance.
    """

    def test_single_trial_collapses_to_one_sided_p_value(self) -> None:
        """
        N=1 => no selection penalty (E[max] = 0); DSR is the
        standard-normal CDF of ``Sh*sqrt(T-1)``.
        """

        result = deflated_sharpe_ratio([0.5], sample_length=_DSR_SAMPLE_LENGTH)
        assert isinstance(result, DeflatedSharpe)
        assert result.n_trials == 1
        assert result.expected_max_sharpe == 0.0
        assert 0.0 <= result.deflated_sharpe <= 1.0
        assert result.deflated_sharpe == pytest.approx(1.0 - result.p_value)

    def test_outlier_best_sharpe_is_significant(self) -> None:
        """
        49 trials drawn from N(0, 0.1^2) plus one outlier at 2.0 -
        the outlier is way past the expected maximum and should deflate
        to a near-1 probability.
        """

        rng = np.random.default_rng(_BOOTSTRAP_SEED)
        trials = rng.normal(loc=0.0, scale=_DSR_NOISE_SCALE, size=_DSR_N_TRIALS - 1).tolist()
        trials.append(_DSR_OUTLIER_SHARPE)
        result = deflated_sharpe_ratio(trials, sample_length=_DSR_SAMPLE_LENGTH)
        assert result.observed_sharpe == _DSR_OUTLIER_SHARPE
        assert result.n_trials == _DSR_N_TRIALS
        assert result.deflated_sharpe > _DSR_SIGNIFICANT_THRESHOLD

    def test_max_of_noise_pool_is_not_significant(self) -> None:
        """
        50 trials all drawn from a centred normal - the best is the
        sample maximum of pure noise, and the DSR should sit well below
        any reasonable significance threshold.
        """

        rng = np.random.default_rng(_BOOTSTRAP_SEED + 1)
        trials = rng.normal(loc=0.0, scale=_DSR_NOISE_POOL_SCALE, size=_DSR_N_TRIALS).tolist()
        result = deflated_sharpe_ratio(trials, sample_length=_DSR_SAMPLE_LENGTH)
        assert result.deflated_sharpe < _DSR_NOISE_UPPER_BAND

    def test_zero_variance_trial_pool_collapses_penalty(self) -> None:
        """
        All trials identical => trial variance = 0 => E[max] = 0;
        deflated value equals the plain one-sided p of the observed Sharpe.
        """

        trials = [_DSR_PERFECT_SHARPE_PLATEAU] * _DSR_N_TRIALS
        result = deflated_sharpe_ratio(trials, sample_length=_DSR_SAMPLE_LENGTH)
        assert result.expected_max_sharpe == 0.0
        assert result.trial_sharpe_variance == 0.0

    def test_round_trip_through_dict(self) -> None:
        result = deflated_sharpe_ratio([0.1, 0.2, 0.5], sample_length=200)
        restored = DeflatedSharpe.from_dict(result.to_dict())
        assert restored == result

    def test_rejects_empty_trials(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            deflated_sharpe_ratio([], sample_length=100)

    def test_rejects_short_sample(self) -> None:
        with pytest.raises(ValueError, match="sample_length"):
            deflated_sharpe_ratio([0.1, 0.2], sample_length=1)
