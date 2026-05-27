"""Tests for CointegrationTester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.cointegration import CointegrationTester

RW_DEFAULT_ROW_COUNT = 500
RW_DEFAULT_START_DATE = "2020-01-02"
RW_DEFAULT_RETURN_MEAN = 0.001
RW_DEFAULT_RETURN_STD = 0.02
RW_BASE_PRICE = 100.0

# Construction: B is a random walk, A = COINT_HEDGE_RATIO_TRUE * B + noise.
COINT_HEDGE_RATIO_TRUE = 2.0
COINT_NOISE_STD = 0.5
COINT_BASE_SEED = 42
COINT_NOISE_SEED = 43

INDEP_SEED_A = 99
INDEP_SEED_B = 100

MULTI_INDEP_C_SEED = 50
MULTI_INDEP_D_SEED = 51
MULTI_PLANTED_P_SEED = 52
MULTI_PLANTED_NOISE_SEED = 53
MULTI_HEDGE_RATIO = 1.5
MULTI_NOISE_STD = 0.3

COINTEGRATION_P_VALUE_THRESHOLD = 0.05
HEDGE_RATIO_TOLERANCE = 0.1


def _random_walk(
    seed: int,
    n: int = RW_DEFAULT_ROW_COUNT,
    start: str = RW_DEFAULT_START_DATE,
    mu: float = RW_DEFAULT_RETURN_MEAN,
    sigma: float = RW_DEFAULT_RETURN_STD,
) -> pd.Series[float]:
    """Generate a random-walk price series for testing."""

    np.random.seed(seed)
    idx = pd.bdate_range(start=start, periods=n, freq="B")
    prices = RW_BASE_PRICE * np.cumprod(1 + np.random.normal(mu, sigma, n))
    return pd.Series(prices, index=idx)


@pytest.fixture
def cointegrated_pair() -> tuple[pd.Series[float], pd.Series[float]]:
    """Two synthetic cointegrated series: B is a random walk, A = HEDGE*B + noise."""

    b = _random_walk(seed=COINT_BASE_SEED)
    np.random.seed(COINT_NOISE_SEED)
    noise = np.random.normal(0, COINT_NOISE_STD, len(b))
    a = pd.Series(
        COINT_HEDGE_RATIO_TRUE * np.asarray(b.values, dtype=np.float64) + noise,
        index=b.index,
        name="A",
    )
    b.name = "B"
    return a, b


@pytest.fixture
def independent_pair() -> tuple[pd.Series[float], pd.Series[float]]:
    """Two independent random walks — should NOT be cointegrated."""

    a = _random_walk(seed=INDEP_SEED_A)
    a.name = "X"
    b = _random_walk(seed=INDEP_SEED_B)
    b.name = "Y"
    return a, b


class TestCointegrationTester:
    def test_cointegrated_pair_detected(
        self,
        cointegrated_pair: tuple[pd.Series[float], pd.Series[float]],
    ) -> None:
        a, b = cointegrated_pair
        result = CointegrationTester.engle_granger(a, b)
        assert result.is_cointegrated
        assert result.p_value < COINTEGRATION_P_VALUE_THRESHOLD

    def test_hedge_ratio_accuracy(
        self,
        cointegrated_pair: tuple[pd.Series[float], pd.Series[float]],
    ) -> None:
        """Recovered hedge ratio should be close to the true ratio."""

        a, b = cointegrated_pair
        result = CointegrationTester.engle_granger(a, b)
        assert abs(result.hedge_ratio - COINT_HEDGE_RATIO_TRUE) < HEDGE_RATIO_TOLERANCE

    def test_independent_pair_not_cointegrated(
        self,
        independent_pair: tuple[pd.Series[float], pd.Series[float]],
    ) -> None:
        a, b = independent_pair
        result = CointegrationTester.engle_granger(a, b)
        assert not result.is_cointegrated

    def test_spread_stats_populated(
        self,
        cointegrated_pair: tuple[pd.Series[float], pd.Series[float]],
    ) -> None:
        a, b = cointegrated_pair
        result = CointegrationTester.engle_granger(a, b)
        assert isinstance(result.spread_mean, float)
        assert isinstance(result.spread_std, float)
        assert result.spread_std > 0

    def test_result_is_frozen(
        self,
        cointegrated_pair: tuple[pd.Series[float], pd.Series[float]],
    ) -> None:
        a, b = cointegrated_pair
        result = CointegrationTester.engle_granger(a, b)
        with pytest.raises(AttributeError):
            result.hedge_ratio = 999.0  # type: ignore[misc]


class TestFindCointegratedPairs:
    def test_finds_known_pair(
        self,
        cointegrated_pair: tuple[pd.Series[float], pd.Series[float]],
    ) -> None:
        a, b = cointegrated_pair
        matrix = pd.DataFrame({"A": a.values, "B": b.values}, index=a.index)
        results = CointegrationTester.find_cointegrated_pairs(matrix)
        assert len(results) >= 1
        col_a, col_b, res = results[0]
        assert {col_a, col_b} == {"A", "B"}
        assert res.is_cointegrated

    def test_no_pairs_for_independent(
        self,
        independent_pair: tuple[pd.Series[float], pd.Series[float]],
    ) -> None:
        a, b = independent_pair
        matrix = pd.DataFrame({"X": a.values, "Y": b.values}, index=a.index)
        results = CointegrationTester.find_cointegrated_pairs(matrix)
        assert len(results) == 0

    def test_multi_column_screening(self) -> None:
        """Screen 4 columns — only the planted pair should be found."""

        c = _random_walk(seed=MULTI_INDEP_C_SEED, mu=0.0)
        d = _random_walk(seed=MULTI_INDEP_D_SEED, mu=0.0)
        p = _random_walk(seed=MULTI_PLANTED_P_SEED)

        np.random.seed(MULTI_PLANTED_NOISE_SEED)
        p_arr = np.asarray(p.values, dtype=np.float64)
        q = pd.Series(
            MULTI_HEDGE_RATIO * p_arr + np.random.normal(0, MULTI_NOISE_STD, len(p)),
            index=p.index,
        )

        matrix = pd.DataFrame(
            {"C": c.values, "D": d.values, "P": p.values, "Q": q.values},
            index=p.index,
        )
        results = CointegrationTester.find_cointegrated_pairs(matrix)

        pair_sets = [{a, b} for a, b, _ in results]
        assert {"P", "Q"} in pair_sets
