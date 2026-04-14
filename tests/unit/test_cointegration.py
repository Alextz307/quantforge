"""Tests for CointegrationTester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.cointegration import CointegrationTester


def _random_walk(
    seed: int,
    n: int = 500,
    start: str = "2020-01-02",
    mu: float = 0.001,
    sigma: float = 0.02,
) -> pd.Series[float]:
    """Generate a random-walk price series for testing."""
    np.random.seed(seed)
    idx = pd.bdate_range(start=start, periods=n, freq="B")
    prices = 100.0 * np.cumprod(1 + np.random.normal(mu, sigma, n))
    return pd.Series(prices, index=idx)


@pytest.fixture
def cointegrated_pair() -> tuple[pd.Series[float], pd.Series[float]]:
    """Two synthetic cointegrated series: B is a random walk, A = 2*B + noise."""
    b = _random_walk(seed=42)
    np.random.seed(43)
    noise = np.random.normal(0, 0.5, len(b))
    a = pd.Series(2.0 * np.asarray(b.values, dtype=np.float64) + noise, index=b.index, name="A")
    b.name = "B"
    return a, b


@pytest.fixture
def independent_pair() -> tuple[pd.Series[float], pd.Series[float]]:
    """Two independent random walks — should NOT be cointegrated."""
    a = _random_walk(seed=99)
    a.name = "X"
    b = _random_walk(seed=100)
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
        assert result.p_value < 0.05

    def test_hedge_ratio_accuracy(
        self,
        cointegrated_pair: tuple[pd.Series[float], pd.Series[float]],
    ) -> None:
        """Recovered hedge ratio should be close to the true ratio of 2.0."""
        a, b = cointegrated_pair
        result = CointegrationTester.engle_granger(a, b)
        assert abs(result.hedge_ratio - 2.0) < 0.1

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
        c = _random_walk(seed=50, mu=0.0)
        d = _random_walk(seed=51, mu=0.0)
        p = _random_walk(seed=52)

        np.random.seed(53)
        p_arr = np.asarray(p.values, dtype=np.float64)
        q = pd.Series(1.5 * p_arr + np.random.normal(0, 0.3, len(p)), index=p.index)

        matrix = pd.DataFrame(
            {"C": c.values, "D": d.values, "P": p.values, "Q": q.values},
            index=p.index,
        )
        results = CointegrationTester.find_cointegrated_pairs(matrix)

        pair_sets = [{a, b} for a, b, _ in results]
        assert {"P", "Q"} in pair_sets
