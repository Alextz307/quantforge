"""Cointegration testing for pairs trading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CointegrationResult:
    """Result of an Engle-Granger cointegration test."""

    hedge_ratio: float
    p_value: float
    spread_mean: float
    spread_std: float
    is_cointegrated: bool


class CointegrationTester:
    """Cointegration testing for pairs trading.

    Uses the Engle-Granger two-step method:
    1. OLS regression to find hedge ratio
    2. ADF test on residuals for stationarity
    """

    @staticmethod
    def engle_granger(
        series_a: pd.Series[float],
        series_b: pd.Series[float],
        p_value_threshold: float = 0.05,
    ) -> CointegrationResult:
        """Run Engle-Granger cointegration test on two price series.

        Args:
            series_a: First price series.
            series_b: Second price series (regressor).
            p_value_threshold: ADF p-value cutoff for cointegration.

        Returns:
            CointegrationResult with hedge ratio, p-value, and spread stats.
        """
        # OLS for hedge ratio and spread statistics
        y = series_a.values
        x = add_constant(series_b.values)
        model = OLS(y, x).fit()
        hedge_ratio = float(model.params[1])
        spread: pd.Series[float] = series_a - hedge_ratio * series_b

        # statsmodels.coint uses correct MacKinnon critical values
        _, p_value_raw, _ = coint(series_a.values, series_b.values, autolag="AIC")
        p_value = float(p_value_raw)

        return CointegrationResult(
            hedge_ratio=hedge_ratio,
            p_value=p_value,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            is_cointegrated=p_value < p_value_threshold,
        )

    @staticmethod
    def find_cointegrated_pairs(
        price_matrix: pd.DataFrame,
        p_value_threshold: float = 0.05,
    ) -> list[tuple[str, str, CointegrationResult]]:
        """Screen all column pairs for cointegration.

        Args:
            price_matrix: DataFrame where each column is a price series.
            p_value_threshold: Maximum ADF p-value for cointegration.

        Returns:
            List of (col_a, col_b, result) tuples for cointegrated pairs.
        """
        columns = list(price_matrix.columns)
        results: list[tuple[str, str, CointegrationResult]] = []

        for col_a, col_b in combinations(columns, 2):
            try:
                result = CointegrationTester.engle_granger(
                    price_matrix[col_a],
                    price_matrix[col_b],
                    p_value_threshold,
                )
                if result.is_cointegrated:
                    results.append((str(col_a), str(col_b), result))
            except (ValueError, RuntimeError):
                logger.warning("Cointegration test failed for %s/%s", col_a, col_b)

        return results
