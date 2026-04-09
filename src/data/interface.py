"""Data source abstract interface with built-in caching and normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from src.core.types import Interval
from src.data.cache import DataCache
from src.data.normalizer import DataNormalizer


class IDataSource(ABC):
    """Pluggable data source — swap yfinance for any provider.

    Concrete implementations only need to implement fetch_raw().
    The base class provides caching, normalization, and validation.
    """

    def __init__(
        self,
        cache: DataCache | None = None,
        normalizer: DataNormalizer | None = None,
    ) -> None:
        self.cache = cache
        self.normalizer = normalizer or DataNormalizer(self.name)

    @abstractmethod
    def fetch_raw(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:
        """Fetch raw OHLCV data from the source."""

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:
        """Fetch data with caching + normalization.

        Args:
            ticker: Stock ticker symbol.
            start: Start date for data range.
            end: End date for data range.
            interval: Bar interval (default: DAILY).

        Returns:
            Normalized DataFrame with DatetimeIndex and standard columns.
        """
        cache_key = self._cache_key(ticker, start, end, interval)

        if self.cache is not None:
            try:
                return self.cache.load(cache_key)
            except FileNotFoundError:
                pass

        df = self.fetch_raw(ticker, start, end, interval)
        df = self.normalizer.normalize(df)

        # TODO: Phase 2 — enable C++ data validation
        # try:
        #     import quant_engine
        #     bars = dataframe_to_bars(df)
        #     report = quant_engine.validate_data(bars)
        #     if not report.is_valid:
        #         raise DataQualityError(report)
        # except ImportError:
        #     pass

        if self.cache is not None:
            self.cache.save(cache_key, df)

        return df

    @abstractmethod
    def available_tickers(self) -> list[str]:
        """List available ticker symbols for this source."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Data source identifier."""

    def _cache_key(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval,
    ) -> str:
        """Generate a deterministic cache key."""
        return f"{self.name}_{ticker}_{start.isoformat()}_{end.isoformat()}_{interval.value}"
