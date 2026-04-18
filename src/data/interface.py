"""Data source abstract interface with built-in caching and normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from src.core.types import Interval
from src.data.cache import DataCache
from src.data.normalizer import DataNormalizer
from src.data.validator import validate_bars


class IDataSource(ABC):
    """Pluggable data source — swap yfinance for any provider.

    Concrete implementations only need to implement fetch_raw().
    The base class handles caching, column normalization, and ingestion-time
    quality validation: every freshly fetched frame is checked by
    ``validate_bars`` (NaN, non-positive prices, OHLC ordering, duplicate
    timestamps) before it reaches the cache, so bad data never fans out to
    downstream strategies or the C++ engine.
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
                cached = self.cache.load(cache_key)
            except FileNotFoundError:
                pass
            else:
                # Re-validate cached frames: a cache written by an older code version
                # (weaker or absent validator) should not silently bypass today's checks.
                validate_bars(cached)
                return cached

        df = self.fetch_raw(ticker, start, end, interval)
        df = self.normalizer.normalize(df)
        validate_bars(df)

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
