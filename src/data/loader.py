"""YFinance data source implementation."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

from src.core.exceptions import DataQualityError
from src.core.registry import data_source_registry
from src.core.types import Interval
from src.data.cache import DataCache
from src.data.interface import IDataSource
from src.data.normalizer import DataNormalizer

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0

_INTERVAL_MAP: dict[Interval, str] = {
    Interval.SECOND: "1s",
    Interval.MINUTE: "1m",
    Interval.FIVE_MINUTE: "5m",
    Interval.FIFTEEN_MINUTE: "15m",
    Interval.HOUR: "1h",
    Interval.DAILY: "1d",
    Interval.WEEKLY: "1wk",
}


@data_source_registry.register("yfinance")
class YFinanceSource(IDataSource):
    """Data source backed by the yfinance library."""

    def __init__(
        self,
        cache: DataCache | None = None,
        normalizer: DataNormalizer | None = None,
    ) -> None:
        super().__init__(cache=cache, normalizer=normalizer)

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch_raw(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance with retry + exponential backoff."""
        yf_interval = _INTERVAL_MAP[interval]
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                df: pd.DataFrame = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    raise DataQualityError(
                        f"No data returned from yfinance for {ticker} "
                        f"({start} to {end}, interval={interval.value})"
                    )
                return df
            except DataQualityError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
                    jitter = random.uniform(0, delay * 0.5)  # noqa: S311
                    logger.warning(
                        "yfinance fetch attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                        attempt + 1,
                        _MAX_RETRIES,
                        ticker,
                        exc,
                        delay + jitter,
                    )
                    time.sleep(delay + jitter)

        assert last_exc is not None  # unreachable: _MAX_RETRIES >= 1
        raise last_exc

    def available_tickers(self) -> list[str]:
        """YFinance supports any ticker — return empty list."""
        return []
