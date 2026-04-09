"""YFinance data source implementation."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from src.core.registry import data_source_registry
from src.core.types import Interval
from src.data.cache import DataCache
from src.data.interface import IDataSource
from src.data.normalizer import DataNormalizer

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
        """Fetch OHLCV data from Yahoo Finance."""
        yf_interval = _INTERVAL_MAP.get(interval, "1d")
        df: pd.DataFrame = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(
                f"No data returned from yfinance for {ticker} "
                f"({start} to {end}, interval={interval.value})"
            )
        return df

    def available_tickers(self) -> list[str]:
        """YFinance supports any ticker — return empty list."""
        return []
