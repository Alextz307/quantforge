"""Parquet file data source implementation.

Mirrors :class:`src.data.csv_source.CSVSource`: a local-file source that
reads OHLCV bars from ``{ticker}.parquet`` under a configurable directory.
Used by the thesis-demo target so ``make thesis-demo`` runs offline against
a committed fixture instead of hitting Yahoo.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.registry import data_source_registry
from src.core.types import Interval
from src.data.cache import DataCache
from src.data.interface import IDataSource
from src.data.normalizer import DataNormalizer


@data_source_registry.register("parquet")
class ParquetSource(IDataSource):
    """Data source for local Parquet files.

    Parquet preserves dtypes on disk, so unlike :class:`CSVSource` there
    is no string-date parsing fallback or NaT-validation step.
    """

    def __init__(
        self,
        data_dir: str | Path = ".",
        cache: DataCache | None = None,
        normalizer: DataNormalizer | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        super().__init__(cache=cache, normalizer=normalizer or DataNormalizer("yfinance"))

    @property
    def name(self) -> str:
        return "parquet"

    def fetch_raw(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:
        parquet_path = self._data_dir / f"{ticker}.parquet"
        try:
            df = pd.read_parquet(parquet_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Parquet file not found: {parquet_path}; fix by placing a "
                f"{ticker}.parquet file under the data_dir or by passing a "
                f"ticker that matches an existing file."
            ) from None

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Parquet file {parquet_path} has index dtype {df.index.dtype}; "
                f"expected DatetimeIndex. Fix by re-writing the file with "
                f"``df.to_parquet(...)`` after promoting the date column to "
                f"the index."
            )

        df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]

        if df.empty:
            raise ValueError(
                f"No data in {parquet_path} for range {start} to {end}; fix by "
                f"widening the date range or by checking the parquet covers "
                f"the requested window."
            )

        return df

    def available_tickers(self) -> list[str]:
        return [p.stem for p in self._data_dir.glob("*.parquet")]
