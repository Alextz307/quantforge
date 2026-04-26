"""CSV file data source implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.registry import data_source_registry
from src.core.types import Interval
from src.data.cache import DataCache
from src.data.interface import IDataSource
from src.data.normalizer import DataNormalizer


@data_source_registry.register("csv")
class CSVSource(IDataSource):
    """Data source for local CSV files."""

    def __init__(
        self,
        data_dir: str | Path = ".",
        cache: DataCache | None = None,
        normalizer: DataNormalizer | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        super().__init__(cache=cache, normalizer=normalizer or DataNormalizer("csv"))

    @property
    def name(self) -> str:
        return "csv"

    def fetch_raw(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        Expects a file named {ticker}.csv in the data directory.
        The CSV must have a date/timestamp column and OHLCV columns.
        """
        csv_path = self._data_dir / f"{ticker}.csv"
        try:
            df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}; fix by placing a "
                f"{ticker}.csv file under the data_dir or by passing a "
                f"ticker that matches an existing file."
            ) from None

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Failed to parse dates in {csv_path}: index dtype is "
                f"{df.index.dtype}, expected datetime; fix by ensuring the "
                f"first column holds ISO-formatted timestamps (YYYY-MM-DD)."
            )
        na_mask = pd.Series(df.index.isna())
        if na_mask.any():
            n_nat = int(na_mask.sum())
            raise ValueError(
                f"Failed to parse {n_nat} date(s) in {csv_path} (got NaT "
                f"values); fix by repairing the date column in the CSV "
                f"(typical cause: blank rows or non-ISO date strings)."
            )

        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        df = df.loc[mask]

        if df.empty:
            raise ValueError(
                f"No data in {csv_path} for range {start} to {end}; fix by "
                f"widening the date range or by checking the CSV covers the "
                f"requested window."
            )

        return df

    def available_tickers(self) -> list[str]:
        """List CSV files available in the data directory."""
        return [p.stem for p in self._data_dir.glob("*.csv")]
