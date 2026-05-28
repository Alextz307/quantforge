"""
Shared scaffolding for local-file data sources (CSV, Parquet).
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.core.types import Interval
from src.data.cache import DataCache
from src.data.interface import IDataSource
from src.data.normalizer import DataNormalizer


class LocalFileSource(IDataSource):
    """
    ABC for sources that read OHLCV from a per-ticker file under a directory.

    Concrete subclasses provide:

    * ``_extension``: file suffix without the dot (e.g. ``"csv"``).
    * ``name`` (from :class:`IDataSource`): registry name.
    * ``_read_file(path)``: open the on-disk format and return a DataFrame
      whose index is a well-formed ``DatetimeIndex``. Subclasses MUST raise
      a format-specific :class:`ValueError` if the index is not a
      ``DatetimeIndex`` (or contains ``NaT`` for text formats), since the
      base class assumes a sorted ``DatetimeIndex`` for date-range filtering.

    Owns:

    * Path resolution (``data_dir / f"{ticker}.{ext}"``).
    * ``FileNotFoundError`` translation into a user-friendly message.
    * Inclusive date-range filter via boolean mask (works for unsorted
      indices too, where label slicing would silently drop rows).
    * Empty-result detection.
    * ``available_tickers`` via ``glob``.
    """

    _extension: ClassVar[str]

    def __init__(
        self,
        data_dir: str | Path = ".",
        cache: DataCache | None = None,
        normalizer: DataNormalizer | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        super().__init__(cache=cache, normalizer=normalizer)

    @abstractmethod
    def _read_file(self, path: Path) -> pd.DataFrame:
        """
        Read the on-disk file and return an OHLCV frame with DatetimeIndex.
        """

    def fetch_raw(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:
        path = self._data_dir / f"{ticker}.{self._extension}"
        try:
            df = self._read_file(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{self._extension} file not found: {path}; fix by placing a "
                f"{ticker}.{self._extension} file under the data_dir or by passing "
                f"a ticker that matches an existing file."
            ) from None

        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        df = df.loc[mask]

        if df.empty:
            raise ValueError(
                f"No data in {path} for range {start} to {end}; fix by widening "
                f"the date range or by checking the file covers the requested window."
            )

        return df

    def available_tickers(self) -> list[str]:
        return [p.stem for p in self._data_dir.glob(f"*.{self._extension}")]
