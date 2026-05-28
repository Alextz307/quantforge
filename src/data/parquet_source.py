"""
Parquet file data source implementation.

Mirrors :class:`src.data.csv_source.CSVSource`: a local-file source that
reads OHLCV bars from ``{ticker}.parquet`` under a configurable directory.
Lets offline tests run against a committed fixture instead of hitting Yahoo.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.core.registry import data_source_registry
from src.data.local_file_source import LocalFileSource


@data_source_registry.register("parquet")
class ParquetSource(LocalFileSource):
    """
    Data source for local Parquet files.

    Parquet preserves dtypes on disk, so unlike :class:`CSVSource` there is
    no string-date parsing fallback or NaT post-check — the index either
    round-trips as ``DatetimeIndex`` or the file was written wrong.
    """

    _extension: ClassVar[str] = "parquet"

    @property
    def name(self) -> str:
        return "parquet"

    def _read_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Parquet file {path} has index dtype {df.index.dtype}; "
                f"expected DatetimeIndex. Fix by re-writing the file with "
                f"``df.to_parquet(...)`` after promoting the date column to "
                f"the index."
            )
        return df
