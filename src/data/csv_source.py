"""
CSV file data source implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.core.registry import data_source_registry
from src.data.local_file_source import LocalFileSource


@data_source_registry.register("csv")
class CSVSource(LocalFileSource):
    """
    Data source for local CSV files.

    Expects the first column to be a parseable timestamp (``parse_dates=[0]``)
    so the index round-trips as ``DatetimeIndex``. Adds a NaT post-check
    because pandas silently produces ``NaT`` rows on unparseable strings.
    """

    _extension: ClassVar[str] = "csv"

    @property
    def name(self) -> str:
        return "csv"

    def _read_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Failed to parse dates in {path}: index dtype is "
                f"{df.index.dtype}, expected datetime; fix by ensuring the "
                f"first column holds ISO-formatted timestamps (YYYY-MM-DD)."
            )
        na_mask = pd.Series(df.index.isna())
        if na_mask.any():
            n_nat = int(na_mask.sum())
            raise ValueError(
                f"Failed to parse {n_nat} date(s) in {path} (got NaT "
                f"values); fix by repairing the date column in the CSV "
                f"(typical cause: blank rows or non-ISO date strings)."
            )
        return df
