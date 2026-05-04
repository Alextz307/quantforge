"""Column normalization across different data sources.

Different providers use different column naming conventions.
This normalizes everything to: timestamp (index), open, high, low, close, volume.
"""

from __future__ import annotations

import pandas as pd

from src.core.constants import OHLCV_COLUMNS


class DataNormalizer:
    """Normalizes DataFrame columns from various data sources."""

    KNOWN_MAPPINGS: dict[str, dict[str, str]] = {
        "yfinance": {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        "polygon": {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        },
    }

    REQUIRED_COLUMNS = set(OHLCV_COLUMNS)

    def __init__(self, source_name: str = "yfinance") -> None:
        self.source_name = source_name
        self._mapping = self.KNOWN_MAPPINGS.get(source_name, {})

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and ensure DatetimeIndex.

        Args:
            df: Raw DataFrame from the data source.

        Returns:
            DataFrame with standardized column names and DatetimeIndex.
        """
        result = df.copy()

        # yfinance returns a (metric, ticker) MultiIndex even for single-ticker
        # downloads in newer versions. Drop the ticker level so the rename map
        # matches against flat column names.
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(0)

        if self._mapping:
            result = result.rename(columns=self._mapping)

        result.columns = pd.Index([str(c).lower() for c in result.columns])

        # Ensure DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            if "date" in result.columns:
                result = result.set_index("date")
                result.index = pd.DatetimeIndex(result.index)
            elif "timestamp" in result.columns:
                result = result.set_index("timestamp")
                result.index = pd.DatetimeIndex(result.index)

        result = result.sort_index()

        missing = self.REQUIRED_COLUMNS - set(result.columns)
        if missing:
            raise ValueError(
                f"Missing required columns after normalization: {missing}; "
                f"available: {list(result.columns)}. Fix by extending the "
                f"upstream source to emit the OHLCV columns or by adding a "
                f"per-source rename map covering the missing names."
            )

        # auto_adjust=True (yfinance) rescales close for splits/dividends but
        # leaves high/low approximate, occasionally violating the OHLC ordering
        # invariant by tiny noise. Snap the envelope so downstream consumers
        # (validator, indicators, C++ engine) see internally consistent bars.
        if not result.empty:
            result["high"] = result[["open", "high", "close"]].max(axis=1)
            result["low"] = result[["open", "low", "close"]].min(axis=1)

        return result
