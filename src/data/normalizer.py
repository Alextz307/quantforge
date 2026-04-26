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

        # Apply column renaming if mapping exists
        if self._mapping:
            result = result.rename(columns=self._mapping)

        # Lowercase all remaining columns
        result.columns = pd.Index([str(c).lower() for c in result.columns])

        # Ensure DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            if "date" in result.columns:
                result = result.set_index("date")
                result.index = pd.DatetimeIndex(result.index)
            elif "timestamp" in result.columns:
                result = result.set_index("timestamp")
                result.index = pd.DatetimeIndex(result.index)

        # Sort by time
        result = result.sort_index()

        # Validate required columns exist
        missing = self.REQUIRED_COLUMNS - set(result.columns)
        if missing:
            raise ValueError(
                f"Missing required columns after normalization: {missing}; "
                f"available: {list(result.columns)}. Fix by extending the "
                f"upstream source to emit the OHLCV columns or by adding a "
                f"per-source rename map covering the missing names."
            )

        return result
