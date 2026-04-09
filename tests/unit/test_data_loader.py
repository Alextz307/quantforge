"""Tests for data layer: normalizer, cache, CSV source, and registry integration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache import DataCache
from src.data.csv_source import CSVSource
from src.data.normalizer import DataNormalizer


class TestDataNormalizer:
    def test_yfinance_column_renaming(self) -> None:
        normalizer = DataNormalizer("yfinance")
        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 4)])
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [103.0, 104.0, 105.0],
                "Volume": [1000, 2000, 3000],
            },
            index=idx,
        )
        result = normalizer.normalize(df)
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}

    def test_polygon_column_renaming(self) -> None:
        normalizer = DataNormalizer("polygon")
        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 4)])
        df = pd.DataFrame(
            {
                "o": [100.0, 101.0, 102.0],
                "h": [105.0, 106.0, 107.0],
                "l": [99.0, 100.0, 101.0],
                "c": [103.0, 104.0, 105.0],
                "v": [1000, 2000, 3000],
            },
            index=idx,
        )
        result = normalizer.normalize(df)
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}

    def test_missing_columns_raises(self) -> None:
        normalizer = DataNormalizer("unknown_source")
        idx = pd.DatetimeIndex([datetime(2024, 1, 1)])
        df = pd.DataFrame({"foo": [1.0], "bar": [2.0]}, index=idx)
        with pytest.raises(ValueError, match="Missing required columns"):
            normalizer.normalize(df)

    def test_sorts_by_index(self) -> None:
        normalizer = DataNormalizer("yfinance")
        idx = pd.DatetimeIndex([datetime(2024, 1, 3), datetime(2024, 1, 1), datetime(2024, 1, 2)])
        df = pd.DataFrame(
            {
                "Open": [102.0, 100.0, 101.0],
                "High": [107.0, 105.0, 106.0],
                "Low": [101.0, 99.0, 100.0],
                "Close": [105.0, 103.0, 104.0],
                "Volume": [3000, 1000, 2000],
            },
            index=idx,
        )
        result = normalizer.normalize(df)
        assert result.index.is_monotonic_increasing

    def test_date_column_becomes_index(self) -> None:
        normalizer = DataNormalizer("yfinance")
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [103.0, 104.0],
                "Volume": [1000, 2000],
            }
        )
        result = normalizer.normalize(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_preserves_extra_columns(self) -> None:
        normalizer = DataNormalizer("yfinance")
        idx = pd.DatetimeIndex([datetime(2024, 1, 1)])
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [103.0],
                "Volume": [1000.0],
                "Dividends": [0.5],
                "Stock Splits": [0.0],
            },
            index=idx,
        )
        result = normalizer.normalize(df)
        assert "dividends" in result.columns
        assert "stock splits" in result.columns


class TestDataCache:
    def test_save_and_load(self, tmp_path: Path) -> None:
        cache = DataCache(cache_dir=tmp_path)
        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 4)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

        cache.save("test_key", df)
        assert cache.has("test_key")

        loaded = cache.load("test_key")
        pd.testing.assert_frame_equal(df, loaded)

    def test_has_returns_false_for_missing(self, tmp_path: Path) -> None:
        cache = DataCache(cache_dir=tmp_path)
        assert not cache.has("nonexistent")

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        cache = DataCache(cache_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            cache.load("nonexistent")

    def test_invalidate(self, tmp_path: Path) -> None:
        cache = DataCache(cache_dir=tmp_path)
        idx = pd.DatetimeIndex([datetime(2024, 1, 1)])
        df = pd.DataFrame({"close": [1.0]}, index=idx)

        cache.save("key", df)
        assert cache.has("key")

        cache.invalidate("key")
        assert not cache.has("key")

    def test_invalidate_missing_key_is_noop(self, tmp_path: Path) -> None:
        cache = DataCache(cache_dir=tmp_path)
        cache.invalidate("nonexistent")  # should not raise

    def test_clear(self, tmp_path: Path) -> None:
        cache = DataCache(cache_dir=tmp_path)
        idx = pd.DatetimeIndex([datetime(2024, 1, 1)])
        df = pd.DataFrame({"close": [1.0]}, index=idx)

        cache.save("key1", df)
        cache.save("key2", df)
        assert cache.has("key1")
        assert cache.has("key2")

        cache.clear()
        assert not cache.has("key1")
        assert not cache.has("key2")


class TestCSVSource:
    def test_fetch_from_csv(self, tmp_path: Path) -> None:
        # Create a test CSV
        csv_path = tmp_path / "SPY.csv"
        idx = pd.bdate_range("2024-01-01", periods=20)
        df = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(20)],
                "high": [105.0 + i for i in range(20)],
                "low": [99.0 + i for i in range(20)],
                "close": [103.0 + i for i in range(20)],
                "volume": [1000 * (i + 1) for i in range(20)],
            },
            index=idx,
        )
        df.to_csv(csv_path)

        source = CSVSource(data_dir=tmp_path)
        result = source.fetch(
            "SPY",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )
        assert len(result) == 20
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}

    def test_fetch_missing_file_raises(self, tmp_path: Path) -> None:
        source = CSVSource(data_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            source.fetch(
                "MISSING",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 12, 31),
            )

    def test_available_tickers(self, tmp_path: Path) -> None:
        (tmp_path / "AAPL.csv").touch()
        (tmp_path / "MSFT.csv").touch()
        (tmp_path / "not_csv.txt").touch()

        source = CSVSource(data_dir=tmp_path)
        tickers = source.available_tickers()
        assert set(tickers) == {"AAPL", "MSFT"}

    def test_fetch_with_cache(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cache_dir = tmp_path / "cache"

        csv_path = data_dir / "SPY.csv"
        idx = pd.bdate_range("2024-01-01", periods=10)
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [105.0] * 10,
                "low": [99.0] * 10,
                "close": [103.0] * 10,
                "volume": [1000] * 10,
            },
            index=idx,
        )
        df.to_csv(csv_path)

        cache = DataCache(cache_dir=cache_dir)
        source = CSVSource(data_dir=data_dir, cache=cache)

        # First fetch — from CSV
        result1 = source.fetch(
            "SPY",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )
        assert len(result1) == 10

        # Second fetch — from cache (even if CSV is deleted)
        csv_path.unlink()
        result2 = source.fetch(
            "SPY",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )
        pd.testing.assert_frame_equal(result1, result2)

    def test_csv_with_unparseable_dates(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "BAD.csv"
        csv_path.write_text("date,open,high,low,close,volume\nnot_a_date,1,2,0.5,1.5,100\n")
        source = CSVSource(data_dir=tmp_path)
        with pytest.raises(ValueError, match="Failed to parse|No data"):
            source.fetch("BAD", start=datetime(2024, 1, 1), end=datetime(2024, 12, 31))

    def test_csv_date_range_filtering(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "RANGE.csv"
        idx = pd.bdate_range("2024-01-01", periods=60)
        df = pd.DataFrame(
            {
                "open": [100.0] * 60,
                "high": [105.0] * 60,
                "low": [99.0] * 60,
                "close": [103.0] * 60,
                "volume": [1000.0] * 60,
            },
            index=idx,
        )
        df.to_csv(csv_path)

        source = CSVSource(data_dir=tmp_path)
        result = source.fetch(
            "RANGE",
            start=datetime(2024, 2, 1),
            end=datetime(2024, 2, 28),
        )
        assert all(result.index >= pd.Timestamp("2024-02-01"))
        assert all(result.index <= pd.Timestamp("2024-02-28"))


class TestRegistryIntegration:
    def test_data_sources_are_registered(self) -> None:
        import src.data.csv_source  # noqa: F401
        import src.data.loader  # noqa: F401
        from src.core.registry import data_source_registry

        assert "yfinance" in data_source_registry
        assert "csv" in data_source_registry
