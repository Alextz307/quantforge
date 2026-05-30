"""
Local Parquet cache to avoid re-downloading data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.core.fs import atomic_write_path


class DataCache:
    """
    Local Parquet cache - no re-downloading on subsequent runs.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".quant_cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """
        Convert a cache key to a file path.
        """

        safe_name = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{safe_name}.parquet"

    def has(self, key: str) -> bool:
        """
        Check if a cache entry exists.
        """

        return self._key_to_path(key).exists()

    def load(self, key: str) -> pd.DataFrame:
        """
        Load a cached DataFrame.
        """

        path = self._key_to_path(key)
        try:
            return pd.read_parquet(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache miss for key: {key}") from None

    def save(self, key: str, df: pd.DataFrame) -> None:
        """
        Save a DataFrame to cache atomically.

        Parallel HPO trials race on the same cache key. Writing directly
        to the final path lets a reader observe a half-written parquet
        between the writer's truncate and close - the symptom is a frame
        with duplicated columns that breaks downstream Series ops.
        ``atomic_write_path`` stages to a per-(pid,tid) tmp file, then
        ``os.replace`` to commit.
        """

        with atomic_write_path(self._key_to_path(key)) as tmp:
            df.to_parquet(tmp)

    def invalidate(self, key: str) -> None:
        """
        Remove a cache entry.
        """

        self._key_to_path(key).unlink(missing_ok=True)

    def clear(self) -> None:
        """
        Remove all cached data.
        """

        for path in self.cache_dir.glob("*.parquet"):
            path.unlink(missing_ok=True)
