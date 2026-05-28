"""
Regenerate the committed ``tests/fixtures/SPY.parquet`` test fixture.

Fetches SPY daily bars from yfinance for the date range cited in
``tests/fixtures/README.md``, flattens the multi-index column header that
``yf.download`` returns even for a single ticker, normalizes via
:class:`DataNormalizer`, validates via :func:`validate_bars`, and writes
the result back to ``tests/fixtures/SPY.parquet``.

Run from the repo root:

    python scripts/regen_spy_fixture.py

The flatten step is required because ``YFinanceSource.fetch_raw`` does not
currently flatten yfinance's column multi-index, so the normalizer rejects
the raw frame. Doing it once at write time keeps the on-disk file in the
canonical OHLCV schema, which is what :class:`ParquetSource` reads back.

``auto_adjust=True`` is intentional — adjusted closes already fold splits
and dividends into the price series, so downstream return / strategy math
matches what a long-only buy-and-hold backtest would produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yfinance as yf

from src.data.normalizer import DataNormalizer
from src.data.validator import validate_bars

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "SPY.parquet"


def main() -> int:
    raw = yf.download(
        "SPY",
        start="2018-01-01",
        end="2024-12-31",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        print(
            "ERROR: yfinance returned no data for SPY 2018-01-01..2024-12-31; "
            "check your network connection or yfinance availability.",
            file=sys.stderr,
        )
        return 1

    if raw.columns.nlevels > 1:
        raw.columns = raw.columns.get_level_values(0)

    df = DataNormalizer("yfinance").normalize(raw)
    validate_bars(df)
    df.to_parquet(FIXTURE_PATH)
    print(
        f"wrote {FIXTURE_PATH.relative_to(REPO_ROOT)} "
        f"({len(df)} bars, {df.index.min().date()} → {df.index.max().date()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
