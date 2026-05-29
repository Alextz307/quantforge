# `src/data/`

OHLCV ingestion: pluggable sources, column normalisation, on-disk
caching, semantic quality validation, and a content fingerprint that
holdout-eval uses to detect vendor drift.

## Public surface

| Symbol | Role |
| --- | --- |
| `IDataSource` | ABC. `fetch(ticker, start, end, interval)` is the public entry point — wraps `fetch_raw` (subclass-provided) with caching, normalisation, and `validate_bars`. |
| `LocalFileSource` | Intermediate ABC: shared scaffolding for local-file sources (path resolution, date-range mask, empty-result error, `available_tickers`). Subclass overrides `_extension` + `_read_file`. |
| `YFinanceSource` (`"yfinance"`) | yfinance-backed source with retry + exponential backoff. Registered on `data_source_registry`. |
| `CSVSource` (`"csv"`) | Local-CSV source for offline / fixture work. |
| `ParquetSource` (`"parquet"`) | Local-parquet source. Used by offline tests against the committed `tests/fixtures/SPY.parquet`. |
| `DataNormalizer` | Source-aware column renamer (`Open` → `open`, etc.); enforces required OHLCV columns + `DatetimeIndex`. |
| `validate_bars(df)` | Raises `DataQualityError` on empty / NaN / non-positive prices / OHLC ordering / duplicate timestamps. Runs at every fetch. |
| `DataCache` | Parquet on-disk cache keyed by SHA-256 of `(source, ticker, start, end, interval)`. |
| `fingerprint_bars(df)` | SHA-256 content hash over (columns, timestamps, OHLCV bytes). Stable across pandas / numpy upgrades. |
| `fingerprint_pair_bars(df)` | Same hash, wide-format pair columns (`open_a` … `volume_b`). |
| `LiveBarFetcher` | Protocol: cadence-specific live OHLCV fetcher used by the deployment layer. |
| `DailyLiveBarFetcher` | Daily implementation backed by `YFinanceSource`; drops the trailing bar while its NYSE session is still open (so a signal is never computed off a forming bar) and rejects non-daily intervals defensively. |
| `resolve_fetcher(interval)` | Dispatch site — picks the right `LiveBarFetcher` for an interval. Daily today; intraday is a future drop-in. |

## Layout

| File | Role |
| --- | --- |
| `interface.py` | `IDataSource` ABC + `fetch` (cache + normalise + validate). |
| `local_file_source.py` | `LocalFileSource` ABC: shared parquet/CSV scaffolding. |
| `loader.py` | `YFinanceSource`. |
| `csv_source.py` | `CSVSource` (extends `LocalFileSource`). |
| `parquet_source.py` | `ParquetSource` (extends `LocalFileSource`). |
| `normalizer.py` | `DataNormalizer` (source → canonical OHLCV mapping). |
| `validator.py` | `validate_bars` semantic checks. |
| `cache.py` | `DataCache` (parquet, ~/.quant_cache by default). |
| `fingerprint.py` | `_fingerprint` core + `fingerprint_bars` / `fingerprint_pair_bars` wrappers. |
| `live_fetcher.py` | `LiveBarFetcher` protocol + `DailyLiveBarFetcher` + `resolve_fetcher` dispatcher (live-inference cadence layer). |

## Single vs pair fingerprint

Single-ticker runs use `fingerprint_bars`; two-ticker (pairs) runs use
`fingerprint_pair_bars`, which expects the `_a` / `_b` suffixed columns
emitted by the multi-ticker fetch path
(`src/orchestration/experiment.py::_fetch_pair_bars`). `Experiment.run`
picks the right fingerprint variant based on `len(cfg.data.tickers)`.

## Validation contract

`validate_bars` runs **at every fetch boundary** — both fresh fetches
and cache hits. A cache parquet written by an older code version (with
weaker validation) does not bypass current checks. Owned concerns:

- Required OHLCV columns + sorted `DatetimeIndex`
- No NaN, no NaT, no duplicate timestamps
- Positive prices; OHLC ordering invariants
- Non-negative volume

Explicitly NOT covered: bars / signals alignment (engine), train / test
overlap (temporal metadata), gap detection (interval-aware, owned by
callers).

## Snippet

```python
from datetime import datetime

from src.data.cache import DataCache
from src.data.fingerprint import fingerprint_bars
from src.data.loader import YFinanceSource

source = YFinanceSource(cache=DataCache())
bars = source.fetch("SPY", datetime(2018, 1, 1), datetime(2024, 12, 31))
print(bars.shape, fingerprint_bars(bars)[:12])
```

## Cross-links

- `Interval` enum + `OHLCV_COLUMNS` / `PAIRS_LEG_SUFFIXES` constants
  live in `src/core/types.py` / `src/core/constants.py`.
- `data_source_registry` is the source registry used by the config
  layer; `Experiment.run` resolves it via `data_source_registry.create_from_config`.
- `Manifest.data_hash` (in `src/orchestration/manifest.py`) stores the
  fingerprint at run time; the holdout-eval workflow re-fetches and
  compares to detect vendor drift.
