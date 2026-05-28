# `tests/fixtures/`

Committed test/demo fixtures. Kept small and immutable so test runs and
the `make thesis-demo` target work offline on a fresh checkout, with no
dependency on Yahoo or any other network endpoint.

## Inventory

| File | Provenance | Used by |
| --- | --- | --- |
| `SPY.parquet` | yfinance `SPY` daily, `2018-01-01` → `2024-12-31`, `auto_adjust=True`. ~1760 bars, normalized to `open/high/low/close/volume` columns with a `DatetimeIndex`. ~93 KB. The bare-ticker filename matches the `ParquetSource` lookup convention (`{ticker}.parquet`); the date range lives in this README, not in the filename. | `make thesis-demo` (via `ParquetSource` pointed at this directory). |

## Regenerating `SPY.parquet`

```bash
python scripts/regen_spy_fixture.py
```
