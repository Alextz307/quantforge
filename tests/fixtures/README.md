# `tests/fixtures/`

Committed test/demo fixtures. Kept small and immutable so test runs and
the `make thesis-demo` target work offline on a fresh checkout, with no
dependency on Yahoo or any other network endpoint.

## Inventory

| File | Provenance | Used by |
| --- | --- | --- |
| `SPY.parquet` | yfinance `SPY` daily, `2018-01-01` → `2024-12-31`, `auto_adjust=True`. ~1760 bars, normalized to `open/high/low/close/volume` columns with a `DatetimeIndex`. ~93 KB. The bare-ticker filename matches the `ParquetSource` lookup convention (`{ticker}.parquet`); the date range lives in this README, not in the filename. | `make thesis-demo` (via `ParquetSource` pointed at this directory). |
| `google_benchmark_sample.json` | Captured output of `quant_bench --benchmark_format=json`. | Benchmark runner unit tests. |

## Regenerating `SPY.parquet`

```bash
python -c "
from pathlib import Path
import yfinance as yf
from src.data.normalizer import DataNormalizer
from src.data.validator import validate_bars

raw = yf.download('SPY', start='2018-01-01', end='2024-12-31', interval='1d', progress=False, auto_adjust=True)
if raw.columns.nlevels > 1:
    raw.columns = raw.columns.get_level_values(0)
df = DataNormalizer('yfinance').normalize(raw)
validate_bars(df)
df.to_parquet(Path('tests/fixtures/SPY.parquet'))
"
```

The flatten step is required because `yf.download` returns a multi-index
column header even for a single ticker; `YFinanceSource.fetch_raw` does
not currently flatten it, so the normalizer rejects the frame. Doing the
flatten + normalize once at write time keeps the on-disk file in the
canonical OHLCV schema, which is what `ParquetSource` reads back.

`auto_adjust=True` is intentional — adjusted closes already fold splits
and dividends into the price series, so downstream return / strategy
math matches what a long-only buy-and-hold backtest would produce.
