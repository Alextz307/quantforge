/**
 * Pre-fill content for the "New universe spec" Monaco editor.
 *
 * A commented self-contained ``UniverseProfile`` covering the canonical
 * fields. Mirrors the descriptions on ``UniverseProfile`` in
 * ``src/core/config.py`` — keep both in sync so the editor surface tracks
 * the schema.
 */
export const UNIVERSE_SPEC_SKELETON = `# Universe spec — names the tickers, interval, and time window for a
# strategy run. Slug is set in the form above; this file holds the spec.
data:
  source: yfinance          # required — data source name (yfinance / parquet / ...)
  tickers: [SPY]            # required — non-empty list of tickers
  start: 2020-01-01         # required — inclusive start date (YYYY-MM-DD)
  end: 2024-12-31           # required — inclusive end date (YYYY-MM-DD)
  interval: daily           # required — one of: daily, hour, ...

validation:
  holdout_pct: 0.20         # optional — fraction reserved as the holdout tail
`;
