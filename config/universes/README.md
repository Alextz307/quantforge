# `config/universes/`

Reusable universe profiles consumed by the study orchestrator. Each
profile is a self-contained `data:` + `validation:` (+ optional
`strategy_params:`) block that the orchestrator deep-merges onto a
strategy YAML at run time. Universe profiles win on conflicts so the
dev/holdout boundary stays pinned per universe across every strategy that
runs against it.

## Schema

Validated by `UniverseProfile` (`src/core/config.py`):

```yaml
data:                                        # required, mirrors DataConfig
  source: yfinance                           # registry name or {name, params}
  tickers: [SPY]                             # 1 single-asset, 2 pairs, N for a basket
  start: 2021-01-01
  end: 2025-12-31
  interval: daily
strategy_params:                             # optional, overrides strategy.params per-key
  primary_ticker: SPY                        # e.g. CrossAssetMomentum basket wiring
  feature_tickers: [QQQ, IWM, GLD, TLT]
validation:                                  # optional, mirrors ValidationConfig
  holdout_pct: 0.20                          # most profiles only pin holdout
```

Walk-forward knobs (`n_splits`, `test_size`, `gap`, `expanding`,
`snap_to_day`) come from the strategy YAML on merge. A universe profile
fixes only the data fetch and the holdout. The optional `strategy_params`
block overrides `strategy.params` per key, so one strategy config runs
different parameterizations across universes. CrossAssetMomentum uses it
to wire each basket's `primary_ticker` / `feature_tickers`.

## Inventory

### Single-asset matrix

Eleven liquid names, each at two lookbacks
(`<ticker>_daily_5y.yaml`, `<ticker>_daily_10y.yaml`):

- Tickers: SPY, QQQ, GLD, AAPL, MSFT, GOOGL, META, AMZN, TSLA, KO, JPM.
- 5y: 2021-2025, `holdout_pct: 0.20`.
- 10y: 2016-2025, `holdout_pct: 0.15` (smaller holdout for the longer history).

### CrossAssetMomentum baskets

Multi-ticker profiles that also carry a `strategy_params` override pinning
the basket's primary and feature tickers (one strategy config, two
baskets), each at 5y/10y:

| Profile | Tickers | Primary | Features |
| --- | --- | --- | --- |
| `megacap_tech_daily_{5y,10y}.yaml` | AAPL, MSFT, GOOGL, META, AMZN | AAPL | MSFT, GOOGL, META, AMZN |
| `cross_asset_daily_{5y,10y}.yaml` | SPY, QQQ, IWM, GLD, TLT | SPY | QQQ, IWM, GLD, TLT |

Basket members were picked for shared structure without redundancy: the
tech names co-move 0.5-0.7 (one factor, no near-duplicates), and the
cross-asset basket pairs the strongest lead-lag peers of SPY (QQQ, IWM,
and long bonds via TLT) with gold as an orthogonal diversifier.

### Pair

| Profile | Tickers | Windows |
| --- | --- | --- |
| `ivv_voo_daily_{5y,10y}.yaml` | IVV, VOO | 2021-2025 / 2016-2025 |

## Loading a profile

```python
from src.core.config import load_universe_profile

profile = load_universe_profile("config/universes/spy_daily_5y.yaml")
profile.data.tickers       # ['SPY']
profile.validation.holdout_pct  # 0.20
```

The study orchestrator loads the profile, deep-merges it into the strategy
YAML, and feeds the merged dict through `ExperimentConfig.model_validate`.

## Cross-links

- Schema: `UniverseProfile` (`src/core/config.py`).
- Study spec that enumerates which strategy runs on which universes:
  `config/study/main_study.yaml`.
- Strategy YAMLs that get composed with a universe profile:
  `config/strategies/`.
