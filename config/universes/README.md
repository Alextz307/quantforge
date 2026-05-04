# `config/universes/`

Reusable universe profiles consumed by the study orchestrator. Each
profile is a self-contained `data:` + `validation:` block that the
orchestrator deep-merges onto a strategy YAML at run time. Universe
profiles win on conflicts so the dev/holdout boundary stays pinned per
universe across every strategy that runs against it.

## Schema

Validated by `UniverseProfile` (`src/core/config.py`):

```yaml
data:                                        # required, mirrors DataConfig
  source: yfinance                           # registry name or {name, params}
  tickers: [SPY]                             # 1 ticker single-asset, 2 for pairs
  start: 2020-01-01
  end: 2024-12-31
  interval: daily
validation:                                  # optional, mirrors ValidationConfig
  holdout_pct: 0.20                          # most profiles only pin holdout
```

Walk-forward knobs (`n_splits`, `test_size`, `gap`, `expanding`,
`snap_to_day`) come from the strategy YAML on merge — universe
profiles intentionally only fix the data fetch and the holdout.

## Inventory

Twelve profiles cover index ETFs, single large-caps, commodity, and a
GFC stress window:

| File | Tickers | Range | Notes |
| --- | --- | --- | --- |
| `spy_daily_5y.yaml` | SPY | 2020–2024 | broad-market baseline |
| `spy_daily_10y.yaml` | SPY | 2014–2024 | spans two cycles |
| `spy_daily_covid.yaml` | SPY | 2019–2022 | pre-COVID + crash + recovery |
| `qqq_daily_5y.yaml` | QQQ | 2020–2024 | tech-tilt index |
| `iwm_daily_5y.yaml` | IWM | 2020–2024 | small-cap index |
| `dia_daily_5y.yaml` | DIA | 2020–2024 | blue-chip index |
| `aapl_daily_5y.yaml` | AAPL | 2020–2024 | single large-cap |
| `msft_daily_5y.yaml` | MSFT | 2020–2024 | single large-cap |
| `jpm_daily_5y.yaml` | JPM | 2020–2024 | financial-sector single name |
| `gld_daily_5y.yaml` | GLD | 2020–2024 | commodity ETF |
| `gld_slv_daily_5y.yaml` | GLD, SLV | 2020–2024 | precious-metals pair |
| `spy_daily_2008.yaml` | SPY | 2007–2010 | GFC stress; `holdout_pct: 0.0` |

`spy_daily_2008` deliberately disables holdout — the regime IS the test;
slicing off the last 20% would discard most of the recovery window.

## Loading a profile

```python
from src.core.config import load_universe_profile

profile = load_universe_profile("config/universes/spy_daily_5y.yaml")
profile.data.tickers       # ['SPY']
profile.validation.holdout_pct  # 0.20
```

The study orchestrator loads the profile, deep-merges it into the
strategy YAML, and feeds the merged dict through
`ExperimentConfig.model_validate`.

## Cross-links

- Schema: `UniverseProfile` (`src/core/config.py`).
- Study spec that enumerates which strategy runs on which universes:
  `config/study/main_study.yaml`.
- Strategy YAMLs that get composed with a universe profile:
  `config/strategies/`.
