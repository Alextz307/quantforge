# `config/`

Validated Pydantic YAML configs that drive every CLI under
`scripts/experiment.py`. Each top-level YAML maps 1:1 to a Pydantic
model in `src/core/`, and every model sets `extra='forbid'` so a typo
in user YAML fails at load time instead of mid-run.

## Layout

| Subdirectory | Purpose | Loaded via |
| --- | --- | --- |
| `strategies/` | One YAML per strategy `experiment run` configuration. | `load_experiment_config(path)` |
| `hpo/` | One per-strategy HPO study spec consumed by `experiment tune`. | `load_hpo_config(path)` |
| `universes/` | Reusable `UniverseProfile` files (`data:` + `validation:` blocks). Deep-merged onto a strategy YAML by the study orchestrator. See `universes/README.md`. | `load_universe_profile(path)` |
| `study/` | Top-level study specs enumerating every (strategy x universe) leg the empirical sweep evaluates. See `study/README.md`. | `load_study_spec(path)` |
| `example.yaml` | Reference `ExperimentConfig` with every field documented inline. Copy-and-edit for new runs. | `load_experiment_config` |

## Top-level YAMLs (`strategies/`)

Six YAML files, one per registered strategy:

- `adaptive_bollinger.yaml`
- `pairs_trading.yaml` - two-ticker (`tickers: [IVV, VOO]`); no
  `features:` block (pairs strategies operate on raw price columns).
- `cross_asset_momentum.yaml`
- `momentum_gatekeeper.yaml`
- `return_forecast.yaml`
- `volatility_targeting.yaml`

## Schema invariants

- **`extra='forbid'`** across every Pydantic model. A typo in a YAML
  field name (e.g. `holdoutStart` vs. `holdout_start`) fails at load.
- **Registry lookups validated at load.** Strategy / data source /
  feature pipeline names map onto the corresponding global
  `ComponentRegistry`; an unknown name raises with the available
  alternatives listed.
- **Holdout boundary set at most one way.** Exactly one of
  `validation.holdout_pct` / `validation.holdout_start` may be set.
- **Pairs cross-checks at build time.**
  `is_pairs_strategy=True` strategies require `len(tickers) == 2` and
  reject a `features:` block; single-leg strategies require
  `len(tickers) == 1`.

## On-disk layout

```
config/
    example.yaml                  # reference ExperimentConfig
    strategies/
        adaptive_bollinger.yaml
        cross_asset_momentum.yaml
        momentum_gatekeeper.yaml
        pairs_trading.yaml
        return_forecast.yaml
        volatility_targeting.yaml
    hpo/
        adaptive_bollinger.yaml
        momentum_gatekeeper.yaml
        pairs_trading.yaml
        return_forecast.yaml
        volatility_targeting.yaml
    universes/
        spy_daily_5y.yaml
        spy_daily_10y.yaml
        spy_daily_covid.yaml
        spy_daily_2008.yaml
        qqq_daily_5y.yaml
        iwm_daily_5y.yaml
        dia_daily_5y.yaml
        aapl_daily_5y.yaml
        msft_daily_5y.yaml
        jpm_daily_5y.yaml
        gld_daily_5y.yaml
        ivv_voo_daily_5y.yaml
    study/
        main_study.yaml
```

## Snippet

```bash
# Single-experiment run on the canonical example:
python -m scripts.experiment run --config config/strategies/adaptive_bollinger.yaml

# HPO study on the same strategy:
python -m scripts.experiment tune \
    --config config/strategies/adaptive_bollinger.yaml \
    --hpo-config config/hpo/adaptive_bollinger.yaml

# Pairs run on IVV/VOO (multi-ticker fetch + pairs engine dispatch):
python -m scripts.experiment run --config config/strategies/pairs_trading.yaml

# Drive the same strategy YAML across universes via --override (avoids
# committing one near-identical YAML per universe):
python -m scripts.experiment run \
    --config config/strategies/momentum_gatekeeper.yaml \
    --override 'data.tickers=[QQQ]' \
    --override 'data.start=2018-01-02'
```

## Composing configs with `--override`

Every CLI subcommand that loads a config (`run`, `tune`, `compare`)
accepts repeated `--override key.path=value` flags. The
value is parsed with `yaml.safe_load` so the surface matches the YAML
files (e.g. `[QQQ]` -> list, `false` -> bool, `2024-01-01` -> date).
Intermediate keys must already exist in the loaded YAML, so a typo like
`--override dat.tickers=[QQQ]` raises instead of being ignored. This
mechanism lets a caller compose an offline parquet data block from the
canonical `config/strategies/*.yaml` files without keeping
source-specific duplicates.

## Cross-links

- Schemas: `ExperimentConfig` (`src/core/config.py`),
  `HPOConfig` (`src/core/hpo_config.py`).
- Wiring: `src/orchestration/builder.py::build_experiment` resolves
  every name via the global registries.
