# `config/`

Validated Pydantic YAML configs that drive every CLI under
`scripts/experiment.py`. Each top-level YAML maps 1:1 to a Pydantic
model in `src/core/`; `extra='forbid'` is enforced everywhere so silent
typos in user YAML fail at load time, not mid-run.

## Layout

| Subdirectory | Purpose | Loaded via |
| --- | --- | --- |
| `strategies/` | One YAML per strategy `experiment run` configuration. Two variants where pretrained-leaf injection is interesting (`*_pretrained.yaml`). | `load_experiment_config(path)` |
| `hpo/` | One per-strategy HPO study spec consumed by `experiment tune`. | `load_hpo_config(path)` |
| `regimes/` | Detector specs consumed by `experiment regime`. | `load_regime_config(path)` |
| `models/` | Standalone leaf-training configs consumed by `experiment train-model` (output → `experiment_results/models/<name>/`). | `load_standalone_model_config(path)` |
| `universes/` | Reusable `UniverseProfile` files (`data:` + `validation:` blocks). Deep-merged onto a strategy YAML by the study orchestrator. See `universes/README.md`. | `load_universe_profile(path)` |
| `study/` | Top-level study specs enumerating every (strategy × universe) leg the empirical sweep evaluates. See `study/README.md`. | `load_study_spec(path)` |
| `example.yaml` | Reference `ExperimentConfig` with every field documented inline. Copy-and-edit for new runs. | `load_experiment_config` |

## Top-level YAMLs (`strategies/`)

Seven YAML files across five registered strategies (two pretrained-leaf
variants for the strategies where injection is interesting):

- `adaptive_bollinger.yaml`
- `pairs_trading.yaml` — two-ticker (`tickers: [GLD, SLV]`); no
  `features:` block (pairs strategies operate on raw price columns).
- `momentum_gatekeeper.yaml` — full HPO from scratch.
- `momentum_gatekeeper_pretrained.yaml` — injects a frozen
  `DirectionalClassifier` artifact via `pretrained_leaves`.
- `return_forecast.yaml` — full HPO.
- `return_forecast_pretrained.yaml` — injects a frozen
  `HybridReturnModel` artifact.
- `volatility_targeting.yaml` — full HPO.

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
- **Pretrained-leaf collisions rejected.** Keys the pinned leaf owns
  (per `_LEAF_KEY_OWNED_PARAMS` in `src/core/config.py`) MUST NOT
  appear in `strategy.params` — the artifact wins, surfacing user
  intent collisions early.

## On-disk layout

```
config/
    example.yaml                  # reference ExperimentConfig
    strategies/
        adaptive_bollinger.yaml
        pairs_trading.yaml
        momentum_gatekeeper.yaml
        momentum_gatekeeper_pretrained.yaml
        return_forecast.yaml
        return_forecast_pretrained.yaml
        volatility_targeting.yaml
    hpo/
        adaptive_bollinger.yaml
        pairs_trading.yaml
        momentum_gatekeeper.yaml
        return_forecast.yaml
        volatility_targeting.yaml
    regimes/
        bull_bear_200ma.yaml
        covid_split.yaml
        vol_quintile.yaml
    models/
        spy_directional_classifier.yaml
        spy_hybrid_return.yaml
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
        gld_slv_daily_5y.yaml
        eurusd_daily_5y.yaml
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

# Pairs run on GLD/SLV (multi-ticker fetch + pairs engine dispatch):
python -m scripts.experiment run --config config/strategies/pairs_trading.yaml

# Drive the same strategy YAML across universes via --override (avoids
# committing one near-identical YAML per universe):
python -m scripts.experiment run \
    --config config/strategies/momentum_gatekeeper.yaml \
    --override 'data.tickers=[QQQ]' \
    --override 'data.start=2018-01-02'
```

## Composing configs with `--override`

Every CLI subcommand that loads a config (`run`, `train-model`, `tune`,
`compare`) accepts repeated `--override key.path=value` flags. The
value is parsed with `yaml.safe_load` so the surface matches the YAML
files (e.g. `[QQQ]` → list, `false` → bool, `2024-01-01` → date).
Intermediate keys must already exist in the loaded YAML — typos like
`--override dat.tickers=[QQQ]` raise instead of silently no-op'ing. The
`make thesis-demo` target uses this mechanism to compose the offline
parquet data block from the canonical `config/strategies/*.yaml` files
without keeping demo-specific duplicates.

## Cross-links

- Schemas: `ExperimentConfig` (`src/core/config.py`),
  `HPOConfig` (`src/core/hpo_config.py`),
  `RegimeConfig` (`src/core/regime_config.py`),
  `StandaloneModelConfig` (`src/core/config.py`).
- Wiring: `src/orchestration/builder.py::build_experiment` resolves
  every name via the global registries.
- Pretrained-leaf workflow lives in
  `src/orchestration/standalone_training.py` and
  `src/orchestration/pretrained_leaves.py`.
