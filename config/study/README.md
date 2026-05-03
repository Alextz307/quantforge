# `config/study/`

Top-level empirical-study specs. A study spec enumerates every
(strategy × universe) leg the study orchestrator should evaluate;
incompatible combinations (e.g. AAPL × PairsTrading) are filtered at
spec-write time, not at run time.

## Schema

Validated by `StudySpec` / `StudyLeg` (`src/core/config.py`):

```yaml
name: main_study
description: <free text>                 # optional
seed: 42
output_dir: experiment_results/studies/main

legs:
  - strategy: AdaptiveBollinger          # registry display name
    strategy_config: config/strategies/adaptive_bollinger.yaml
    hpo_config: config/hpo/adaptive_bollinger.yaml
    universes:                           # bare names; resolved against config/universes/
      - spy_daily_5y
      - spy_daily_10y
      ...
```

See `StudySpec` / `StudyLeg` (`src/core/config.py`) for the full set of
invariants. Path fields are typed `Path` but not checked for existence at
schema validation time — the orchestrator and the test suite open them.

## Inventory

`main_study.yaml` — five strategies (`AdaptiveBollinger`, `PairsTrading`,
`MomentumGatekeeper`, `VolatilityTargeting`, `ReturnForecast`) over the
universes in `config/universes/`, with each strategy's universe list
filtered for compatibility (e.g. only the GLD/SLV pair for
`PairsTrading`).

## Loading a spec

```python
from src.core.config import load_study_spec

spec = load_study_spec("config/study/main_study.yaml")
for leg in spec.legs:
    for universe_name in leg.universes:
        ...  # orchestrator composes leg.strategy_config + universes/<name>.yaml
```

## Driving the spec — `experiment study`

Two subcommands consume a study spec end-to-end:

```bash
# Train every (universe, leaf_key) artifact needed by ML-bearing legs.
# Skip-on-existing makes the command resumable across transient failures.
python -m scripts.experiment study train-leaves \
    --spec config/study/main_study.yaml

# Drive the sweep: tune -> run -> regime -> holdout-eval per leg, then
# per-universe cross-strategy compare. Resumable via study_state.json.
python -m scripts.experiment study run \
    --spec config/study/main_study.yaml \
    --regime-config config/regimes/vol_quintile.yaml
```

Per-leg outputs land under `<store_root>/<spec.output_dir>/`:

| Path | Owner |
| --- | --- |
| `study_state.json` | Orchestrator-owned resume state (atomic write). |
| `spec.yaml` | Frozen copy of the input spec (provenance). |
| `hpo/<leg_id>/` | Standard `tune` output. |
| `runs/<run_experiment_id>/` | `run` materialised from `best_config.yaml`. |
| `regime_reports/<leg_id>/` | Per-leg regime split (when `--regime-config` set). |
| `holdout_evals/<leg_id>/` | Honest OOS (when `validation.holdout_pct > 0`). |
| `comparisons/<universe>/` | Cross-strategy compare (skipped for single-strategy universes). |

`leg_id = "<strategy>__<universe>"`. The auto-generated
`run_experiment_id` is recorded on each `LegState` so cross-strategy
compare can resolve the run dir without re-walking `runs/`.

## Pretrained-leaf artifact convention

ML-bearing legs (those whose strategy YAML declares
`pretrained_leaves:`) consume artifacts at the conventional path

```
<store_root>/models/{universe}_{leaf_key}/
```

`train-leaves` builds these artifacts by composing each leaf-type
template (`config/models/spy_directional_classifier.yaml`,
`spy_hybrid_return.yaml`, `spy_hybrid_volatility.yaml`) with the
universe's `data:` block and the artifact name. Artifacts live under
`<store_root>/models/` (NOT `<study_dir>/`) so a HybridReturn trained
on SPY 2020-2024 is reusable across many studies that share the same
universe definition.

## Cross-links

- Schemas: `StudySpec`, `StudyLeg`, `UniverseProfile` (`src/core/config.py`).
- Orchestrator: `src/orchestration/study.py`,
  `src/orchestration/study_state.py`.
- CLI adapter: `scripts/study.py` (registered under
  `scripts/experiment.py`'s `study` group).
- Universe profiles consumed by every leg: `config/universes/`.
- Per-strategy and per-HPO YAMLs referenced by each leg:
  `config/strategies/`, `config/hpo/`.
- Leaf-type templates for `train-leaves`: `config/models/`.
- Default regime config for the sweep: `config/regimes/vol_quintile.yaml`.
