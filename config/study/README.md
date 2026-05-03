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

## Cross-links

- Schemas: `StudySpec`, `StudyLeg` (`src/core/config.py`).
- Universe profiles consumed by every leg: `config/universes/`.
- Per-strategy and per-HPO YAMLs referenced by each leg:
  `config/strategies/`, `config/hpo/`.
