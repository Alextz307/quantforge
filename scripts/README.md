# `scripts/`

User-facing CLIs (experiment + benchmark) and stdlib-only drift guards
that protect invariants between two on-disk sources of truth (CI vs.
pyproject, leaf-keys vs. strategy ctors).

## Public surface

| Script | Role |
| --- | --- |
| `experiment.py` (`make experiment`) | Click group with `run`, `train-model`, `list-models`, `tune`, `compare`, `regime` subcommands. Drives the orchestration layer end to end. |
| `benchmark.py` (`make bench`) | Click group with `run`, `compare`, `latex`, `history`, `show-baseline` over `BenchmarkRunner` / `BenchmarkStore`. |
| `check_ci_deps.py` | Drift guard: every runtime dep in `pyproject.toml` appears in CI's `python-test` pip install line; every `types-*` / `*-stubs` dev dep appears in CI's `lint-and-typecheck` pip install line. Runs in CI as an early lint step. |
| `check_leaf_keys_consistent.py` | Drift guard: `_LEAF_KEY_OWNED_PARAMS` (config layer) vs. each strategy's `_leaf_keys` (ctor layer). |
| `regen_stubs.py` (`make stubs`) | Regenerate `quant_engine` pybind11 stubs and apply ruff lint / format so the checked-in artefact passes `make lint`. |

## Layout

| File | Role |
| --- | --- |
| `experiment.py` | `experiment run / train-model / list-models / tune / compare / regime`. |
| `benchmark.py` | `benchmark run / compare / latex / history / show-baseline`. |
| `check_ci_deps.py` | Stdlib-only (no PyYAML) so it runs in CI before deps install. |
| `check_leaf_keys_consistent.py` | Imports `src.strategies` so the registry is populated. |
| `regen_stubs.py` | Wraps `pybind11-stubgen` + `ruff check --fix` + `ruff format`. |

## `experiment` subcommands

| Command | Output | Notes |
| --- | --- | --- |
| `run --config <yaml>` | `experiment_results/runs/<experiment_id>/` | Walk-forward → `manifest.json` + `fold_results.jsonl` + `metrics.json` + optional `strategy_state/`. |
| `train-model --config <yaml>` | `experiment_results/models/<name>/` | Standalone leaf fit (HybridReturn, HybridVolatility, DirectionalClassifier) for `pretrained_leaves` injection. |
| `list-models` | stdout | Enumerates saved model artifacts. |
| `tune --config <exp.yaml> --hpo-config <hpo.yaml>` | `experiment_results/hpo/<study>/` | Optuna study via `StrategyTuner`; resumable. |
| `compare --config <yaml> ... --out <name>` | `experiment_results/comparisons/<out>/` | N strategies on aligned data, ranked + pairwise-bootstrapped. |
| `regime --exp-id <id> --regime-config <yaml> --out <name>` | `experiment_results/regime_reports/<out>/` | Re-tag a persisted run by regime detector + emit per-regime stats. |

Multi-ticker pairs configs route through `experiment run` with no
special flag — the builder dispatches on the strategy class's
`is_pairs_strategy` capability flag.

## Drift-guard invariants

- **CI ↔ pyproject.** A new runtime / type-stub dep landing in
  `pyproject.toml` without a matching update to `.github/workflows/ci.yml`
  fails `check_ci_deps.py` in the same PR.
- **Config-layer leaf keys ↔ strategy ctor leaf keys.** The two sets
  are checked for equality per strategy by `check_leaf_keys_consistent.py`.

Both guards are stdlib-only so they run in CI's lint job before
project deps install.

## Snippet

```bash
# Single-experiment run on a standalone strategy YAML
make experiment ARGS="run --config config/strategies/adaptive_bollinger.yaml"

# HPO study
make experiment ARGS="tune --config config/strategies/adaptive_bollinger.yaml \
                            --hpo-config config/hpo/adaptive_bollinger.yaml"

# Cross-strategy comparison
make experiment ARGS="compare \
    --config config/strategies/adaptive_bollinger.yaml \
    --config config/strategies/momentum_gatekeeper.yaml \
    --out demo_comparison"
```

## Cross-links

- `experiment.py` is a thin click wrapper over
  `src/orchestration/builder.py`, `src/orchestration/comparison.py`,
  `src/orchestration/regime_run.py`, and
  `src/orchestration/standalone_training.py`.
- `benchmark.py` wraps `src/benchmarking/`.
- The Makefile binds these CLIs to `make experiment` / `make bench` /
  `make stubs`.
