# `scripts/`

User-facing CLIs (experiment + benchmark) and stdlib-only drift guards
that protect invariants between two on-disk sources of truth (CI vs.
pyproject, leaf-keys vs. strategy ctors).

## Public surface

| Script | Role |
| --- | --- |
| `experiment.py` (`make experiment`) | Click group with `run`, `tune`, `compare`, `holdout-eval`, `study`, `clean` subcommands. Drives the orchestration layer end to end. |
| `benchmark.py` (`make bench`) | Click group with `run`, `compare`, `latex`, `history`, `show-baseline` over `BenchmarkRunner` / `BenchmarkStore`. |
| `check_ci_deps.py` | Drift guard: every runtime dep in `pyproject.toml` appears in CI's `python-test` pip install line; every `types-*` / `*-stubs` dev dep appears in CI's `lint-and-typecheck` pip install line. Runs in CI as an early lint step. (The `webapp` + `webapp-frontend` jobs use `pip install -e ".[webapp]"` so their installs cannot drift from `[webapp]` extras.) |
| `check_readme_test_counts.py` | Drift guard: README's "**N Python tests** (+M opt-in skips), **K C++ tests**" phrase agrees with `pytest --collect-only` and `ctest -N`. Pass `--fix` to rewrite the README in place from the live numbers (runs the suite to split passed vs skipped). C++ check downgraded to a notice when `cpp/build/` is absent. |
| `dump_openapi.py` (`make webapp-openapi-snapshot`) | Boot FastAPI, write its OpenAPI 3.1 spec to `webapp/frontend/openapi.snapshot.json` (the committed contract that `npm run gen:api` reads). |
| `check_openapi_snapshot.py` (`make webapp-check-openapi-snapshot`) | Drift guard: re-build the OpenAPI spec, fail if it diverges from the committed snapshot. Tells the developer to rerun `make webapp-openapi-snapshot` and commit. |
| `check_webapp_schema_mirror.py` (`make webapp-check-schema-mirror`) | Drift guard: extract every Pydantic write-DTO mirrored as a zod schema (`LoginRequest`, `UserCreate`) into `webapp/frontend/schema-mirror.snapshot.json`; pair vitest test asserts the zod schema agrees on field names, types, min/max constraints. `--write` regenerates the snapshot. |
| `regen_stubs.py` (`make stubs`) | Regenerate `quant_engine` pybind11 stubs and apply ruff lint / format so the checked-in artefact passes `make lint`. |
| `regen_spy_fixture.py` | Refetch + normalize + validate `tests/fixtures/SPY.parquet` (`SPY` daily, `2018-01-01` → `2024-12-31`, `auto_adjust=True`). Run when the committed fixture goes stale. |

## Layout

| File | Role |
| --- | --- |
| `experiment.py` | `experiment run / tune / compare / holdout-eval / study / clean`. |
| `study.py` | `experiment study run / report` — sub-group registered under `experiment.py`'s `cli`. |
| `benchmark.py` | `benchmark run / compare / latex / history / show-baseline`. |
| `check_ci_deps.py` | Stdlib-only (no PyYAML) so it runs in CI before deps install. |
| `check_readme_test_counts.py` | Stdlib-only; runs after `pip install -e .` so it can spawn `pytest --collect-only`. |
| `dump_openapi.py` | Lazy-imports `webapp.backend.app.main` so `--out` consumers without webapp deps can still load the module's `DEFAULT_SNAPSHOT_PATH` constant. |
| `check_openapi_snapshot.py` | Reuses `dump_openapi.build_openapi_spec()` for the live spec; lazy webapp import keeps `diff_against_snapshot` testable without fastapi. |
| `check_webapp_schema_mirror.py` | Walks `model_fields` on Pydantic v2 models; `--write` regenerates the snapshot, default mode diffs. Frontend pair: `webapp/frontend/tests/lib/schemas/mirror.test.ts`. |
| `regen_stubs.py` | Wraps `pybind11-stubgen` + `ruff check --fix` + `ruff format`. |
| `regen_spy_fixture.py` | Wraps `yfinance.download` + `DataNormalizer` + `validate_bars`; not run in CI. |

## `experiment` subcommands

| Command | Output | Notes |
| --- | --- | --- |
| `run --config <yaml>` | `experiment_results/runs/<experiment_id>/` | Walk-forward → `manifest.json` + `fold_results.jsonl` + `metrics.json` + optional `strategy_state/`. |
| `tune --config <exp.yaml> --hpo-config <hpo.yaml>` | `experiment_results/hpo/<study>/` | Optuna study via `StrategyTuner`; resumable. |
| `compare --config <yaml> ... --out <name> [--reuse-runs <dirs>]` | `experiment_results/comparisons/<out>/` | N strategies on aligned data, ranked + pairwise-bootstrapped. With `--reuse-runs <a,b,...>` (one path per `--config` in matching order) the per-strategy walk-forward step is skipped and prior fold artifacts feed ranking + bootstrap. |
| `holdout-eval --run-dir <path> \| --hpo-best <path>` | `experiment_results/holdout_evals/<out>/` | Refit on full dev, evaluate once on the reserved holdout — the honest one-shot OOS number. Sources are mutually exclusive; manifest cross-checks `holdout_start` + `data_hash` before fitting. |
| `study run --spec <yaml>` | `<store_root>/<spec.output_dir>/` | Cross-strategy × cross-universe sweep: tune → run → holdout-eval per leg, then per-universe cross-strategy compare. Resumable via `study_state.json`; per-leg failures isolated. |
| `study report --study-dir <path>` | `<study_dir>/{tables,plots,manifest.json}` | Walk a completed study tree; emit master / per-universe / holdout rankings (`.tex`+`.csv`), strategy×universe heatmap, dev-vs-holdout scatter, and per-universe equity-overlay / per-leg holdout-equity copies. Read-only with respect to the per-leg tree. |
| `clean [--store-root experiment_results] [--apply] [--keep <name>]` | `<store-root>/` | Remove ephemeral child directories under the store root (default: `experiment_results/`). Always preserves `thesis_demo/`; refuses to delete any directory containing git-tracked files. Default = dry-run; pass `--apply` to delete. |

Multi-ticker pairs configs route through `experiment run` with no
special flag — the builder dispatches on the strategy class's
`is_pairs_strategy` capability flag.

### Shared flags across config-loading subcommands

| Flag | Applies to | Role |
| --- | --- | --- |
| `--override key.path=value` (repeatable) | `run`, `tune`, `compare` | Dotted-path mutation of the loaded YAML before pydantic re-validation. Value parsed with `yaml.safe_load`; intermediate keys must already exist (typo guard). On `compare` the same set applies to every `--config`. |
| `--publish-label <slug>` | `run`, `compare`, `holdout-eval` | Stable LaTeX `\caption` + `\label` slug for the emitted tables. When unset the legacy volatile id (`experiment_id` / `out_name` / `source_id`) is used; when set the slug overrides it so thesis-prose `\ref` stays valid across reruns. Slug regex: starts with a letter, then letters / digits / `_` / `-` / `:`. |

### Per-invocation log files (`cli_logs/`)

Every persistent CLI subcommand (`run`, `tune`, `compare`,
`holdout-eval`, `study run`, `study report`) tees its full Python-logger
stream to a timestamped file under
`<store_root>/cli_logs/<command>_<UTC_YYYYMMDD_HHMMSS>_<pid>.log` (or
`<study_dir>/cli_logs/...` for `study report`). The first line of every
invocation echoes the resolved log path, e.g.
``running study from spec ... → log: experiment_results/cli_logs/study_run_20260504_193000_12345.log``.
The file contains the same lines `tail -f` would show on stdout (same
formatter), so a dropped terminal during a multi-day sweep doesn't lose
the diagnostic trail. Utility commands (`clean`) skip the file handler
since they're fast and produce no logger output worth persisting.

## Drift-guard invariants

- **CI ↔ pyproject.** A new runtime / type-stub dep landing in
  `pyproject.toml` without a matching update to `.github/workflows/ci.yml`
  fails `check_ci_deps.py` in the same PR. The `webapp` + `webapp-frontend`
  jobs use `pip install -e ".[webapp]"`, so their installs cannot drift from
  `[webapp]` extras and need no separate guard.
- **README test counts ↔ runners.** `check_readme_test_counts.py`
  re-collects the pytest + ctest counts and compares them against the
  README prose; running tests but forgetting to update the README fails
  CI lint.
- **OpenAPI snapshot ↔ live FastAPI app.** `check_openapi_snapshot.py`
  re-dumps the spec and fails if it diverges from
  `webapp/frontend/openapi.snapshot.json`. The committed snapshot is
  what `npm run gen:api` reads to emit `src/api/generated/schema.ts`.
- **Pydantic write-DTOs ↔ zod form schemas.**
  `check_webapp_schema_mirror.py` extracts the canonical Pydantic shape
  into `webapp/frontend/schema-mirror.snapshot.json`; the paired vitest
  test asserts each zod schema's `.shape` agrees on field names, types,
  and min/max constraints.

`check_ci_deps.py` and `check_readme_test_counts.py` are stdlib-only;
`check_openapi_snapshot.py` and `check_webapp_schema_mirror.py` need
`[webapp]` deps installed.

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
  `src/orchestration/builder.py`, `src/orchestration/comparison.py`, and
  `src/orchestration/holdout_eval.py`.
- `benchmark.py` wraps `src/benchmarking/`.
- The Makefile binds these CLIs to `make experiment` / `make bench` /
  `make stubs`.
