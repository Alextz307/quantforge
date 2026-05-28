# Webapp

A FastAPI + React webapp that puts a configurable runner and an artifact
viewer in front of the existing quant framework. The backend never
re-implements strategy logic — it spawns the existing CLI as
subprocesses and reads the predictable artifact tree under
`experiment_results/`, so anything that runs from the UI is bit-identical
to the same command issued from `bash`.

## Layout

| Path | Role |
| --- | --- |
| `backend/app/` | FastAPI application: `api/` routers, `core/` (settings, lifespan, version), `services/`, `infrastructure/`, `schemas/`. |
| `backend/tests/` | `unit/` (service-layer + router-level tests via `TestClient`) and `integration/` (subprocess-driven). |
| `frontend/` | React + TypeScript + Vite SPA. `src/{api,components,features,hooks,lib,pages}` — type-safe API client generated from the backend's OpenAPI spec via `openapi-typescript`. |
| `frontend/openapi.snapshot.json` | Committed contract between backend and frontend; regenerated via `python -m scripts.dump_openapi`. |
| `data/` | Local SQLite store for users + jobs. Gitignored. |

## Public surface (so far)

Public:
- `GET /api/health` — `{status, version}` liveness probe.
- `GET /openapi.json`, `GET /docs` — OpenAPI 3.1 schema + Swagger UI.

Auth:
- `POST /api/auth/login` — `{username, password}` → `{user}` + sets HttpOnly session cookie. Rate-limited (5 / 15 minutes / IP).
- `POST /api/auth/logout` — clears the session cookie.
- `GET /api/auth/me` — returns the current authenticated user.

Admin (role=admin):
- `GET /api/users`, `POST /api/users`, `DELETE /api/users/{id}` — user CRUD (soft delete).

Read-only artifact API (auth-gated, all `GET`):
- `/api/strategies`, `/api/strategies/{name}/schema`, `/api/models` — registry introspection + per-strategy ctor schema for the Configure form.
- `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/folds`, `/api/runs/{id}/plots/{plot_name}` — persisted runs.
- `/api/comparisons`, `/api/comparisons/{name}`, `/api/comparisons/{name}/plots/{plot_name}` — cross-strategy comparisons.
- `/api/holdout-evals`, `/api/holdout-evals/{name}`, `/api/holdout-evals/{name}/plots/{plot_name}` — holdout evaluations.
- `/api/studies`, `/api/studies/{name}` — multi-leg study state + completion progress.
- `/api/hpo`, `/api/hpo/{name}`, `/api/hpo/{name}/trials?after_trial=N` — HPO study summaries + Optuna trial feeds.
- `/api/hpo/{name}/param-importance` — fANOVA-style relative importance per hyperparameter, computed on demand from the study's Optuna SQLite. Returns `{importance, message}` with `importance={}` plus a human-readable `message` while the study has too few completed trials, no DB yet, or a degenerate search space — the endpoint stays 200 across the live lifecycle.

Deployments (auth-gated; per-user scope, admins may pass `?all=1`):
- `GET /api/deployments`, `POST /api/deployments`, `GET /api/deployments/{id}`, `PATCH /api/deployments/{id}`, `DELETE /api/deployments/{id}` — saved pointers to a previously trained run or HPO best trial that accumulate a daily signal log. Create with `{source_kind: "run"|"hpo", source_id, name?, warmup_bars?}`; `name` and `warmup_bars` auto-derive from the source's manifest + strategy when omitted. The DB row carries denormalised `ticker / strategy_name / interval / train_end` columns read from the source at create time. Pairs and non-daily sources are refused upfront with 422.
- `GET /api/deployments/{id}/signals?limit=N` — append-only signal log read from `signals.jsonl`. `submitted_at` (wall-clock predict time) and `bar_ts` (the bar the signal is for) are tracked separately.
- `POST /api/deployments/{id}/predict-if-stale` — synchronous. Recalls today's signal from `signals.jsonl` if its `bar_ts` is at or beyond the vendor's latest available bar; otherwise invokes the framework's `predict()` inside FastAPI's threadpool, appends a new row (idempotent on `bar_ts`), and returns it. Surfaces the framework's `LeakageError` / `WarmupInsufficientError` as 422.

Configs (auth-gated):
- `GET /api/configs/{kind}` — list `*.yaml` under `config/<kind>/` (kinds: experiment / universe / strategy / hpo / study / model).
- `GET /api/configs/{kind}/{name}` — raw + parsed body for a single config.
- `POST /api/configs/validate` — validate a payload against the matching Pydantic model; returns `{valid, errors[]}` with structured `loc/msg/type` items.

Jobs (auth-gated; per-user scope, admins may pass `?all=1`):
- `POST /api/jobs` — submit a `JobSubmission`. The `kind` discriminator selects the payload:
  - `kind=run` + `config_payload` — spawn `experiment run`.
  - `kind=tune` + `config_payload` + `hpo_payload` — spawn `experiment tune`.
  - `kind=compare` + `compare_payload` — spawn `experiment compare` in `--reuse-runs` mode over 2–8 existing run dirs.
  - `kind=holdout` + `holdout_payload` — spawn `experiment holdout-eval` against a run (with `holdout_start`) or an HPO study (with `best_config.yaml`).
  Validates upfront and returns 422 with the same `loc/msg/type` shape on bad payloads.
- `GET /api/jobs`, `GET /api/jobs/{id}`, `DELETE /api/jobs/{id}` — list / fetch / cancel.
- `GET /api/jobs/{id}/log` — full log file (text response).

WebSockets (auth-gated):
- `WS /api/jobs/{id}/stream` — multiplexed log + status frames (`{type:"log",line}` / `{type:"status",status,exit_code,experiment_id}`); auto-closes on terminal status.
- `WS /api/hpo/{name}/stream?after_trial=N` — `{type:"trial",trial}` frames as Optuna writes them. The handshake replays existing trials in order before forwarding live broker events; the SPA pairs this with the polled `/api/hpo/{name}/param-importance` endpoint for the live HPO monitor on `/hpo/:name`.

## First-run setup

```bash
# 1. Generate a session secret (≥32 chars). Set it via .env or shell:
export WEBAPP_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"

# 2. Create your first admin account (prompts for password):
python -m scripts.create_user alex --role admin

# 3. Boot the server:
make webapp
```

If `WEBAPP_SECRET_KEY` is unset or shorter than 32 chars, the server crashes
at startup with a clear message. The DB lives at `webapp/data/webapp.sqlite`
by default (override with `WEBAPP_DB_PATH`). Both paths are gitignored.

## Run it

Backend:

```bash
make webapp-dev      # uvicorn --factory --reload, http://127.0.0.1:8000
make webapp          # production-style boot (no reload)
make webapp-test     # pytest + coverage gate
make webapp-typecheck
make webapp-lint
```

Frontend (separate Vite dev server on port 5173, proxies `/api/*` to the
backend on `:8000`):

```bash
make webapp-frontend-install     # npm ci (one-time)
make webapp-openapi-snapshot     # regenerate openapi.snapshot.json
make webapp-frontend-dev         # vite dev, http://localhost:5173
make webapp-frontend-build       # gen:api + production build → dist/
make webapp-frontend-test        # vitest + coverage gate (70%)
make webapp-frontend-typecheck   # tsc --noEmit
make webapp-frontend-lint        # eslint + prettier
```

For local dev, run the backend with `WEBAPP_ENV=development make webapp-dev`
in one terminal and `make webapp-frontend-dev` in another. CORS is open to
`http://localhost:5173` only when `WEBAPP_ENV=development`. The Vite proxy
forwards both `/api/*` HTTP and the `/api/jobs/{id}/stream` WebSocket
upgrade so the session cookie travels through dev mode unchanged.

The backend accepts `POST /api/jobs`, spawns the matching `experiment`
subcommand (`run` / `tune` / `compare` / `holdout-eval`) under
`webapp/data/jobs/<job_id>.{yaml,log}`, and the SPA's Configure + Jobs
pages drive the lifecycle. `/configure` is a 4-card hub: Run / Tune build
experiments from scratch, while Compare / Holdout reuse completed
artifacts (the run/HPO detail pages also surface contextual "Run holdout
eval" CTAs when the source carries the required artifact).

`/deployments` lists saved deployments and hosts an inline picker that shows
**only** models with a holdout evaluation (run- or HPO-sourced, one row per
source), ranked by out-of-sample Sharpe, for one-click deploy. A run that has
no holdout never appears in the picker — deploy it instead from its own detail
page (reachable via the `Runs` page), which warns that you are deploying
without out-of-sample validation. `Deploy` actions also live on each
holdout-eval row. `/deployments/:deploymentId`
computes today's signal on mount via `predict-if-stale`, shows the
append-only signal history, and offers inline rename. The shared
`SignalBadge` renders the LONG / SHORT / FLAT / computing states.

## Cross-links

- Top-level [`README.md`](../README.md) — research framework overview.
- The CLI this app fronts: `python -m scripts.experiment --help`.
