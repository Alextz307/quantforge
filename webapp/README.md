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
| `backend/app/` | FastAPI application: `api/` routers, `core/` (settings, lifespan, version), and (later) `services/`, `infrastructure/`, `schemas/`. |
| `backend/tests/` | `unit/` (service-layer + router-level tests via `TestClient`) and `integration/` (subprocess-driven). |
| `frontend/` | React + TypeScript + Vite SPA (added in a later sub-batch). |
| `data/` | Local SQLite store for users + jobs. Gitignored. |

## Public surface (so far)

- `GET /api/health` — `{status, version}` liveness probe (no auth).
- `GET /openapi.json` — OpenAPI 3.1 schema (drives the typed frontend client).
- `GET /docs` — Swagger UI for dev convenience.

## Run it

```bash
make webapp-dev      # uvicorn with --reload, http://127.0.0.1:8000
make webapp          # production-style boot (no reload)
make webapp-test     # pytest + coverage gate
make webapp-typecheck
make webapp-lint
```

## Cross-links

- Top-level [`README.md`](../README.md) — research framework overview.
- The CLI this app fronts: `python -m scripts.experiment --help`.
