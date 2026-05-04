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
- `GET /openapi.json`, `GET /docs` — OpenAPI 3.1 schema + Swagger UI.
- `POST /api/auth/login` — `{username, password}` → `{user}` + sets HttpOnly session cookie. Rate-limited (5 / 15 minutes / IP).
- `POST /api/auth/logout` — clears the session cookie.
- `GET /api/auth/me` — returns the current authenticated user.
- `GET /api/users`, `POST /api/users`, `DELETE /api/users/{id}` — admin-only user CRUD (soft delete).

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

```bash
make webapp-dev      # uvicorn --factory --reload, http://127.0.0.1:8000
make webapp          # production-style boot (no reload)
make webapp-test     # pytest + coverage gate
make webapp-typecheck
make webapp-lint
```

## Cross-links

- Top-level [`README.md`](../README.md) — research framework overview.
- The CLI this app fronts: `python -m scripts.experiment --help`.
