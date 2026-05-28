.PHONY: install test test-cpp test-python lint typecheck experiment stubs clean webapp webapp-dev webapp-test webapp-typecheck webapp-lint webapp-frontend-install webapp-frontend-dev webapp-frontend-build webapp-frontend-test webapp-frontend-typecheck webapp-frontend-lint webapp-openapi-snapshot webapp-check-openapi-snapshot webapp-check-schema-mirror

install:
	pip install -e ".[dev]"

test: test-cpp test-python typecheck

test-cpp:
	cd cpp/build && ctest --output-on-failure

test-python:
	OMP_NUM_THREADS=1 pytest tests/ -v --tb=short

typecheck:
	mypy --strict src/ tests/ scripts/

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

experiment:
	@test -n "$(CONFIG)" || { echo "usage: make experiment CONFIG=<path/to/config.yaml>"; exit 1; }
	python -m scripts.experiment run --config $(CONFIG)

tune:
	@test -n "$(CONFIG)" || { echo "usage: make tune CONFIG=<cfg.yaml> HPO=<hpo.yaml> [TRIALS=n] [NJOBS=n]"; exit 1; }
	@test -n "$(HPO)" || { echo "usage: make tune CONFIG=<cfg.yaml> HPO=<hpo.yaml> [TRIALS=n] [NJOBS=n]"; exit 1; }
	python -m scripts.experiment tune --config $(CONFIG) --hpo-config $(HPO) \
		$(if $(TRIALS),--trials $(TRIALS)) $(if $(NJOBS),--n-jobs $(NJOBS))

stubs:
	python scripts/regen_stubs.py

webapp-dev:
	uvicorn webapp.backend.app.main:create_app --factory --reload --host 127.0.0.1 --port 8000

webapp:
	WEBAPP_ENV=local uvicorn webapp.backend.app.main:create_app --factory --host 127.0.0.1 --port 8000

webapp-test:
	OMP_NUM_THREADS=1 pytest webapp/backend/tests/ -v --tb=short \
		--cov=webapp/backend/app --cov-report=term-missing --cov-fail-under=80

webapp-typecheck:
	mypy --strict webapp/backend

webapp-lint:
	ruff check webapp/backend
	ruff format --check webapp/backend

webapp-openapi-snapshot:
	python -m scripts.dump_openapi

webapp-check-openapi-snapshot:
	python -m scripts.check_openapi_snapshot

webapp-check-schema-mirror:
	python -m scripts.check_webapp_schema_mirror

webapp-frontend-install:
	cd webapp/frontend && npm ci

webapp-frontend-dev:
	cd webapp/frontend && npm run dev

webapp-frontend-build:
	cd webapp/frontend && npm run gen:api && npm run build

webapp-frontend-test:
	cd webapp/frontend && npm run gen:api && npm run test:cov

webapp-frontend-typecheck:
	cd webapp/frontend && npm run gen:api && npm run typecheck

webapp-frontend-lint:
	cd webapp/frontend && npm run gen:api && npm run lint && npm run format:check

clean:
	rm -rf cpp/build/ dist/ *.egg-info .mypy_cache .pytest_cache
