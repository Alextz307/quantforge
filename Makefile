.PHONY: install test test-cpp test-python lint typecheck bench bench-cpp bench-baseline bench-report experiment thesis-demo stubs clean

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

bench:
	python -m scripts.benchmark run

bench-cpp:
	cd cpp/build && ./quant_bench --benchmark_format=console

bench-baseline:
	@test -n "$(NAME)" || { echo "usage: make bench-baseline NAME=<baseline-name>"; exit 1; }
	python -m scripts.benchmark run --save-baseline $(NAME)

bench-report:
	python -m scripts.benchmark run
	@echo "Reports written under benchmark_results/reports/"

experiment:
	@test -n "$(CONFIG)" || { echo "usage: make experiment CONFIG=<path/to/config.yaml>"; exit 1; }
	python -m scripts.experiment run --config $(CONFIG)

tune:
	@test -n "$(CONFIG)" || { echo "usage: make tune CONFIG=<cfg.yaml> HPO=<hpo.yaml> [TRIALS=n] [NJOBS=n]"; exit 1; }
	@test -n "$(HPO)" || { echo "usage: make tune CONFIG=<cfg.yaml> HPO=<hpo.yaml> [TRIALS=n] [NJOBS=n]"; exit 1; }
	python -m scripts.experiment tune --config $(CONFIG) --hpo-config $(HPO) \
		$(if $(TRIALS),--trials $(TRIALS)) $(if $(NJOBS),--n-jobs $(NJOBS))

THESIS_DEMO_DIR := experiment_results/thesis_demo

# Compose the demo's offline data block from the canonical strategy YAMLs
# via --override so we don't keep parquet-pointing duplicates in
# config/. The strategy YAMLs default to yfinance + SPY + the same date
# window, so only the data.source needs swapping.
THESIS_DEMO_DATA_OVERRIDES := \
	--override 'data.source.name=parquet' \
	--override 'data.source.params.data_dir=tests/fixtures'

thesis-demo:
	@echo "──────────────────────────────────────────────────────────────"
	@echo "  thesis-demo: end-to-end pipeline smoke on cached SPY"
	@echo "  Output is illustrative — NOT a benchmark and NOT empirical"
	@echo "  results. The comprehensive empirical study will land under"
	@echo "  experiment_results/studies/ separately."
	@echo "──────────────────────────────────────────────────────────────"
	@rm -rf $(THESIS_DEMO_DIR)/runs \
	        $(THESIS_DEMO_DIR)/comparisons \
	        $(THESIS_DEMO_DIR)/regime_reports
	python -m scripts.experiment run \
		--config config/strategies/adaptive_bollinger.yaml \
		--store-root $(THESIS_DEMO_DIR) \
		$(THESIS_DEMO_DATA_OVERRIDES)
	python -m scripts.experiment compare \
		--config config/strategies/adaptive_bollinger.yaml \
		--config config/strategies/momentum_gatekeeper.yaml \
		--out-name pipeline_compare \
		--store-root $(THESIS_DEMO_DIR) \
		$(THESIS_DEMO_DATA_OVERRIDES)
	@EXP_ID=$$(ls -1t $(THESIS_DEMO_DIR)/runs/ | head -n 1); \
		test -n "$$EXP_ID" || { echo "no run found under $(THESIS_DEMO_DIR)/runs/"; exit 1; }; \
		echo "regime split on $$EXP_ID"; \
		python -m scripts.experiment regime \
			--exp-id $$EXP_ID \
			--regime-config config/regimes/bull_bear_200ma.yaml \
			--out-name pipeline_regime \
			--store-root $(THESIS_DEMO_DIR)
	@echo "──────────────────────────────────────────────────────────────"
	@echo "  thesis-demo finished. Fresh artifacts under:"
	@echo "    $(THESIS_DEMO_DIR)/runs/<exp_id>/"
	@echo "    $(THESIS_DEMO_DIR)/comparisons/pipeline_compare/"
	@echo "    $(THESIS_DEMO_DIR)/regime_reports/pipeline_regime/"
	@echo "  Committed sample lives at $(THESIS_DEMO_DIR)/sample/."
	@echo "──────────────────────────────────────────────────────────────"

stubs:
	python scripts/regen_stubs.py

clean:
	rm -rf cpp/build/ dist/ *.egg-info .mypy_cache .pytest_cache
