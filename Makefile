.PHONY: install test test-cpp test-python lint typecheck bench bench-cpp bench-baseline bench-report stubs clean

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

stubs:
	python scripts/regen_stubs.py

clean:
	rm -rf cpp/build/ dist/ *.egg-info .mypy_cache .pytest_cache
