.PHONY: install test test-cpp test-python lint typecheck bench bench-cpp stubs clean

install:
	pip install -e ".[dev]"

test: test-cpp test-python typecheck

test-cpp:
	cd cpp/build && ctest --output-on-failure

test-python:
	pytest tests/ -v --tb=short

typecheck:
	mypy --strict src/ tests/ scripts/

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

bench:
	python scripts/benchmark.py run

bench-cpp:
	cd cpp/build && ./quant_bench --benchmark_format=console

stubs:
	python scripts/regen_stubs.py

clean:
	rm -rf cpp/build/ dist/ *.egg-info .mypy_cache .pytest_cache
