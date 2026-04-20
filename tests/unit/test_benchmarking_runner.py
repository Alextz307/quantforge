"""Runner JSON-parsing tests — exercise the gbench output shape via a fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmarking.runner import parse_gbench_json

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "google_benchmark_sample.json"

EXPECTED_NAMES = [
    "BM_RSI/10000",
    "BM_RSI/100000",
    "BM_RSI/1000000",
    "BM_MACD/10000",
]
US_TO_NS = 1_000.0


@pytest.fixture
def fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def test_parse_gbench_json_returns_one_result_per_benchmark(fixture_text: str) -> None:
    results = parse_gbench_json(fixture_text)
    assert [r.name for r in results] == EXPECTED_NAMES


def test_parse_extracts_family_and_size_params(fixture_text: str) -> None:
    results = parse_gbench_json(fixture_text)
    rsi = next(r for r in results if r.name == "BM_RSI/10000")
    assert rsi.family == "BM_RSI"
    assert rsi.params == {"n": 10000}


def test_parse_converts_time_unit_to_ns(fixture_text: str) -> None:
    results = parse_gbench_json(fixture_text)
    macd = next(r for r in results if r.name == "BM_MACD/10000")
    # fixture has real_time=60us -> must convert to 60000 ns
    assert macd.real_time_ns == 60.0 * US_TO_NS


def test_parse_captures_custom_counters(fixture_text: str) -> None:
    results = parse_gbench_json(fixture_text)
    macd = next(r for r in results if r.name == "BM_MACD/10000")
    assert macd.custom_counters["Cycles"] == 180000.0
    assert macd.custom_counters["IPC"] == 2.0


def test_parse_ignores_aggregate_rows() -> None:
    payload = """{
        "context": {},
        "benchmarks": [
            {
                "name": "BM_X/100",
                "iterations": 1,
                "real_time": 1.0,
                "cpu_time": 1.0,
                "time_unit": "ns",
                "items_per_second": 1.0,
                "run_type": "iteration"
            },
            {
                "name": "BM_X/100_mean",
                "iterations": 1,
                "real_time": 1.0,
                "cpu_time": 1.0,
                "time_unit": "ns",
                "items_per_second": 1.0,
                "run_type": "aggregate"
            }
        ]
    }"""
    results = parse_gbench_json(payload)
    assert [r.name for r in results] == ["BM_X/100"]


@pytest.mark.parametrize(
    ("unit", "value", "expected_ns"),
    [
        ("ns", 500.0, 500.0),
        ("us", 60.0, 60_000.0),
        ("ms", 2.5, 2_500_000.0),
        ("s", 0.1, 100_000_000.0),
    ],
)
def test_parse_converts_every_supported_unit(unit: str, value: float, expected_ns: float) -> None:
    payload = f"""{{
        "context": {{}},
        "benchmarks": [
            {{
                "name": "BM_X/100",
                "iterations": 1,
                "real_time": {value},
                "cpu_time": {value},
                "time_unit": "{unit}",
                "items_per_second": 1.0,
                "run_type": "iteration"
            }}
        ]
    }}"""
    results = parse_gbench_json(payload)
    assert results[0].real_time_ns == expected_ns


def test_parse_rejects_unknown_time_unit() -> None:
    payload = """{
        "context": {},
        "benchmarks": [
            {
                "name": "BM_X/100",
                "iterations": 1,
                "real_time": 1.0,
                "cpu_time": 1.0,
                "time_unit": "fortnight",
                "items_per_second": 1.0,
                "run_type": "iteration"
            }
        ]
    }"""
    with pytest.raises(ValueError, match="unknown time unit"):
        parse_gbench_json(payload)
