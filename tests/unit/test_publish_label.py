"""Tests for the ``--publish-label`` flow.

Two surfaces:
1. :func:`src.visualization.latex.validate_publish_label` — the regex
   that gatekeeps slugs. Tests canonicalised here so every reporter's
   call site has a single source of truth for the rules.
2. End-to-end: each reporter's ``generate_full_report`` switches to
   the slug-based caption + label when ``publish_label`` is set, and
   keeps the legacy wording (so committed sample artifacts remain
   stable) when it is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.visualization.latex import validate_publish_label

if TYPE_CHECKING:
    from src.orchestration.types import StrategyComparisonReport

_SAMPLE_GOOD_SLUG = "metrics_demo_v1:rev2"


class TestValidatePublishLabel:
    @pytest.mark.parametrize(
        "slug",
        [
            "metrics_demo",
            "tab_demo",
            "Strategy-Spy:2024",
            _SAMPLE_GOOD_SLUG,
            "X",
        ],
    )
    def test_accepts_valid_slugs(self, slug: str) -> None:
        assert validate_publish_label(slug) == slug

    @pytest.mark.parametrize(
        ("slug", "reason"),
        [
            ("", "empty"),
            ("1leading_digit", "starts with digit"),
            ("has space", "space"),
            ("brace{x}", "brace"),
            ("percent%here", "percent"),
            ("hash#tag", "hash"),
            ("dot.in.middle", "dot"),
        ],
    )
    def test_rejects_invalid_slugs(self, slug: str, reason: str) -> None:
        with pytest.raises(ValueError, match="invalid publish_label"):
            validate_publish_label(slug)


class TestStrategyReporterPublishLabel:
    def test_emits_slug_caption_when_set(self, tmp_path: Path) -> None:
        from src.visualization.strategy_reporter import StrategyReporter
        from tests.conftest import (
            comparison_curve_seed,
            make_log_return_equity_curve,
            make_stub_experiment_result,
            make_stub_fold_record,
        )

        folds = (
            make_stub_fold_record(
                0,
                sharpe=1.0,
                equity_curve=make_log_return_equity_curve(
                    1.0, n=12, seed=comparison_curve_seed("Demo", 0)
                ),
            ),
        )
        result = make_stub_experiment_result("Demo", folds=folds)
        StrategyReporter().generate_full_report(result, tmp_path, publish_label="metrics_demo")
        tex = (tmp_path / "tables" / "metrics_summary.tex").read_text()
        assert r"\caption{Fold metrics — metrics_demo}" in tex
        assert r"\label{tab:metrics_metrics_demo}" in tex

    def test_legacy_caption_when_unset(self, tmp_path: Path) -> None:
        from src.visualization.strategy_reporter import StrategyReporter
        from tests.conftest import (
            comparison_curve_seed,
            make_log_return_equity_curve,
            make_stub_experiment_result,
            make_stub_fold_record,
        )

        folds = (
            make_stub_fold_record(
                0,
                sharpe=1.0,
                equity_curve=make_log_return_equity_curve(
                    1.0, n=12, seed=comparison_curve_seed("Demo", 0)
                ),
            ),
        )
        result = make_stub_experiment_result("Demo", folds=folds)
        StrategyReporter().generate_full_report(result, tmp_path)
        tex = (tmp_path / "tables" / "metrics_summary.tex").read_text()
        assert "experiment stub_Demo" in tex
        assert r"\label{tab:metrics_stub_Demo}" in tex


class TestComparisonReporterPublishLabel:
    @staticmethod
    def _stub_report(out_name: str) -> StrategyComparisonReport:
        from datetime import UTC, datetime

        from src.orchestration.types import StrategyComparisonReport
        from tests.conftest import make_stub_aggregate_stats

        stats = {
            "Alpha": make_stub_aggregate_stats(sharpe=1.0),
            "Bravo": make_stub_aggregate_stats(sharpe=0.5),
        }
        ranking = pd.DataFrame(
            [
                {"strategy": "Alpha", "mean_sharpe": 1.0, "std_sharpe": 0.1, "rank": 1},
                {"strategy": "Bravo", "mean_sharpe": 0.5, "std_sharpe": 0.1, "rank": 2},
            ]
        )
        return StrategyComparisonReport(
            out_name=out_name,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            git_sha="stubsha1",
            per_strategy_experiment_id={"Alpha": "stub_Alpha", "Bravo": "stub_Bravo"},
            per_strategy_stats=stats,
            ranking=ranking,
            pairwise=(),
        )

    def test_publish_label_replaces_out_name_in_ranking_tex(self, tmp_path: Path) -> None:
        from src.visualization.comparison_reporter import ComparisonReporter

        report = self._stub_report("compare_dir")
        ComparisonReporter().generate_full_report(
            report,
            tmp_path,
            publish_label="strategies_spy",
        )
        tex = (tmp_path / "tables" / "ranking.tex").read_text()
        assert r"comparison strategies_spy" in tex
        assert r"\label{tab:ranking_strategies_spy}" in tex

    def test_legacy_caption_uses_out_name(self, tmp_path: Path) -> None:
        from src.visualization.comparison_reporter import ComparisonReporter

        report = self._stub_report("compare_dir")
        ComparisonReporter().generate_full_report(report, tmp_path)
        tex = (tmp_path / "tables" / "ranking.tex").read_text()
        assert r"comparison compare_dir" in tex
        assert r"\label{tab:ranking_compare_dir}" in tex


class TestHoldoutPublishLabel:
    def test_invalid_slug_raises_through_reporter(self, tmp_path: Path) -> None:
        """Invalid slugs raise from the validator, regardless of which
        reporter receives the value — covers the shared codepath without
        materialising heavy fixtures for every reporter."""
        from src.visualization.strategy_reporter import StrategyReporter
        from tests.conftest import make_stub_experiment_result, make_stub_fold_record

        result = make_stub_experiment_result(
            "Demo",
            folds=(make_stub_fold_record(0, sharpe=1.0, equity_curve=(1.0, 1.01)),),
        )
        with pytest.raises(ValueError, match="invalid publish_label"):
            StrategyReporter().generate_full_report(result, tmp_path, publish_label="has space")
