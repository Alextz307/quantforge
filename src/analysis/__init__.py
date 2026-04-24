"""Cross-run / cross-fold statistical analysis — aggregation, significance, ranking.

Kept deliberately separate from :mod:`src.orchestration`: the orchestration
layer is concerned with *producing* fold records, this package is concerned
with *interpreting* them (summary stats, confidence intervals, pairwise
significance tests, ranking tables). A future regime-analysis split plugs
in here too.
"""

from __future__ import annotations
