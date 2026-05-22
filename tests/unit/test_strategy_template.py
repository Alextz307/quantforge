"""Drift guards for ``src/strategies/_template.py``.

The template is an underscored module on purpose — ``autoload_package``
must skip it so it never registers itself, and its abstract-method stubs
must surface a clear error if a reader instantiates it without renaming
and filling in the logic. Both invariants are tested here so a future
refactor that, say, adds a new abstract method to ``IStrategy`` without
stubbing it on the template fails loudly at CI time.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.strategies._template import _TemplateStrategy
from tests.conftest import make_synthetic_close_df


@pytest.fixture
def unfitted_template() -> tuple[_TemplateStrategy, pd.DataFrame]:
    """Unfitted template + a tiny synthetic close frame, shared by the stub-and-guard tests."""
    return _TemplateStrategy(), make_synthetic_close_df()


def test_template_module_does_not_register() -> None:
    """``_template.py`` must NOT appear under any plausible name in the registry.

    The leading-underscore skip in ``autoload_package`` is the mechanism that
    keeps this honest; this test asserts the resulting state. If a future
    refactor changes the autoload skip rules, this fails before any
    user-visible "strategy not found" surprise.

    The ``import src.strategies`` is local to this test so the cold autoload
    cost (torch / xgboost / statsmodels / pmdarima) doesn't get charged to
    the other six tests, which only need ``_TemplateStrategy``.
    """
    import src.strategies  # noqa: F401 — populates strategy_registry via autoload side-effect
    from src.core.registry import strategy_registry

    forbidden = {"Template", "template", "_Template", "_template", "TemplateStrategy"}
    registered = set(strategy_registry.list_all())
    assert forbidden.isdisjoint(registered), (
        f"_TemplateStrategy leaked into the registry under "
        f"{sorted(forbidden & registered)}; check autoload_package skip rules."
    )


def test_template_train_raises_not_implemented(
    unfitted_template: tuple[_TemplateStrategy, pd.DataFrame],
) -> None:
    """``train()`` is stubbed; calling it must surface a clear remediation message."""
    s, df = unfitted_template
    with pytest.raises(NotImplementedError, match="train"):
        s.train(df)


def test_template_generate_signals_before_train_raises_runtime_error(
    unfitted_template: tuple[_TemplateStrategy, pd.DataFrame],
) -> None:
    """Calling ``generate_signals`` before ``train`` must raise ``RuntimeError``.

    Exercises the read-side fitted-state guard. Triggering the post-guard
    ``NotImplementedError`` would require a working ``train()`` call, which
    the template doesn't have — that branch is left unexercised by design.
    """
    s, df = unfitted_template
    with pytest.raises(RuntimeError, match="before train"):
        s.generate_signals(df)


def test_template_invalid_window_raises() -> None:
    """Ctor validation fires — readers see the same error shape as real strategies."""
    with pytest.raises(ValueError, match="window"):
        _TemplateStrategy(window=1)


def test_template_invalid_threshold_raises() -> None:
    """Ctor validation fires — readers see the same error shape as real strategies."""
    with pytest.raises(ValueError, match="threshold"):
        _TemplateStrategy(threshold=0.0)


def test_template_suggest_params_keys_match_ctor() -> None:
    """``suggest_params`` keys must be a subset of ctor kwarg names.

    If a future edit renames a ctor kwarg without updating ``suggest_params``,
    StrategyTuner would raise ``TypeError`` at trial-build time — this test
    catches that drift at unit-test time.
    """
    import inspect

    import optuna

    study = optuna.create_study()
    trial = study.ask()
    suggested = _TemplateStrategy.suggest_params(trial)

    ctor_kwargs = set(inspect.signature(_TemplateStrategy.__init__).parameters) - {"self"}
    assert set(suggested).issubset(ctor_kwargs), (
        f"suggest_params keys {sorted(suggested)} are not a subset of ctor "
        f"kwargs {sorted(ctor_kwargs)} — StrategyTuner would TypeError."
    )
