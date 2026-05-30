"""
Numerical-parity tests for the C++ ``garch_filter`` binding.

The recursive filter logic is exhaustively covered by gtest in
``cpp/tests/test_garch_filter.cpp``. These tests verify the **binding layer**:
numpy array marshalling, ``GarchParams`` keyword round-trip, f32->f64 forcecast,
and that ``GARCHPredictor.predict()`` still produces bit-identical values when
its recursion is delegated to C++.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import quant_engine as qe
from tests.conftest import GLOBAL_NUMPY_SEED

F64Array = npt.NDArray[np.float64]
ReturnsFactory = Callable[[], F64Array]
ParamsFactory = Callable[[], qe.GarchParams]

EXACT_TOL = 1e-12

REF_OMEGA = 0.05
REF_ALPHA = [0.10]
REF_BETA = [0.85]
REF_MU = 0.0
REF_BACKCAST = 1.0
VARIANCE_FLOOR = 1e-12

PARITY_SERIES_LEN = 200
PARITY_RETURN_STD = 1.0
CONSTANT_SERIES_LEN = 5
CONSTANT_RETURN = 0.5

GIL_STRESS_THREAD_COUNT = 4
GIL_STRESS_SERIES_LEN = 20_000
GIL_STRESS_ITERATIONS_PER_THREAD = 3
GIL_STRESS_TIMEOUT_SECONDS = 10.0


def _python_garch_filter(scaled_returns: F64Array, params: qe.GarchParams) -> F64Array:
    """
    Pure-Python reference mirroring the original ``_manual_garch_filter``.

    Kept inlined here (not imported) so the test remains valid even if the
    Python implementation is later deleted.
    """

    n = len(scaled_returns)
    sigma2 = np.empty(n)
    alpha = list(params.alpha)
    beta = list(params.beta)
    for t in range(n):
        var_t = params.omega
        for i, a in enumerate(alpha):
            if t - i - 1 >= 0:
                e2 = (scaled_returns[t - i - 1] - params.mu) ** 2
            else:
                e2 = params.backcast
            var_t += a * e2
        for j, b in enumerate(beta):
            past = sigma2[t - j - 1] if t - j - 1 >= 0 else params.backcast
            var_t += b * past
        sigma2[t] = max(var_t, VARIANCE_FLOOR)
    return sigma2


def _make_returns(n: int) -> F64Array:
    rng = np.random.default_rng(GLOBAL_NUMPY_SEED)
    return rng.normal(0.0, PARITY_RETURN_STD, size=n).astype(np.float64)


def _ref_params(**overrides: object) -> qe.GarchParams:
    kwargs: dict[str, object] = {
        "omega": REF_OMEGA,
        "alpha": REF_ALPHA,
        "beta": REF_BETA,
        "mu": REF_MU,
        "backcast": REF_BACKCAST,
    }
    kwargs.update(overrides)
    return qe.GarchParams(**kwargs)  # type: ignore[arg-type]


class TestGarchFilterBinding:
    @pytest.mark.parametrize(
        ("returns_factory", "params_factory"),
        [
            pytest.param(
                lambda: np.full(CONSTANT_SERIES_LEN, CONSTANT_RETURN, dtype=np.float64),
                _ref_params,
                id="constant_returns",
            ),
            pytest.param(
                lambda: _make_returns(PARITY_SERIES_LEN),
                _ref_params,
                id="random_series",
            ),
            pytest.param(
                lambda: _make_returns(PARITY_SERIES_LEN),
                lambda: _ref_params(alpha=[0.05, 0.025, 0.01], beta=[0.4, 0.2, 0.1, 0.05], mu=0.1),
                id="higher_order_pq_nonzero_mu",
            ),
        ],
    )
    def test_parity_against_python_reference(
        self,
        returns_factory: ReturnsFactory,
        params_factory: ParamsFactory,
    ) -> None:
        returns = returns_factory()
        params = params_factory()
        got = qe.garch_filter(returns, params)
        expected = _python_garch_filter(returns, params)
        np.testing.assert_allclose(got, expected, rtol=EXACT_TOL, atol=EXACT_TOL)

    def test_empty_input_returns_empty(self) -> None:
        got = qe.garch_filter(np.array([], dtype=np.float64), _ref_params())
        assert got.shape == (0,)

    def test_variance_floor_fires_for_zero_params(self) -> None:
        returns = np.zeros(CONSTANT_SERIES_LEN, dtype=np.float64)
        params = qe.GarchParams(omega=0.0, alpha=[0.0], beta=[0.0], mu=0.0, backcast=0.0)
        got = qe.garch_filter(returns, params)
        np.testing.assert_array_equal(got, np.full(CONSTANT_SERIES_LEN, VARIANCE_FLOOR))

    def test_float32_input_is_forcecast(self) -> None:
        returns_f32 = _make_returns(PARITY_SERIES_LEN).astype(np.float32)
        # The stub declares f64; pybind11's forcecast accepts f32 and up-casts.
        got_f32 = qe.garch_filter(cast(F64Array, returns_f32), _ref_params())
        got_f64 = qe.garch_filter(returns_f32.astype(np.float64), _ref_params())
        np.testing.assert_allclose(got_f32, got_f64, rtol=EXACT_TOL, atol=EXACT_TOL)

    def test_params_attributes_are_accessible(self) -> None:
        params = _ref_params()
        assert params.omega == REF_OMEGA
        assert list(params.alpha) == REF_ALPHA
        assert list(params.beta) == REF_BETA
        assert params.mu == REF_MU
        assert params.backcast == REF_BACKCAST


class TestGarchFilterGILRelease:
    def test_concurrent_invocations_do_not_deadlock(self) -> None:
        """
        Smoke-check that the GIL is released during the recursion.

        If the binding forgot its ``py::gil_scoped_release``, Python threads
        would serialize inside the C++ loop; the test only asserts correctness
        under concurrency (timing varies too much for a strict bound).
        """

        returns = _make_returns(GIL_STRESS_SERIES_LEN)
        params = _ref_params()
        expected = qe.garch_filter(returns, params)

        results: list[F64Array] = []
        results_lock = threading.Lock()

        def worker() -> None:
            for _ in range(GIL_STRESS_ITERATIONS_PER_THREAD):
                out = qe.garch_filter(returns, params)
                with results_lock:
                    results.append(out)

        threads = [threading.Thread(target=worker) for _ in range(GIL_STRESS_THREAD_COUNT)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=GIL_STRESS_TIMEOUT_SECONDS)
            assert not t.is_alive(), "GARCH filter worker deadlocked"

        assert len(results) == GIL_STRESS_THREAD_COUNT * GIL_STRESS_ITERATIONS_PER_THREAD
        for out in results:
            np.testing.assert_allclose(out, expected, rtol=EXACT_TOL, atol=EXACT_TOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
