"""
Smoke test - the compiled C++ extension module is importable and responsive.

If this test fails, the pybind11 + scikit-build-core wiring is broken and
every subsequent engine/metrics binding test will fail too. Keep it as the
first sanity check before adding real C++ surface area.
"""


class TestCppModuleLoads:
    def test_package_imports(self) -> None:
        import quant_engine

        assert hasattr(quant_engine, "hello")

    def test_hello_returns_ok(self) -> None:
        from quant_engine import hello

        assert hello() == "ok"
