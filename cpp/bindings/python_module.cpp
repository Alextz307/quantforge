#include <pybind11/pybind11.h>

#include <string>

PYBIND11_MODULE(quant_engine, m) {
    m.doc() = "C++ quantitative engine (backtesting, metrics) — Python bindings";

    m.def("hello", []() { return std::string("ok"); },
          "Smoke-test hook confirming the compiled C++ extension is loadable.");
}
