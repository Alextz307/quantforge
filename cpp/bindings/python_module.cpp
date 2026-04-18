#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <quant/core/types.hpp>
#include <quant/engine/backtest_engine.hpp>
#include <quant/engine/slippage.hpp>
#include <quant/indicators/bollinger_bands.hpp>
#include <quant/indicators/garman_klass.hpp>
#include <quant/indicators/macd.hpp>
#include <quant/indicators/parkinson.hpp>
#include <quant/indicators/rsi.hpp>
#include <quant/metrics/performance.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using ContigF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ContigI64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

[[nodiscard]] std::span<const double> as_span(const ContigF64& arr) {
    return {arr.data(), static_cast<size_t>(arr.size())};
}

// Copy a std::vector<double> into a fresh 1-D numpy f64 array.
// Must be called with the GIL held (allocates a Python object).
// TODO(Phase 6): eliminate the copy by returning a numpy array that owns the
// vector via a pybind11 capsule (zero-copy); potentially significant under
// Optuna HPO where indicators are called in inner sweep loops.
[[nodiscard]] py::array_t<double> as_numpy(const std::vector<double>& v) {
    return py::array_t<double>(static_cast<py::ssize_t>(v.size()), v.data());
}

[[nodiscard]] std::vector<quant::Bar> build_bars(
    const ContigI64& timestamps,
    const ContigF64& open,
    const ContigF64& high,
    const ContigF64& low,
    const ContigF64& close,
    const ContigF64& volume
) {
    const py::ssize_t n = timestamps.size();
    if (open.size() != n || high.size() != n || low.size() != n
        || close.size() != n || volume.size() != n) {
        throw py::value_error(
            "timestamps, open, high, low, close, volume must all have the same length"
        );
    }
    std::vector<quant::Bar> bars;
    bars.reserve(static_cast<size_t>(n));
    const auto* ts_p = timestamps.data();
    const auto* o_p = open.data();
    const auto* h_p = high.data();
    const auto* l_p = low.data();
    const auto* c_p = close.data();
    const auto* v_p = volume.data();
    for (py::ssize_t i = 0; i < n; ++i) {
        bars.push_back(quant::Bar{ts_p[i], o_p[i], h_p[i], l_p[i], c_p[i], v_p[i]});
    }
    return bars;
}

[[nodiscard]] std::pair<std::vector<quant::Bar>, std::span<const double>>
marshal_bars_and_signals(
    const ContigI64& timestamps, const ContigF64& open, const ContigF64& high,
    const ContigF64& low, const ContigF64& close, const ContigF64& volume,
    const ContigF64& signals
) {
    auto bars = build_bars(timestamps, open, high, low, close, volume);
    auto sig = as_span(signals);
    if (sig.size() != bars.size()) {
        throw py::value_error("signals length must equal bars length");
    }
    return {std::move(bars), sig};
}

}  // namespace

// Shared kwarg prefix for `run` / `run_scenarios` — the six OHLCV numpy arrays
// plus the signal array, forced by the "pass six arrays" plan (Batch C plan
// §Bridge data convention, avoids structured-array fragility). A macro is the
// only way to share a pack of `py::arg(...)` expansions across `.def()` calls.
#define QE_BARS_SIGNALS_KWARGS                                                 \
    py::arg("timestamps"), py::arg("open"), py::arg("high"), py::arg("low"),   \
    py::arg("close"), py::arg("volume"), py::arg("signals")

PYBIND11_MODULE(quant_engine, m) {
    m.doc() = "C++ quantitative engine (backtesting, metrics) — Python bindings";

    m.def("hello", []() { return std::string("ok"); },
          "Smoke-test hook confirming the compiled C++ extension is loadable.");

    // ── Slippage ──
    py::enum_<quant::SlippageModel>(m, "SlippageModel")
        .value("NoSlippage", quant::SlippageModel::NoSlippage)
        .value("Fixed", quant::SlippageModel::Fixed)
        .value("VolumeScaled", quant::SlippageModel::VolumeScaled);

    py::class_<quant::SlippageConfig>(m, "SlippageConfig")
        .def(py::init([](quant::SlippageModel model, double base_bps,
                         double volume_impact_coeff) {
                 return quant::SlippageConfig{model, base_bps, volume_impact_coeff};
             }),
             py::kw_only(),
             py::arg("model") = quant::SlippageModel::Fixed,
             py::arg("base_bps") = 1.0,
             py::arg("volume_impact_coeff") = 0.0)
        .def_readwrite("model", &quant::SlippageConfig::model)
        .def_readwrite("base_bps", &quant::SlippageConfig::base_bps)
        .def_readwrite("volume_impact_coeff",
                       &quant::SlippageConfig::volume_impact_coeff);

    // ── BacktestResult ──
    py::class_<quant::BacktestResult>(m, "BacktestResult")
        .def_readonly("total_return", &quant::BacktestResult::total_return)
        .def_readonly("annualized_return", &quant::BacktestResult::annualized_return)
        .def_readonly("annualized_volatility",
                      &quant::BacktestResult::annualized_volatility)
        .def_readonly("sharpe_ratio", &quant::BacktestResult::sharpe_ratio)
        .def_readonly("sortino_ratio", &quant::BacktestResult::sortino_ratio)
        .def_readonly("max_drawdown", &quant::BacktestResult::max_drawdown)
        .def_readonly("win_rate", &quant::BacktestResult::win_rate)
        .def_readonly("trade_count", &quant::BacktestResult::trade_count)
        // TODO(Phase 6): zero-copy via pybind11 capsule keeping the
        // BacktestResult alive. Current path copies the whole vector each
        // access — fine for single-scenario runs, potentially hot under HPO.
        .def_property_readonly(
            "equity_curve",
            [](const quant::BacktestResult& r) {
                return py::array_t<double>(
                    static_cast<py::ssize_t>(r.equity_curve.size()),
                    r.equity_curve.data());
            })
        .def_readonly("scenario_label", &quant::BacktestResult::scenario_label);

    // ── BacktestEngine ──
    py::class_<quant::BacktestEngine>(m, "BacktestEngine")
        .def(py::init([](double initial_capital, double transaction_fee_rate,
                         bool allow_short) {
                 return quant::BacktestEngine(quant::BacktestEngine::Config{
                     initial_capital, transaction_fee_rate,
                     quant::SlippageConfig{}, allow_short});
             }),
             py::kw_only(),
             py::arg("initial_capital") = 10000.0,
             py::arg("transaction_fee_rate") = 0.001,
             py::arg("allow_short") = true)
        .def(
            "run",
            [](const quant::BacktestEngine& self, const ContigI64& timestamps,
               const ContigF64& open, const ContigF64& high, const ContigF64& low,
               const ContigF64& close, const ContigF64& volume,
               const ContigF64& signals, const quant::SlippageConfig& slippage) {
                const auto [bars, sig_span] = marshal_bars_and_signals(
                    timestamps, open, high, low, close, volume, signals);
                return self.run(bars, sig_span, slippage);
            },
            QE_BARS_SIGNALS_KWARGS, py::arg("slippage"))
        .def(
            "run_scenarios",
            [](const quant::BacktestEngine& self, const ContigI64& timestamps,
               const ContigF64& open, const ContigF64& high, const ContigF64& low,
               const ContigF64& close, const ContigF64& volume,
               const ContigF64& signals,
               const std::vector<quant::SlippageConfig>& scenarios) {
                if (scenarios.empty()) {
                    return std::vector<quant::BacktestResult>{};
                }
                const auto [bars, sig_span] = marshal_bars_and_signals(
                    timestamps, open, high, low, close, volume, signals);
                std::vector<quant::BacktestResult> results;
                results.reserve(scenarios.size());
                for (const auto& sc : scenarios) {
                    results.push_back(self.run(bars, sig_span, sc));
                }
                return results;
            },
            QE_BARS_SIGNALS_KWARGS, py::arg("scenarios"));

#undef QE_BARS_SIGNALS_KWARGS

    // ── PerformanceMetrics ──
    py::class_<quant::PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("annualized_return",
                      &quant::PerformanceMetrics::annualized_return)
        .def_readonly("annualized_volatility",
                      &quant::PerformanceMetrics::annualized_volatility)
        .def_readonly("sharpe_ratio", &quant::PerformanceMetrics::sharpe_ratio)
        .def_readonly("sortino_ratio", &quant::PerformanceMetrics::sortino_ratio)
        .def_readonly("max_drawdown", &quant::PerformanceMetrics::max_drawdown)
        .def_readonly("calmar_ratio", &quant::PerformanceMetrics::calmar_ratio)
        .def_readonly("win_rate", &quant::PerformanceMetrics::win_rate);

    // ── MetricsCalculator ── uninstantiable namespace wrapper.
    py::class_<quant::MetricsCalculator>(m, "MetricsCalculator")
        .def_static(
            "compute",
            [](const ContigF64& equity_curve, int annualization_factor,
               double risk_free_rate) {
                return quant::MetricsCalculator::compute(
                    as_span(equity_curve), annualization_factor, risk_free_rate);
            },
            py::arg("equity_curve"), py::arg("annualization_factor"),
            py::arg("risk_free_rate") = 0.0)
        .def_static(
            "sharpe_ratio",
            [](const ContigF64& returns, int annualization_factor,
               double risk_free_rate) {
                return quant::MetricsCalculator::sharpe_ratio(
                    as_span(returns), annualization_factor, risk_free_rate);
            },
            py::arg("returns"), py::arg("annualization_factor"),
            py::arg("risk_free_rate") = 0.0)
        .def_static(
            "sortino_ratio",
            [](const ContigF64& returns, int annualization_factor,
               double risk_free_rate) {
                return quant::MetricsCalculator::sortino_ratio(
                    as_span(returns), annualization_factor, risk_free_rate);
            },
            py::arg("returns"), py::arg("annualization_factor"),
            py::arg("risk_free_rate") = 0.0)
        .def_static(
            "max_drawdown",
            [](const ContigF64& equity_curve) {
                return quant::MetricsCalculator::max_drawdown(as_span(equity_curve));
            },
            py::arg("equity_curve"))
        .def_static(
            "win_rate",
            [](const ContigF64& returns) {
                return quant::MetricsCalculator::win_rate(as_span(returns));
            },
            py::arg("returns"))
        .def_static(
            "annualized_return",
            [](const ContigF64& equity_curve, int annualization_factor) {
                return quant::MetricsCalculator::annualized_return(
                    as_span(equity_curve), annualization_factor);
            },
            py::arg("equity_curve"), py::arg("annualization_factor"))
        .def_static(
            "annualized_volatility",
            [](const ContigF64& returns, int annualization_factor) {
                return quant::MetricsCalculator::annualized_volatility(
                    as_span(returns), annualization_factor);
            },
            py::arg("returns"), py::arg("annualization_factor"));

    // ── RSI ──
    py::class_<quant::RSI>(m, "RSI")
        .def(py::init<int>(), py::arg("period") = 14)
        .def(
            "compute",
            [](const quant::RSI& self, const ContigF64& prices) {
                const auto input = as_span(prices);
                std::vector<double> out;
                {
                    py::gil_scoped_release release;
                    out = self.compute(input);
                }
                return as_numpy(out);
            },
            py::arg("prices"))
        .def_property_readonly("warmup_period", &quant::RSI::warmup_period)
        .def_property_readonly("name", &quant::RSI::name);

    // ── MACDResult ──
    py::class_<quant::MACDResult>(m, "MACDResult")
        .def_property_readonly(
            "macd_line",
            [](const quant::MACDResult& r) { return as_numpy(r.macd_line); })
        .def_property_readonly(
            "signal_line",
            [](const quant::MACDResult& r) { return as_numpy(r.signal_line); })
        .def_property_readonly(
            "histogram",
            [](const quant::MACDResult& r) { return as_numpy(r.histogram); });

    // ── MACD ──
    py::class_<quant::MACD>(m, "MACD")
        .def(py::init<int, int, int>(),
             py::arg("fast_period") = 12,
             py::arg("slow_period") = 26,
             py::arg("signal_period") = 9)
        .def(
            "compute",
            [](const quant::MACD& self, const ContigF64& prices) {
                const auto input = as_span(prices);
                std::vector<double> out;
                {
                    py::gil_scoped_release release;
                    out = self.compute(input);
                }
                return as_numpy(out);
            },
            py::arg("prices"))
        .def(
            "compute_all",
            [](const quant::MACD& self, const ContigF64& prices) {
                const auto input = as_span(prices);
                py::gil_scoped_release release;
                return self.compute_all(input);
            },
            py::arg("prices"))
        .def_property_readonly("warmup_period", &quant::MACD::warmup_period)
        .def_property_readonly("name", &quant::MACD::name);

    // ── BollingerResult ──
    py::class_<quant::BollingerResult>(m, "BollingerResult")
        .def_property_readonly(
            "upper",
            [](const quant::BollingerResult& r) { return as_numpy(r.upper); })
        .def_property_readonly(
            "mid",
            [](const quant::BollingerResult& r) { return as_numpy(r.mid); })
        .def_property_readonly(
            "lower",
            [](const quant::BollingerResult& r) { return as_numpy(r.lower); });

    // ── BollingerBands ──
    py::class_<quant::BollingerBands>(m, "BollingerBands")
        .def(py::init<int, double>(),
             py::arg("period") = 20, py::arg("num_std") = 2.0)
        .def(
            "compute",
            [](const quant::BollingerBands& self, const ContigF64& prices) {
                const auto input = as_span(prices);
                std::vector<double> out;
                {
                    py::gil_scoped_release release;
                    out = self.compute(input);
                }
                return as_numpy(out);
            },
            py::arg("prices"))
        .def(
            "compute_all",
            [](const quant::BollingerBands& self, const ContigF64& prices) {
                const auto input = as_span(prices);
                py::gil_scoped_release release;
                return self.compute_all(input);
            },
            py::arg("prices"))
        .def_property_readonly("warmup_period",
                               &quant::BollingerBands::warmup_period)
        .def_property_readonly("name", &quant::BollingerBands::name);

    // ── Parkinson ──
    py::class_<quant::Parkinson>(m, "Parkinson")
        .def(py::init<int>(), py::arg("window") = 22)
        .def(
            "compute",
            [](const quant::Parkinson& self, const ContigF64& open,
               const ContigF64& high, const ContigF64& low,
               const ContigF64& close) {
                const auto o = as_span(open);
                const auto h = as_span(high);
                const auto l = as_span(low);
                const auto c = as_span(close);
                std::vector<double> out;
                {
                    py::gil_scoped_release release;
                    out = self.compute(o, h, l, c);
                }
                return as_numpy(out);
            },
            py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"))
        .def_property_readonly("warmup_period", &quant::Parkinson::warmup_period)
        .def_property_readonly("name", &quant::Parkinson::name);

    // ── GarmanKlass ──
    py::class_<quant::GarmanKlass>(m, "GarmanKlass")
        .def(py::init<int>(), py::arg("window") = 22)
        .def(
            "compute",
            [](const quant::GarmanKlass& self, const ContigF64& open,
               const ContigF64& high, const ContigF64& low,
               const ContigF64& close) {
                const auto o = as_span(open);
                const auto h = as_span(high);
                const auto l = as_span(low);
                const auto c = as_span(close);
                std::vector<double> out;
                {
                    py::gil_scoped_release release;
                    out = self.compute(o, h, l, c);
                }
                return as_numpy(out);
            },
            py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"))
        .def_property_readonly("warmup_period", &quant::GarmanKlass::warmup_period)
        .def_property_readonly("name", &quant::GarmanKlass::name);
}
