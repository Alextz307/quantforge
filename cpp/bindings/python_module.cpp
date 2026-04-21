#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <quant/core/types.hpp>
#include <quant/engine/backtest_engine.hpp>
#include <quant/engine/slippage.hpp>
#include <quant/filters/garch_filter.hpp>
#include <quant/indicators/bollinger_bands.hpp>
#include <quant/indicators/garman_klass.hpp>
#include <quant/indicators/macd.hpp>
#include <quant/indicators/parkinson.hpp>
#include <quant/indicators/rsi.hpp>
#include <quant/metrics/performance.hpp>
#include <quant/statistics/spread.hpp>
#include <quant/strategies/adaptive_bollinger.hpp>
#include <quant/strategies/pairs_trading.hpp>
#include <quant/strategies/state_machines.hpp>

#include <cstdint>
#include <memory>
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

// The capsule holds the sole shared_ptr ref, keeping the vector alive for the
// numpy array's lifetime; the shared_ptr releases when the capsule destructs.
// Destructor touches no Python state, so no GIL acquire is needed.
// Must be called with the GIL held — constructs a py::capsule (Python object).
[[nodiscard]] py::array_t<double> wrap_vector_zero_copy(
    std::shared_ptr<std::vector<double>> vec) {
    double* data_ptr = vec->data();
    const auto n = static_cast<py::ssize_t>(vec->size());
    auto* raw_owner = new std::shared_ptr<std::vector<double>>(std::move(vec));
    py::capsule owner(raw_owner, [](void* p) {
        delete static_cast<std::shared_ptr<std::vector<double>>*>(p);
    });
    return py::array_t<double>({n}, {sizeof(double)}, data_ptr, owner);
}

// Zero-copy view into a ``std::vector<double>`` field of a pybind11-held
// struct. Passing ``self_obj`` as the numpy array's ``base`` makes numpy
// hold an inc_ref'd handle to the parent Python wrapper; storage stays
// valid until the numpy array is GC'd. Cheaper than a capsule owning a
// fresh ``py::object`` and sidesteps GIL-at-finalizer concerns (numpy
// releases the base via the normal refcount path).
[[nodiscard]] py::array_t<double> make_field_view(
    py::handle self_obj, std::vector<double>& field) {
    return py::array_t<double>(
        {static_cast<py::ssize_t>(field.size())},
        {sizeof(double)},
        field.data(),
        self_obj);
}

// Pointer-to-member factory: produces a property-reader lambda for a
// ``std::vector<double>`` field on a pybind11-held struct. Collapses the
// seven per-field copies across MACDResult / BollingerResult /
// BacktestResult into a single declarative binding.
template <typename T, std::vector<double> T::*FieldPtr>
[[nodiscard]] auto field_view_reader() {
    return [](py::object self_obj) {
        return make_field_view(self_obj, self_obj.cast<T&>().*FieldPtr);
    };
}

// Allocate a std::shared_ptr-owned vector, run the compute on it while holding
// the GIL-released scope, and return a zero-copy numpy view. Shared path for
// every single-output indicator binding.
template <typename Fn>
[[nodiscard]] py::array_t<double> allocate_and_compute(std::size_t n, Fn&& fn) {
    auto buf = std::make_shared<std::vector<double>>(n);
    {
        py::gil_scoped_release release;
        fn(std::span<double>(*buf));
    }
    return wrap_vector_zero_copy(std::move(buf));
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

// Shared OHLC kwarg prefix for the volatility-estimator bindings
// (Parkinson, GarmanKlass).
#define QE_OHLC_KWARGS                                                         \
    py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close")

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
    // ``equity_curve`` is a zero-copy numpy view into the result's vector;
    // the capsule keeps the BacktestResult instance alive so the view stays
    // valid even if the Python-side BacktestResult handle is released after
    // the property access.
    py::class_<quant::BacktestResult, std::shared_ptr<quant::BacktestResult>>(
        m, "BacktestResult")
        .def_readonly("total_return", &quant::BacktestResult::total_return)
        .def_readonly("annualized_return", &quant::BacktestResult::annualized_return)
        .def_readonly("annualized_volatility",
                      &quant::BacktestResult::annualized_volatility)
        .def_readonly("sharpe_ratio", &quant::BacktestResult::sharpe_ratio)
        .def_readonly("sortino_ratio", &quant::BacktestResult::sortino_ratio)
        .def_readonly("max_drawdown", &quant::BacktestResult::max_drawdown)
        .def_readonly("win_rate", &quant::BacktestResult::win_rate)
        .def_readonly("trade_count", &quant::BacktestResult::trade_count)
        .def_property_readonly(
            "equity_curve",
            field_view_reader<quant::BacktestResult, &quant::BacktestResult::equity_curve>())
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
                auto result = std::make_shared<quant::BacktestResult>();
                {
                    py::gil_scoped_release release;
                    self.run(bars, sig_span, slippage, *result);
                }
                return result;
            },
            QE_BARS_SIGNALS_KWARGS, py::arg("slippage"))
        .def(
            "run_scenarios",
            [](const quant::BacktestEngine& self, const ContigI64& timestamps,
               const ContigF64& open, const ContigF64& high, const ContigF64& low,
               const ContigF64& close, const ContigF64& volume,
               const ContigF64& signals,
               const std::vector<quant::SlippageConfig>& scenarios) {
                std::vector<std::shared_ptr<quant::BacktestResult>> results;
                if (scenarios.empty()) {
                    return results;
                }
                const auto [bars, sig_span] = marshal_bars_and_signals(
                    timestamps, open, high, low, close, volume, signals);
                results.reserve(scenarios.size());
                {
                    py::gil_scoped_release release;
                    for (const auto& sc : scenarios) {
                        auto r = std::make_shared<quant::BacktestResult>();
                        self.run(bars, sig_span, sc, *r);
                        results.push_back(std::move(r));
                    }
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
                return allocate_and_compute(input.size(),
                    [&](std::span<double> out) { self.compute(input, out); });
            },
            py::arg("prices"))
        .def_property_readonly("warmup_period", &quant::RSI::warmup_period)
        .def_property_readonly("name", &quant::RSI::name);

    // ── MACDResult ── three vectors owned by a shared_ptr-held MACDResult;
    // each property returns a zero-copy numpy view with the MACDResult
    // Python wrapper as its numpy base.
    py::class_<quant::MACDResult, std::shared_ptr<quant::MACDResult>>(m, "MACDResult")
        .def_property_readonly(
            "macd_line",
            field_view_reader<quant::MACDResult, &quant::MACDResult::macd_line>())
        .def_property_readonly(
            "signal_line",
            field_view_reader<quant::MACDResult, &quant::MACDResult::signal_line>())
        .def_property_readonly(
            "histogram",
            field_view_reader<quant::MACDResult, &quant::MACDResult::histogram>());

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
                return allocate_and_compute(input.size(),
                    [&](std::span<double> out) { self.compute(input, out); });
            },
            py::arg("prices"))
        .def(
            "compute_all",
            [](const quant::MACD& self, const ContigF64& prices) {
                const auto input = as_span(prices);
                auto result = std::make_shared<quant::MACDResult>();
                {
                    py::gil_scoped_release release;
                    result->macd_line.resize(input.size());
                    result->signal_line.resize(input.size());
                    result->histogram.resize(input.size());
                    self.compute_all(input, *result);
                }
                return result;
            },
            py::arg("prices"))
        .def_property_readonly("warmup_period", &quant::MACD::warmup_period)
        .def_property_readonly("name", &quant::MACD::name);

    // ── BollingerResult ── same zero-copy pattern as MACDResult.
    py::class_<quant::BollingerResult, std::shared_ptr<quant::BollingerResult>>(
        m, "BollingerResult")
        .def_property_readonly(
            "upper",
            field_view_reader<quant::BollingerResult, &quant::BollingerResult::upper>())
        .def_property_readonly(
            "mid",
            field_view_reader<quant::BollingerResult, &quant::BollingerResult::mid>())
        .def_property_readonly(
            "lower",
            field_view_reader<quant::BollingerResult, &quant::BollingerResult::lower>());

    // ── BollingerBands ──
    py::class_<quant::BollingerBands>(m, "BollingerBands")
        .def(py::init<int, double>(),
             py::arg("period") = 20, py::arg("num_std") = 2.0)
        .def(
            "compute",
            [](const quant::BollingerBands& self, const ContigF64& prices) {
                const auto input = as_span(prices);
                return allocate_and_compute(input.size(),
                    [&](std::span<double> out) { self.compute(input, out); });
            },
            py::arg("prices"))
        .def(
            "compute_all",
            [](const quant::BollingerBands& self, const ContigF64& prices) {
                const auto input = as_span(prices);
                auto result = std::make_shared<quant::BollingerResult>();
                {
                    py::gil_scoped_release release;
                    result->upper.resize(input.size());
                    result->mid.resize(input.size());
                    result->lower.resize(input.size());
                    self.compute_all(input, *result);
                }
                return result;
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
                return allocate_and_compute(o.size(),
                    [&](std::span<double> out) { self.compute(o, h, l, c, out); });
            },
            QE_OHLC_KWARGS)
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
                return allocate_and_compute(o.size(),
                    [&](std::span<double> out) { self.compute(o, h, l, c, out); });
            },
            QE_OHLC_KWARGS)
        .def_property_readonly("warmup_period", &quant::GarmanKlass::warmup_period)
        .def_property_readonly("name", &quant::GarmanKlass::name);

#undef QE_OHLC_KWARGS

    // ── GarchParams ──
    // Fields are read-only; GARCH parameters are frozen after the Python fit.
    py::class_<quant::filters::GarchParams>(m, "GarchParams")
        .def(py::init([](double omega, std::vector<double> alpha,
                         std::vector<double> beta, double mu, double backcast) {
                 return quant::filters::GarchParams{
                     omega, std::move(alpha), std::move(beta), mu, backcast};
             }),
             py::kw_only(),
             py::arg("omega"),
             py::arg("alpha"),
             py::arg("beta"),
             py::arg("mu"),
             py::arg("backcast"))
        .def_readonly("omega", &quant::filters::GarchParams::omega)
        .def_readonly("alpha", &quant::filters::GarchParams::alpha)
        .def_readonly("beta", &quant::filters::GarchParams::beta)
        .def_readonly("mu", &quant::filters::GarchParams::mu)
        .def_readonly("backcast", &quant::filters::GarchParams::backcast);

    // ── garch_filter ──
    m.def(
        "garch_filter",
        [](const ContigF64& scaled_returns,
           const quant::filters::GarchParams& params) {
            const auto input = as_span(scaled_returns);
            auto buf = std::make_shared<std::vector<double>>();
            {
                py::gil_scoped_release release;
                *buf = quant::filters::garch_filter(input, params);
            }
            return wrap_vector_zero_copy(std::move(buf));
        },
        py::arg("scaled_returns"), py::arg("params"),
        "Run the GARCH(p,q) recursion; returns conditional variances.");

    // ── State machines ──
    m.def(
        "run_mean_reversion_state_machine",
        [](const ContigF64& close, const ContigF64& mid, const ContigF64& upper,
           const ContigF64& lower, const ContigF64& trend_ma) {
            const auto close_span = as_span(close);
            const auto mid_span = as_span(mid);
            const auto upper_span = as_span(upper);
            const auto lower_span = as_span(lower);
            const auto trend_ma_span = as_span(trend_ma);
            return allocate_and_compute(close_span.size(),
                [&](std::span<double> out) {
                    quant::strategies::run_mean_reversion_state_machine(
                        close_span, mid_span, upper_span, lower_span,
                        trend_ma_span, out);
                });
        },
        py::arg("close"), py::arg("mid"), py::arg("upper"),
        py::arg("lower"), py::arg("trend_ma"),
        "Run the AdaptiveBollinger state machine; returns a position series.");

    m.def(
        "run_pairs_state_machine",
        [](const ContigF64& zscore, double entry_zscore, double exit_zscore,
           double stop_loss_zscore) {
            const auto zscore_span = as_span(zscore);
            return allocate_and_compute(zscore_span.size(),
                [&](std::span<double> out) {
                    quant::strategies::run_pairs_state_machine(
                        zscore_span, entry_zscore, exit_zscore,
                        stop_loss_zscore, out);
                });
        },
        py::arg("zscore"), py::arg("entry_zscore"), py::arg("exit_zscore"),
        py::arg("stop_loss_zscore"),
        "Run the pairs-trading state machine; returns a position series.");

    // ── CointegrationParams ──
    // spread_mean / spread_std are training-time provenance only — the
    // rolling z-score recomputes them on the inference window. Defaulted
    // so callers who only care about the hedge ratio can omit them.
    py::class_<quant::statistics::CointegrationParams>(m, "CointegrationParams")
        .def(py::init([](double hedge_ratio, double spread_mean, double spread_std) {
                 return quant::statistics::CointegrationParams{
                     hedge_ratio, spread_mean, spread_std};
             }),
             py::kw_only(),
             py::arg("hedge_ratio"),
             py::arg("spread_mean") = 0.0,
             py::arg("spread_std") = 1.0)
        .def_readonly("hedge_ratio", &quant::statistics::CointegrationParams::hedge_ratio)
        .def_readonly("spread_mean", &quant::statistics::CointegrationParams::spread_mean)
        .def_readonly("spread_std", &quant::statistics::CointegrationParams::spread_std);

    // ── SpreadCalculator ──
    py::class_<quant::statistics::SpreadCalculator>(m, "SpreadCalculator")
        .def_static(
            "compute_spread",
            [](const ContigF64& a, const ContigF64& b, double hedge_ratio) {
                const auto a_span = as_span(a);
                const auto b_span = as_span(b);
                return allocate_and_compute(a_span.size(),
                    [&](std::span<double> out) {
                        quant::statistics::SpreadCalculator::compute_spread(
                            a_span, b_span, hedge_ratio, out);
                    });
            },
            py::arg("a"), py::arg("b"), py::arg("hedge_ratio"))
        .def_static(
            "compute_zscore",
            [](const ContigF64& spread, int window) {
                const auto spread_span = as_span(spread);
                return allocate_and_compute(spread_span.size(),
                    [&](std::span<double> out) {
                        quant::statistics::SpreadCalculator::compute_zscore(
                            spread_span, window, out);
                    });
            },
            py::arg("spread"), py::arg("window"));

    // ── PairsTradingStrategy ──
    py::class_<quant::strategies::PairsTradingStrategy> pairs_trading(m, "PairsTradingStrategy");
    py::class_<quant::strategies::PairsTradingStrategy::Config>(pairs_trading, "Config")
        .def(py::init([](double entry_zscore, double exit_zscore,
                         double stop_loss_zscore, int zscore_lookback) {
                 return quant::strategies::PairsTradingStrategy::Config{
                     entry_zscore, exit_zscore, stop_loss_zscore, zscore_lookback};
             }),
             py::kw_only(),
             py::arg("entry_zscore") = 2.0,
             py::arg("exit_zscore") = 0.5,
             py::arg("stop_loss_zscore") = 4.0,
             py::arg("zscore_lookback") = 60)
        .def_readonly("entry_zscore",
                      &quant::strategies::PairsTradingStrategy::Config::entry_zscore)
        .def_readonly("exit_zscore",
                      &quant::strategies::PairsTradingStrategy::Config::exit_zscore)
        .def_readonly("stop_loss_zscore",
                      &quant::strategies::PairsTradingStrategy::Config::stop_loss_zscore)
        .def_readonly("zscore_lookback",
                      &quant::strategies::PairsTradingStrategy::Config::zscore_lookback);
    pairs_trading
        .def(py::init<quant::strategies::PairsTradingStrategy::Config>(), py::arg("config"))
        .def(
            "generate_signals",
            [](const quant::strategies::PairsTradingStrategy& self,
               const ContigF64& prices_a, const ContigF64& prices_b,
               const quant::statistics::CointegrationParams& coint) {
                const auto a_span = as_span(prices_a);
                const auto b_span = as_span(prices_b);
                return allocate_and_compute(a_span.size(),
                    [&](std::span<double> out) {
                        self.generate_signals(a_span, b_span, coint, out);
                    });
            },
            py::arg("prices_a"), py::arg("prices_b"), py::arg("coint"))
        .def_property_readonly("name", &quant::strategies::PairsTradingStrategy::name)
        .def_property_readonly("required_warmup",
                               &quant::strategies::PairsTradingStrategy::required_warmup);

    // ── AdaptiveBollingerStrategy ──
    py::class_<quant::strategies::AdaptiveBollingerStrategy> adaptive_bollinger(
        m, "AdaptiveBollingerStrategy");
    py::class_<quant::strategies::AdaptiveBollingerStrategy::Config>(adaptive_bollinger, "Config")
        .def(py::init([](int band_window, double k, int trend_window) {
                 return quant::strategies::AdaptiveBollingerStrategy::Config{
                     band_window, k, trend_window};
             }),
             py::kw_only(),
             py::arg("band_window") = 20,
             py::arg("k") = 2.0,
             py::arg("trend_window") = 100)
        .def_readonly("band_window",
                      &quant::strategies::AdaptiveBollingerStrategy::Config::band_window)
        .def_readonly("k", &quant::strategies::AdaptiveBollingerStrategy::Config::k)
        .def_readonly("trend_window",
                      &quant::strategies::AdaptiveBollingerStrategy::Config::trend_window);
    adaptive_bollinger
        .def(py::init<quant::strategies::AdaptiveBollingerStrategy::Config>(), py::arg("config"))
        .def(
            "generate_signals",
            [](const quant::strategies::AdaptiveBollingerStrategy& self,
               const ContigF64& close, const ContigF64& cond_vol) {
                const auto close_span = as_span(close);
                const auto cond_vol_span = as_span(cond_vol);
                return allocate_and_compute(close_span.size(),
                    [&](std::span<double> out) {
                        self.generate_signals(close_span, cond_vol_span, out);
                    });
            },
            py::arg("close"), py::arg("cond_vol"))
        .def_property_readonly("name", &quant::strategies::AdaptiveBollingerStrategy::name)
        .def_property_readonly("required_warmup",
                               &quant::strategies::AdaptiveBollingerStrategy::required_warmup);
}
