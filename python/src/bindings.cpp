/*
 * bindings.cpp
 *
 * pybind11 bindings for the Cyclops C++ facade (src/cyclops/api).
 *
 * This file is intentionally mechanical: every function forwards straight to
 * `cyclops::api` with no reordering, defaulting, or validation of its own.
 * Ergonomics live in the Python package; numerics live in the C++ core; this
 * layer only marshals. Keeping it dumb is what makes the facade worth having.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "cyclops/api/CyclopsApi.h"

namespace py = pybind11;
using namespace cyclops::api;

namespace {

/// Copies a std::vector into a fresh 1-D NumPy array.
///
/// Returning `std::vector` directly would hand Python a `list`; every consumer
/// in the package wants an ndarray, so the conversion happens once, here.
template <typename T>
py::array_t<T> toArray(const std::vector<T>& values) {
    py::array_t<T> array(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::copy(values.begin(), values.end(), array.mutable_data());
    }
    return array;
}

}  // namespace

PYBIND11_MODULE(_cyclops, m) {
    m.doc() = "Low-level pybind11 bindings for the Cyclops C++ core. "
              "Use the `cyclops` package instead.";

    py::register_exception<CyclopsError>(m, "CyclopsError", PyExc_RuntimeError);

    // -- enums --------------------------------------------------------------

    py::enum_<ModelKind>(m, "ModelKind")
        .value("NORMAL", ModelKind::Normal)
        .value("POISSON", ModelKind::Poisson)
        .value("LOGISTIC", ModelKind::Logistic)
        .value("CONDITIONAL_LOGISTIC", ModelKind::ConditionalLogistic)
        .value("TIED_CONDITIONAL_LOGISTIC", ModelKind::TiedConditionalLogistic)
        .value("EFRON_CONDITIONAL_LOGISTIC", ModelKind::EfronConditionalLogistic)
        .value("CONDITIONAL_POISSON", ModelKind::ConditionalPoisson)
        .value("SELF_CONTROLLED_CASE_SERIES", ModelKind::SelfControlledCaseSeries)
        .value("COX", ModelKind::Cox)
        .value("COX_RAW", ModelKind::CoxRaw)
        .value("TIME_VARYING_COX", ModelKind::TimeVaryingCox)
        .value("FINE_GRAY", ModelKind::FineGray);

    py::enum_<PriorKind>(m, "PriorKind")
        .value("NONE", PriorKind::None)
        .value("LAPLACE", PriorKind::Laplace)
        .value("NORMAL", PriorKind::Normal)
        .value("BAR_UPDATE", PriorKind::BarUpdate)
        .value("JEFFREYS", PriorKind::Jeffreys);

    py::enum_<ConvergenceKind>(m, "ConvergenceKind")
        .value("GRADIENT", ConvergenceKind::Gradient)
        .value("LANGE", ConvergenceKind::Lange)
        .value("MITTAL", ConvergenceKind::Mittal)
        .value("ONE_STEP", ConvergenceKind::OneStep)
        .value("ZHANG_OLES", ConvergenceKind::ZhangOles);

    py::enum_<SelectorKind>(m, "SelectorKind")
        .value("AUTO", SelectorKind::Auto)
        .value("BY_PID", SelectorKind::ByPid)
        .value("BY_ROW", SelectorKind::ByRow);

    py::enum_<AlgorithmKind>(m, "AlgorithmKind")
        .value("CCD", AlgorithmKind::Ccd)
        .value("MM", AlgorithmKind::Mm);

    py::enum_<NormalizationKind>(m, "NormalizationKind")
        .value("STANDARD_DEVIATION", NormalizationKind::StandardDeviation)
        .value("MAX", NormalizationKind::Max)
        .value("MEDIAN", NormalizationKind::Median)
        .value("Q95", NormalizationKind::Q95);

    py::enum_<NoiseLevel>(m, "NoiseLevel")
        .value("SILENT", NoiseLevel::Silent)
        .value("QUIET", NoiseLevel::Quiet)
        .value("NOISY", NoiseLevel::Noisy);

    py::enum_<Precision>(m, "Precision")
        .value("FP64", Precision::Fp64)
        .value("FP32", Precision::Fp32);

    py::enum_<ColumnFormat>(m, "ColumnFormat")
        .value("DENSE", ColumnFormat::Dense)
        .value("SPARSE", ColumnFormat::Sparse)
        .value("INDICATOR", ColumnFormat::Indicator)
        .value("INTERCEPT", ColumnFormat::Intercept);

    m.attr("USE_TIME_AS_OFFSET") = py::int_(kUseTimeAsOffset);

    // -- option / result aggregates -----------------------------------------

    py::class_<PriorOptions>(m, "PriorOptions")
        .def(py::init<>())
        .def_readwrite("kind", &PriorOptions::kind)
        .def_readwrite("variance", &PriorOptions::variance)
        .def_readwrite("use_cross_validation", &PriorOptions::use_cross_validation)
        .def_readwrite("exclude", &PriorOptions::exclude)
        .def_readwrite("force_intercept", &PriorOptions::force_intercept)
        .def_readwrite("kinds", &PriorOptions::kinds)
        .def_readwrite("variances", &PriorOptions::variances);

    py::class_<FitOptions>(m, "FitOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &FitOptions::max_iterations)
        .def_readwrite("tolerance", &FitOptions::tolerance)
        .def_readwrite("convergence", &FitOptions::convergence)
        .def_readwrite("algorithm", &FitOptions::algorithm)
        .def_readwrite("initial_bound", &FitOptions::initial_bound)
        .def_readwrite("max_bound_count", &FitOptions::max_bound_count)
        .def_readwrite("use_kkt_swindle", &FitOptions::use_kkt_swindle)
        .def_readwrite("swindle_multiplier", &FitOptions::swindle_multiplier)
        .def_readwrite("do_it_all", &FitOptions::do_it_all)
        .def_readwrite("auto_search", &FitOptions::auto_search)
        .def_readwrite("fold", &FitOptions::fold)
        .def_readwrite("cv_repetitions", &FitOptions::cv_repetitions)
        .def_readwrite("lower_limit", &FitOptions::lower_limit)
        .def_readwrite("upper_limit", &FitOptions::upper_limit)
        .def_readwrite("grid_steps", &FitOptions::grid_steps)
        .def_readwrite("starting_variance", &FitOptions::starting_variance)
        .def_readwrite("selector", &FitOptions::selector)
        .def_readwrite("min_cv_data", &FitOptions::min_cv_data)
        .def_readwrite("sync_cv", &FitOptions::sync_cv)
        .def_readwrite("noise", &FitOptions::noise)
        .def_readwrite("threads", &FitOptions::threads)
        .def_readwrite("seed", &FitOptions::seed)
        .def_readwrite("reset_coefficients", &FitOptions::reset_coefficients)
        .def_readwrite("retry_on_poor_blr_step",
                       &FitOptions::retry_on_poor_blr_step);

    py::class_<FitResult>(m, "FitResult")
        .def_property_readonly("covariate_ids",
            [](const FitResult& self) { return toArray(self.covariate_ids); })
        .def_property_readonly("coefficients",
            [](const FitResult& self) { return toArray(self.coefficients); })
        .def_readonly("log_likelihood", &FitResult::log_likelihood)
        .def_readonly("log_prior", &FitResult::log_prior)
        .def_readonly("return_flag", &FitResult::return_flag)
        .def_readonly("iterations", &FitResult::iterations)
        .def_readonly("prior_info", &FitResult::prior_info)
        .def_readonly("variance", &FitResult::variance)
        .def_readonly("cross_validation_info", &FitResult::cross_validation_info)
        .def_readonly("covariate_count", &FitResult::covariate_count)
        .def_readonly("fit_seconds", &FitResult::fit_seconds)
        .def_property_readonly("converged", &FitResult::converged);

    py::class_<ProfileInterval>(m, "ProfileInterval")
        .def_readonly("covariate_id", &ProfileInterval::covariate_id)
        .def_readonly("lower", &ProfileInterval::lower)
        .def_readonly("upper", &ProfileInterval::upper)
        .def_readonly("evaluations", &ProfileInterval::evaluations);

    py::class_<ProfileCurve>(m, "ProfileCurve")
        .def_property_readonly("points",
            [](const ProfileCurve& self) { return toArray(self.points); })
        .def_property_readonly("values",
            [](const ProfileCurve& self) { return toArray(self.values); })
        .def_property_readonly("derivatives",
            [](const ProfileCurve& self) { return toArray(self.derivatives); });

    // -- ModelData ----------------------------------------------------------

    py::class_<ModelData>(m, "ModelData")
        .def_static("create", &ModelData::create,
                    py::arg("kind"),
                    py::arg("precision") = Precision::Fp64,
                    py::arg("silent") = true)
        .def("set_outcome", &ModelData::set_outcome,
             py::arg("y"),
             py::arg("time") = std::vector<double>{},
             py::arg("stratum_id") = std::vector<std::int64_t>{},
             py::arg("row_id") = std::vector<std::int64_t>{},
             py::call_guard<py::gil_scoped_release>())
        .def("add_covariates_csc", &ModelData::add_covariates_csc,
             py::arg("indptr"), py::arg("row_indices"), py::arg("values"),
             py::arg("covariate_ids"), py::arg("force_sparse") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("add_dense_covariate", &ModelData::add_dense_covariate,
             py::arg("covariate_id"), py::arg("values"),
             py::call_guard<py::gil_scoped_release>())
        .def("add_intercept", &ModelData::add_intercept)
        .def("set_offset_covariate", &ModelData::set_offset_covariate,
             py::arg("covariate_id"), py::arg("already_on_log_scale") = false)
        .def("make_dense", &ModelData::make_dense, py::arg("covariate_ids"))
        .def("normalize",
             [](ModelData& self, NormalizationKind kind) {
                 return toArray(self.normalize(kind));
             }, py::arg("kind"))
        .def("finalize", &ModelData::finalize)
        .def_property_readonly("is_finalized", &ModelData::is_finalized)
        .def_property_readonly("kind", &ModelData::kind)
        .def_property_readonly("precision", &ModelData::precision)
        .def_property_readonly("row_count", &ModelData::row_count)
        .def_property_readonly("covariate_count", &ModelData::covariate_count)
        .def_property_readonly("stratum_count", &ModelData::stratum_count)
        .def_property_readonly("has_intercept", &ModelData::has_intercept)
        .def_property_readonly("has_offset", &ModelData::has_offset)
        .def_property_readonly("intercept_label", &ModelData::intercept_label)
        .def_property_readonly("covariate_ids",
            [](const ModelData& self) { return toArray(self.covariate_ids()); })
        .def_property_readonly("covariate_formats", &ModelData::covariate_formats)
        .def_property_readonly("outcome",
            [](const ModelData& self) { return toArray(self.outcome()); })
        .def_property_readonly("time",
            [](const ModelData& self) { return toArray(self.time()); })
        .def_property_readonly("stratum_index",
            [](const ModelData& self) { return toArray(self.stratum_index()); })
        .def("univariable_correlation",
             [](const ModelData& self, const std::vector<std::int64_t>& ids) {
                 return toArray(self.univariable_correlation(ids));
             }, py::arg("ids") = std::vector<std::int64_t>{})
        .def("column_sum", &ModelData::column_sum,
             py::arg("covariate_id"), py::arg("power") = 1)
        .def("sum_by_stratum",
             [](const ModelData& self, std::int64_t id, int power) {
                 return toArray(self.sum_by_stratum(id, power));
             }, py::arg("covariate_id"), py::arg("power") = 1);

    // -- Model --------------------------------------------------------------

    py::class_<Model>(m, "Model")
        // keep_alive: Cyclops holds the ModelData by reference for the lifetime
        // of the optimizer.
        .def_static("create", &Model::create,
                    py::arg("data"), py::arg("compute_device") = "native",
                    py::keep_alive<0, 1>())
        .def("set_prior", &Model::set_prior, py::arg("prior"))
        .def("set_options", &Model::set_options, py::arg("options"))
        .def("set_weights", &Model::set_weights, py::arg("weights"))
        .def("set_censor_weights", &Model::set_censor_weights, py::arg("weights"))
        .def("set_start_values", &Model::set_start_values, py::arg("beta"))
        .def("set_fixed", &Model::set_fixed, py::arg("fixed"))
        .def("fit", &Model::fit, py::call_guard<py::gil_scoped_release>())
        .def("predict",
             [](const Model& self) {
                 std::vector<double> values;
                 {
                     py::gil_scoped_release release;
                     values = self.predict();
                 }
                 return toArray(values);
             })
        .def_property_readonly("log_likelihood", &Model::log_likelihood)
        .def_property_readonly("log_prior", &Model::log_prior)
        .def_property_readonly("coefficients",
            [](const Model& self) { return toArray(self.coefficients()); })
        .def("gradient",
             [](const Model& self) { return toArray(self.gradient()); })
        .def("hessian_diagonal",
             [](const Model& self, const std::vector<std::int64_t>& ids) {
                 return toArray(self.hessian_diagonal(ids));
             }, py::arg("ids"))
        .def("standard_errors",
             [](const Model& self, const std::vector<std::int64_t>& ids) {
                 return toArray(self.standard_errors(ids));
             }, py::arg("ids") = std::vector<std::int64_t>{})
        .def("fisher_information",
             [](const Model& self, const std::vector<std::int64_t>& ids) {
                 const auto flat = self.fisher_information(ids);
                 // Square by construction; derive the side from the payload
                 // rather than from `ids`, which is empty for "all covariates".
                 const auto n = static_cast<py::ssize_t>(
                     std::llround(std::sqrt(static_cast<double>(flat.size()))));
                 auto array = toArray(flat);
                 array.resize({n, n});
                 return array;
             }, py::arg("ids") = std::vector<std::int64_t>{})
        .def("is_regularized", &Model::is_regularized)
        .def("profile", &Model::profile,
             py::arg("ids"), py::arg("threads") = 1,
             py::arg("threshold") = 1.920729,
             py::arg("include_penalty") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("profile_curve", &Model::profile_curve,
             py::arg("covariate_id"), py::arg("points"), py::arg("threads") = 1,
             py::arg("include_penalty") = false,
             py::arg("with_derivatives") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("take_log", &Model::take_log);

    // -- free functions -----------------------------------------------------

    m.def("version", &version);
    m.def("list_gpu_devices", &list_gpu_devices);
    m.def("removes_intercept", &removes_intercept, py::arg("kind"));
    m.def("requires_strata", &requires_strata, py::arg("kind"));
    m.def("requires_time", &requires_time, py::arg("kind"));
    m.def("requires_offset", &requires_offset, py::arg("kind"));
}
