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
#include "cyclops/FitterPresets.h"

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
        .value("ZHANG_OLES", ConvergenceKind::ZhangOles)
        .value("KKT", ConvergenceKind::Kkt)
        .value("GAP", ConvergenceKind::Gap);

    py::enum_<SelectorKind>(m, "SelectorKind")
        .value("AUTO", SelectorKind::Auto)
        .value("BY_PID", SelectorKind::ByPid)
        .value("BY_ROW", SelectorKind::ByRow);

    py::enum_<AlgorithmKind>(m, "AlgorithmKind")
        .value("CCD", AlgorithmKind::Ccd)
        .value("MM", AlgorithmKind::Mm)
        .value("AUTO", AlgorithmKind::Auto)
        .value("NEWTON", AlgorithmKind::Newton);

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
        .def_readwrite("fitter", &FitOptions::fitter)
        .def_readwrite("do_it_all", &FitOptions::do_it_all)
        .def_readwrite("active_set_every", &FitOptions::active_set_every)
        .def_readwrite("active_set_threshold", &FitOptions::active_set_threshold)
        .def_readwrite("adaptive_intercept", &FitOptions::adaptive_intercept)
        .def_readwrite("merge_duplicates", &FitOptions::merge_duplicates)
        .def_readwrite("fallback_convergence", &FitOptions::fallback_convergence)
        .def_readwrite("fallback_tolerance", &FitOptions::fallback_tolerance)
        .def_readwrite("fallback_max_iterations", &FitOptions::fallback_max_iterations)
        .def_readwrite("cv_search_direction", &FitOptions::cv_search_direction)
        .def_readwrite("cv_convergence", &FitOptions::cv_convergence)
        .def_readwrite("cv_tolerance", &FitOptions::cv_tolerance)
        .def_readwrite("cv_max_iterations", &FitOptions::cv_max_iterations)
        .def_readwrite("cv_min_iterations", &FitOptions::cv_min_iterations)
        .def_readwrite("cv_warm_start", &FitOptions::cv_warm_start)
        .def_readwrite("cv_refine", &FitOptions::cv_refine)
        .def_readwrite("cv_null_model_bound", &FitOptions::cv_null_model_bound)
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
        .def_property_readonly("convergence", [](const FitResult& self) {
            py::dict d;
            for (const auto& [k, v] : self.convergence) d[py::str(k)] = v;
            for (const auto& [k, v] : self.convergence_text) d[py::str(k)] = v;
            return d;
        })
        .def_readonly("aliased_groups", &FitResult::aliased_groups)
        .def_readonly("divergent_coefficients", &FitResult::divergent_coefficients)
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
    // The fitter's presets for the settings a control leaves unset (FitterPresets.cpp, shared with R's
    // createControl()): None is unset. Returns the resolved settings (None where fold fits follow the fit), the
    // warnings to issue and an error message (empty when the settings are consistent)
    m.def("resolve_control",
          [](const std::string& fitter, std::optional<std::string> model_type, std::optional<int> n_covariates,
             bool reset_coefficients, std::optional<std::string> algorithm, std::optional<std::string> convergence,
             std::optional<double> tolerance, std::optional<int> max_iterations,
             std::optional<std::string> cv_convergence, std::optional<double> cv_tolerance,
             std::optional<int> cv_max_iterations, std::optional<int> cv_min_iterations,
             std::optional<std::string> cv_search_direction, std::optional<bool> cv_refine,
             std::optional<bool> cv_null_model_bound, std::optional<bool> cv_warm_start,
             std::optional<int> active_set_every, std::optional<double> active_set_threshold,
             std::optional<bool> adaptive_intercept) {
              bsccs::ControlRequest q;
              q.fitter = fitter;
              q.modelType = model_type;
              q.nCovariates = n_covariates;
              q.resetCoefficients = reset_coefficients;
              q.algorithm = algorithm;
              q.convergenceType = convergence;
              q.tolerance = tolerance;
              q.maxIterations = max_iterations;
              q.cvConvergenceType = cv_convergence;
              q.cvTolerance = cv_tolerance;
              q.cvMaxIterations = cv_max_iterations;
              q.cvMinIterations = cv_min_iterations;
              q.cvSearchDirection = cv_search_direction;
              q.cvRefine = cv_refine;
              q.cvNullModelBound = cv_null_model_bound;
              q.cvWarmStart = cv_warm_start;
              q.activeSetEvery = active_set_every;
              q.activeSetThreshold = active_set_threshold;
              q.adaptiveIntercept = adaptive_intercept;
              const bsccs::ResolvedControl r = bsccs::resolveControl(q);
              py::dict d;
              d["algorithm"] = r.algorithm;
              d["convergence"] = r.convergenceType;
              d["tolerance"] = r.tolerance;
              d["max_iterations"] = r.maxIterations;
              d["cv_convergence"] = r.cvConvergenceType;
              d["cv_tolerance"] = r.cvTolerance;
              d["cv_max_iterations"] = r.cvMaxIterations;
              d["cv_min_iterations"] = r.cvMinIterations;
              d["cv_search_direction"] = r.cvSearchDirection;
              d["cv_refine"] = r.cvRefine;
              d["cv_null_model_bound"] = r.cvNullModelBound;
              d["cv_warm_start"] = r.cvWarmStart;
              d["active_set_every"] = r.activeSetEvery;
              d["active_set_threshold"] = r.activeSetThreshold;
              d["adaptive_intercept"] = r.adaptiveIntercept;
              d["fallback_convergence"] = r.fallbackConvergenceType;
              d["fallback_tolerance"] = r.fallbackTolerance;
              d["fallback_max_iterations"] = r.fallbackMaxIterations;
              d["warnings"] = r.warnings;
              d["error"] = r.error;
              return d;
          },
          py::arg("fitter"), py::arg("model_type"), py::arg("n_covariates"), py::arg("reset_coefficients"),
          py::arg("algorithm") = py::none(), py::arg("convergence") = py::none(), py::arg("tolerance") = py::none(),
          py::arg("max_iterations") = py::none(), py::arg("cv_convergence") = py::none(),
          py::arg("cv_tolerance") = py::none(), py::arg("cv_max_iterations") = py::none(),
          py::arg("cv_min_iterations") = py::none(), py::arg("cv_search_direction") = py::none(),
          py::arg("cv_refine") = py::none(), py::arg("cv_null_model_bound") = py::none(),
          py::arg("cv_warm_start") = py::none(), py::arg("active_set_every") = py::none(),
          py::arg("active_set_threshold") = py::none(), py::arg("adaptive_intercept") = py::none());
    m.def("list_gpu_devices", &list_gpu_devices);
    m.def("removes_intercept", &removes_intercept, py::arg("kind"));
    m.def("requires_strata", &requires_strata, py::arg("kind"));
    m.def("requires_time", &requires_time, py::arg("kind"));
    m.def("requires_offset", &requires_offset, py::arg("kind"));
}
