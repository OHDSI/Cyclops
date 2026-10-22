/*
 * CyclopsApi.h
 *
 * A small, stable, language-neutral C++ facade over the Cyclops implementation.
 *
 * Design constraints:
 *   - No Rcpp, no JNI, no Python, no Eigen in this header.
 *   - Only standard-library types and POD structs cross the boundary.
 *   - No `bsccs` type is named here; implementation classes stay hidden (pImpl).
 *   - Errors are reported as exceptions derived from std::runtime_error.
 *
 * The facade exists so that a binding layer for any host language is mechanical:
 * translate host containers to std::vector, call, translate back.
 */

#ifndef CYCLOPS_API_CYCLOPSAPI_H_
#define CYCLOPS_API_CYCLOPSAPI_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyclops {
namespace api {

// ---------------------------------------------------------------------------
// Enumerations (mirror bsccs enums, but are part of the stable API surface)
// ---------------------------------------------------------------------------

enum class ModelKind {
    Normal,
    Poisson,
    Logistic,
    ConditionalLogistic,
    TiedConditionalLogistic,
    EfronConditionalLogistic,
    ConditionalPoisson,
    SelfControlledCaseSeries,
    Cox,
    CoxRaw,
    TimeVaryingCox,
    FineGray
};

enum class PriorKind { None, Laplace, Normal, BarUpdate, Jeffreys };

enum class ConvergenceKind { Gradient, Lange, Mittal, OneStep, ZhangOles };

enum class SelectorKind { Auto, ByPid, ByRow };

enum class AlgorithmKind { Ccd, Mm };

enum class NormalizationKind { StandardDeviation, Max, Median, Q95 };

enum class NoiseLevel { Silent, Quiet, Noisy };

enum class Precision { Fp64, Fp32 };

enum class ColumnFormat { Dense, Sparse, Indicator, Intercept };

/// Sentinel accepted by `ModelData::set_offset_covariate` meaning "promote the
/// outcome `time` vector to an offset covariate".
constexpr std::int64_t kUseTimeAsOffset = -1;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every failure originating inside Cyclops surfaces as this type.
class CyclopsError : public std::runtime_error {
public:
    explicit CyclopsError(const std::string& what) : std::runtime_error(what) {}
};

// ---------------------------------------------------------------------------
// Option / result aggregates
// ---------------------------------------------------------------------------

/// Regularisation specification.
///
/// Cyclops parameterises Laplace/Normal priors by *variance*; the Laplace rate
/// is `sqrt(2 / variance)`. `variance == 0` means "fix the coefficient at zero".
struct PriorOptions {
    PriorKind kind = PriorKind::None;
    double variance = 1.0;

    /// Tune `variance` by cross-validation instead of using it directly.
    bool use_cross_validation = false;

    /// Covariate ids left unpenalised.
    std::vector<std::int64_t> exclude;

    /// When false (the default) and the design has an intercept, the intercept
    /// is added to `exclude` automatically. Matches `fitCyclopsModel()`.
    bool force_intercept = false;

    /// Optional per-covariate override. When non-empty both must have length
    /// equal to the number of covariates and take precedence over
    /// `kind`/`variance`.
    std::vector<PriorKind> kinds;
    std::vector<double> variances;
};

/// Mode-finding, cross-validation and runtime controls.
///
/// Defaults are deliberately identical to R's `createControl()` so that a fit
/// issued from Python and one issued from R follow the same iterate path.
struct FitOptions {
    // -- mode finding --
    int max_iterations = 1000;
    double tolerance = 1e-6;
    ConvergenceKind convergence = ConvergenceKind::Gradient;
    AlgorithmKind algorithm = AlgorithmKind::Ccd;
    double initial_bound = 2.0;
    int max_bound_count = 5;
    bool use_kkt_swindle = false;
    int swindle_multiplier = 10;
    bool do_it_all = true;

    // -- cross-validation --
    bool auto_search = true;   ///< false ⇒ fixed grid search
    int fold = 10;
    int cv_repetitions = 1;
    double lower_limit = 0.01;
    double upper_limit = 20.0;
    int grid_steps = 10;
    double starting_variance = -1.0;  ///< < 0 ⇒ Genkins et al. default
    SelectorKind selector = SelectorKind::Auto;
    int min_cv_data = 100;
    bool sync_cv = false;

    // -- runtime --
    NoiseLevel noise = NoiseLevel::Silent;
    int threads = 1;
    /// Cross-validation fold seed. `CCDArguments::seed` is a `long`, which is
    /// 32-bit on Windows, so callers must keep this inside the signed 32-bit
    /// range for a given seed to select the same folds on every platform.
    /// 0 ⇒ the binding should substitute a time-based seed.
    long seed = 0;
    bool reset_coefficients = false;

    /// Reproduce `fitCyclopsModel()`'s behaviour of retrying with the Lange
    /// convergence criterion when a gradient-based fit reports POOR_BLR_STEP.
    bool retry_on_poor_blr_step = true;
};

/// Everything the R `cyclopsFit` object reports, plus the coefficient vector.
struct FitResult {
    std::vector<std::int64_t> covariate_ids;
    std::vector<double> coefficients;

    double log_likelihood = 0.0;
    double log_prior = 0.0;
    std::string return_flag;   ///< SUCCESS | MAX_ITERATIONS | ILLCONDITIONED | ...
    int iterations = 0;
    std::string prior_info;
    std::vector<double> variance;   ///< hyperparameter(s) actually used
    std::string cross_validation_info;
    int covariate_count = 0;
    double fit_seconds = 0.0;

    bool converged() const { return return_flag == "SUCCESS"; }
};

/// One likelihood-profile confidence interval.
struct ProfileInterval {
    std::int64_t covariate_id = 0;
    double lower = 0.0;
    double upper = 0.0;
    int evaluations = 0;
};

/// Result of evaluating the profile likelihood on a grid of points.
struct ProfileCurve {
    std::vector<double> points;
    std::vector<double> values;
    std::vector<double> derivatives;   ///< empty unless requested
};

// ---------------------------------------------------------------------------
// ModelData
// ---------------------------------------------------------------------------

class Model;

/// Outcome + design matrix in the layout Cyclops' kernels expect.
///
/// Usage order matters and mirrors the R `createCyclopsData` /
/// `loadNewSqlData` / `finalizeSqlCyclopsData` sequence:
///
///   1. `set_outcome(...)`             — establishes the row count and strata
///   2. `add_covariates_csc(...)` etc. — one or more calls
///   3. `add_intercept()` / `set_offset_covariate(...)` — optional
///   4. `finalize()`
class ModelData {
public:
    static std::unique_ptr<ModelData> create(ModelKind kind,
                                             Precision precision = Precision::Fp64,
                                             bool silent = true);

    ~ModelData();

    ModelData(const ModelData&) = delete;
    ModelData& operator=(const ModelData&) = delete;

    // -- loading ------------------------------------------------------------

    /// @param y            outcome, length = number of rows
    /// @param time         survival/exposure time; empty when unused
    /// @param stratum_id   stratum per row, ascending and grouped; empty ⇒ unstratified
    /// @param row_id       external row identifiers; empty ⇒ row indices are 0..n-1
    ///                     and `add_*` calls address rows positionally
    void set_outcome(const std::vector<double>& y,
                     const std::vector<double>& time = {},
                     const std::vector<std::int64_t>& stratum_id = {},
                     const std::vector<std::int64_t>& row_id = {});

    /// Append covariates supplied in compressed-sparse-column layout.
    ///
    /// @param indptr         length = n_columns + 1 (as in scipy.sparse.csc_matrix)
    /// @param row_indices    length = nnz, ascending within each column
    /// @param values         length = nnz, or empty for a pure indicator matrix
    /// @param covariate_ids  length = n_columns; ids must be unique and unused
    /// @param force_sparse   store 0/1 columns as SPARSE rather than INDICATOR
    ///
    /// Columns whose values are all in {0, 1} are stored as INDICATOR (indices
    /// only). Explicit zeros are dropped, matching Cyclops' loaders.
    void add_covariates_csc(const std::vector<std::int64_t>& indptr,
                            const std::vector<std::int64_t>& row_indices,
                            const std::vector<double>& values,
                            const std::vector<std::int64_t>& covariate_ids,
                            bool force_sparse = false);

    /// Append a single fully dense covariate (length = number of rows).
    void add_dense_covariate(std::int64_t covariate_id,
                             const std::vector<double>& values);

    /// Prepend an intercept column (label 0 by convention).
    void add_intercept();

    /// Move a covariate to column 0 and mark it as a fixed-coefficient offset.
    /// Pass `kUseTimeAsOffset` to promote the `time` vector instead.
    void set_offset_covariate(std::int64_t covariate_id,
                              bool already_on_log_scale = false);

    /// Convert the named covariates to dense storage.
    void make_dense(const std::vector<std::int64_t>& covariate_ids);

    /// Rescale every DENSE and SPARSE covariate (INDICATOR, INTERCEPT and
    /// offset columns are left alone) and return the factor applied to each,
    /// in stored column order.
    ///
    /// Columns are *multiplied* by the returned factor — `1/sd` for
    /// `StandardDeviation`, and so on — so a coefficient fitted afterwards is
    /// returned to the original scale by multiplying it by the same factor.
    /// This matches `coef(fit, rescale = TRUE)` in R.
    std::vector<double> normalize(NormalizationKind kind);

    void finalize();

    // -- introspection ------------------------------------------------------

    bool is_finalized() const;
    ModelKind kind() const;
    Precision precision() const;
    std::size_t row_count() const;
    std::size_t covariate_count() const;
    std::size_t stratum_count() const;
    bool has_intercept() const;
    bool has_offset() const;
    std::int64_t intercept_label() const;

    std::vector<std::int64_t> covariate_ids() const;
    std::vector<ColumnFormat> covariate_formats() const;
    std::vector<double> outcome() const;
    std::vector<double> time() const;
    std::vector<int> stratum_index() const;


    /// Pearson correlation of each covariate with the outcome. Empty `ids`
    /// means "all covariates".
    std::vector<double> univariable_correlation(
        const std::vector<std::int64_t>& ids = {}) const;

    /// Sum of `x^power` over a covariate column. Throws for an unknown id;
    /// there is no way to reduce over the outcome, which `outcome()` returns.
    double column_sum(std::int64_t covariate_id, int power = 1) const;

    /// Per-stratum sums of `x^power`.
    std::vector<double> sum_by_stratum(std::int64_t covariate_id,
                                       int power = 1) const;

private:
    ModelData();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Model;
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A configured optimizer bound to a `ModelData`.
///
/// The `ModelData` must outlive the `Model` (Cyclops holds it by reference).
class Model {
public:
    /// @param compute_device "native" for CPU; a device name selects a GPU build
    static std::unique_ptr<Model> create(ModelData& data,
                                         const std::string& compute_device = "native");

    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // -- configuration ------------------------------------------------------

    void set_prior(const PriorOptions& prior);
    void set_options(const FitOptions& options);

    /// Row weights; length = number of rows. Zero excludes a row.
    void set_weights(const std::vector<double>& weights);

    /// Fine-Gray subject-specific censoring weights; each in [0, 1].
    void set_censor_weights(const std::vector<double>& weights);

    /// Starting coefficients, one per *estimated* coefficient — that is, one per
    /// covariate excluding any offset column, matching `coefficients()`. When
    /// not called, `fit()` applies the same intercept warm start as
    /// `fitCyclopsModel()`.
    void set_start_values(const std::vector<double>& beta);

    /// Hold coefficients at their starting value. Sized as `set_start_values`.
    void set_fixed(const std::vector<bool>& fixed);

    // -- fitting ------------------------------------------------------------

    /// Runs cross-validation first when the prior requests it, then fits.
    FitResult fit();

    // -- inference ----------------------------------------------------------

    /// E[y | X] on the response scale (probability, rate, or risk score).
    std::vector<double> predict() const;

    double log_likelihood() const;
    double log_prior() const;

    /// Coefficients in `covariate_ids()` order, offset column excluded.
    std::vector<double> coefficients() const;

    /// ∂ log L / ∂β for every non-offset coefficient.
    std::vector<double> gradient() const;

    /// Diagonal of the log-likelihood Hessian for the named ids, so entries are
    /// negative at a maximum. Negate for the observed information. Matches R's
    /// `.cyclopsGetLogLikelihoodHessianDiagonal`.
    std::vector<double> hessian_diagonal(const std::vector<std::int64_t>& ids) const;

    /// Asymptotic standard errors from the inverse Fisher information.
    std::vector<double> standard_errors(const std::vector<std::int64_t>& ids) const;

    /// Row-major Fisher information for the named ids (empty ⇒ all covariates).
    std::vector<double> fisher_information(
        const std::vector<std::int64_t>& ids) const;

    /// Which coefficients the prior actually penalises.
    std::vector<bool> is_regularized() const;

    /// Adaptive-bisection likelihood-profile confidence intervals.
    std::vector<ProfileInterval> profile(const std::vector<std::int64_t>& ids,
                                         int threads = 1,
                                         double threshold = 1.920729,
                                         bool include_penalty = false);

    /// Profile log-likelihood evaluated at supplied values of one coefficient.
    ProfileCurve profile_curve(std::int64_t covariate_id,
                               const std::vector<double>& points,
                               int threads = 1,
                               bool include_penalty = false,
                               bool with_derivatives = false);

    /// Diagnostics emitted while fitting (empty when `noise == Silent`).
    std::vector<std::string> take_log();

private:
    Model();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Cyclops version string, e.g. "3.7.1".
std::string version();

/// Names of GPU devices visible to this build; empty for a CPU-only build.
std::vector<std::string> list_gpu_devices();

/// True when `kind` drops the intercept because it is absorbed by strata.
bool removes_intercept(ModelKind kind);

/// True when `kind` needs `stratum_id`.
bool requires_strata(ModelKind kind);

/// True when `kind` needs `time`.
bool requires_time(ModelKind kind);

/// True when `kind` needs an offset covariate.
bool requires_offset(ModelKind kind);

}  // namespace api
}  // namespace cyclops

#endif  // CYCLOPS_API_CYCLOPSAPI_H_
