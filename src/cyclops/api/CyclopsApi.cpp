/*
 * CyclopsApi.cpp
 *
 * Implementation of the language-neutral Cyclops facade.
 *
 * This is the only place where the stable API types meet `bsccs` internals. The
 * translation is deliberately thin: nothing here reimplements Cyclops, it only
 * sequences existing calls in the same order the R package does (see
 * `R/ModelFit.R::fitCyclopsModel`), so that a fit issued through this facade and
 * one issued from R follow an identical iterate path.
 */

#include "cyclops/api/CyclopsApi.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <utility>

#include <Eigen/Dense>

#include "cyclops/CcdInterface.h"
#include "cyclops/CyclicCoordinateDescent.h"
#include "cyclops/ModelData.h"
#include "cyclops/Types.h"
#include "cyclops/api/BufferedLogger.h"
#include "cyclops/engine/AbstractModelSpecifics.h"
#include "cyclops/priors/CovariatePrior.h"
#include "cyclops/priors/JointPrior.h"

namespace cyclops {
namespace api {
namespace {

// ---------------------------------------------------------------------------
// Enum translation
// ---------------------------------------------------------------------------

bsccs::ModelType toModelType(ModelKind kind) {
    switch (kind) {
        case ModelKind::Normal:                   return bsccs::ModelType::NORMAL;
        case ModelKind::Poisson:                  return bsccs::ModelType::POISSON;
        case ModelKind::Logistic:                 return bsccs::ModelType::LOGISTIC;
        case ModelKind::ConditionalLogistic:      return bsccs::ModelType::CONDITIONAL_LOGISTIC;
        case ModelKind::TiedConditionalLogistic:  return bsccs::ModelType::TIED_CONDITIONAL_LOGISTIC;
        case ModelKind::EfronConditionalLogistic: return bsccs::ModelType::EFRON_CONDITIONAL_LOGISTIC;
        case ModelKind::ConditionalPoisson:       return bsccs::ModelType::CONDITIONAL_POISSON;
        case ModelKind::SelfControlledCaseSeries: return bsccs::ModelType::SELF_CONTROLLED_MODEL;
        case ModelKind::Cox:                      return bsccs::ModelType::COX;
        case ModelKind::CoxRaw:                   return bsccs::ModelType::COX_RAW;
        case ModelKind::TimeVaryingCox:           return bsccs::ModelType::TIME_VARYING_COX;
        case ModelKind::FineGray:                 return bsccs::ModelType::FINE_GRAY;
    }
    throw CyclopsError("Unknown model kind");
}

ModelKind fromModelType(bsccs::ModelType type) {
    switch (type) {
        case bsccs::ModelType::NORMAL:                     return ModelKind::Normal;
        case bsccs::ModelType::POISSON:                    return ModelKind::Poisson;
        case bsccs::ModelType::LOGISTIC:                   return ModelKind::Logistic;
        case bsccs::ModelType::CONDITIONAL_LOGISTIC:       return ModelKind::ConditionalLogistic;
        case bsccs::ModelType::TIED_CONDITIONAL_LOGISTIC:  return ModelKind::TiedConditionalLogistic;
        case bsccs::ModelType::EFRON_CONDITIONAL_LOGISTIC: return ModelKind::EfronConditionalLogistic;
        case bsccs::ModelType::CONDITIONAL_POISSON:        return ModelKind::ConditionalPoisson;
        case bsccs::ModelType::SELF_CONTROLLED_MODEL:      return ModelKind::SelfControlledCaseSeries;
        case bsccs::ModelType::COX:                        return ModelKind::Cox;
        case bsccs::ModelType::COX_RAW:                    return ModelKind::CoxRaw;
        case bsccs::ModelType::TIME_VARYING_COX:           return ModelKind::TimeVaryingCox;
        case bsccs::ModelType::FINE_GRAY:                  return ModelKind::FineGray;
        default: throw CyclopsError("Unknown model type");
    }
}

bsccs::priors::PriorType toPriorType(PriorKind kind) {
    switch (kind) {
        case PriorKind::None:      return bsccs::priors::NONE;
        case PriorKind::Laplace:   return bsccs::priors::LAPLACE;
        case PriorKind::Normal:    return bsccs::priors::NORMAL;
        case PriorKind::BarUpdate: return bsccs::priors::BAR_UPDATE;
        case PriorKind::Jeffreys:  return bsccs::priors::JEFFREYS;
    }
    throw CyclopsError("Unknown prior kind");
}

int toConvergenceType(ConvergenceKind kind) {
    switch (kind) {
        case ConvergenceKind::Gradient:  return bsccs::GRADIENT;
        case ConvergenceKind::Lange:     return bsccs::LANGE;
        case ConvergenceKind::Mittal:    return bsccs::MITTAL;
        case ConvergenceKind::OneStep:   return bsccs::ONE_STEP;
        case ConvergenceKind::ZhangOles: return bsccs::ZHANG_OLES;
    }
    throw CyclopsError("Unknown convergence kind");
}

bsccs::SelectorType toSelectorType(SelectorKind kind) {
    switch (kind) {
        case SelectorKind::Auto:  return bsccs::SelectorType::DEFAULT;
        case SelectorKind::ByPid: return bsccs::SelectorType::BY_PID;
        case SelectorKind::ByRow: return bsccs::SelectorType::BY_ROW;
    }
    throw CyclopsError("Unknown selector kind");
}

bsccs::NoiseLevels toNoiseLevel(NoiseLevel level) {
    switch (level) {
        case NoiseLevel::Silent: return bsccs::SILENT;
        case NoiseLevel::Quiet:  return bsccs::QUIET;
        case NoiseLevel::Noisy:  return bsccs::NOISY;
    }
    throw CyclopsError("Unknown noise level");
}

bsccs::NormalizationType toNormalizationType(NormalizationKind kind) {
    switch (kind) {
        case NormalizationKind::StandardDeviation: return bsccs::NormalizationType::STANDARD_DEVIATION;
        case NormalizationKind::Max:               return bsccs::NormalizationType::MAX;
        case NormalizationKind::Median:            return bsccs::NormalizationType::MEDIAN;
        case NormalizationKind::Q95:               return bsccs::NormalizationType::Q95;
    }
    throw CyclopsError("Unknown normalization kind");
}

ColumnFormat fromFormatType(bsccs::FormatType type) {
    switch (type) {
        case bsccs::DENSE:     return ColumnFormat::Dense;
        case bsccs::SPARSE:    return ColumnFormat::Sparse;
        case bsccs::INDICATOR: return ColumnFormat::Indicator;
        case bsccs::INTERCEPT: return ColumnFormat::Intercept;
    }
    throw CyclopsError("Unknown column format");
}

std::string returnFlagString(bsccs::UpdateReturnFlags flag) {
    // Mirrors DiagnosticsOutputWriter::returnFlagString so that `return_flag`
    // strings are identical to the R package's.
    switch (flag) {
        case bsccs::SUCCESS:            return "SUCCESS";
        case bsccs::MAX_ITERATIONS:     return "MAX_ITERATIONS";
        case bsccs::ILLCONDITIONED:     return "ILLCONDITIONED";
        case bsccs::MISSING_COVARIATES: return "MISSING_COVARIATES";
        case bsccs::POOR_BLR_STEP:      return "POOR_BLR_STEP";
        default:                        return "FAILED";
    }
}

// ---------------------------------------------------------------------------
// Concrete CcdInterface
// ---------------------------------------------------------------------------

/// The fourth `CcdInterface` implementation, alongside `RcppCcdInterface` and
/// `CmdLineCcdInterface`.
///
/// `CcdInterface` requires four hooks. Only `initializeModelImpl` does real work
/// here; results are read straight off `CyclicCoordinateDescent` rather than
/// routed through the `*OutputWriter` classes, which exist to serialise into a
/// host container (`Rcpp::List`, a CSV stream) that this facade does not have.
class ApiCcdInterface : public bsccs::CcdInterface {
public:
    ApiCcdInterface(bsccs::AbstractModelData& data,
                    bsccs::loggers::ProgressLoggerPtr progressLogger,
                    bsccs::loggers::ErrorHandlerPtr errorHandler,
                    const std::string& computeDevice)
        : source_(data) {
        logger = std::move(progressLogger);
        error = std::move(errorHandler);
        arguments.noiseLevel = bsccs::SILENT;   // as RcppCcdInterface does
        arguments.computeDevice.name = computeDevice;
        arguments.computeMLE = true;            // as .checkInterface() does
    }

    ~ApiCcdInterface() override {
        delete ccd_;
        delete specifics_;
    }

    void initialize() {
        bsccs::AbstractModelData* data = nullptr;
        CcdInterface::initializeModel(&data, &ccd_, &specifics_);
    }

    bsccs::CyclicCoordinateDescent& ccd() { return *ccd_; }
    const bsccs::CyclicCoordinateDescent& ccd() const { return *ccd_; }
    bsccs::AbstractModelData& data() { return source_; }
    const bsccs::AbstractModelData& data() const { return source_; }

    double fit() { return CcdInterface::fitModel(ccd_); }
    double crossValidate() { return CcdInterface::runCrossValidation(ccd_, &source_); }
    void predict() { CcdInterface::predictModel(ccd_, &source_); }

    double profile(const bsccs::ProfileVector& ids,
                   bsccs::ProfileInformationMap& map,
                   int threads, double threshold, bool includePenalty) {
        return CcdInterface::profileModel(ccd_, &source_, ids, map, threads,
                                          threshold, /*overrideNoRegularization=*/false,
                                          includePenalty);
    }

    double evaluateProfile(bsccs::IdType id,
                           const std::vector<double>& points,
                           std::vector<double>& values,
                           std::vector<double>* derivatives,
                           int threads, bool includePenalty) {
        return CcdInterface::evaluateProfileModel(ccd_, &source_, id, points, values,
                                                  derivatives, threads, includePenalty,
                                                  /*startPoints=*/{}, /*startBetas=*/{},
                                                  /*betas=*/nullptr, /*chain=*/false);
    }

    void setNoiseLevel(bsccs::NoiseLevels level) {
        arguments.noiseLevel = level;
        ccd_->setNoiseLevel(level);
        logger->setSilent(level == bsccs::SILENT);
    }

    using CcdInterface::getArguments;

protected:
    void initializeModelImpl(bsccs::AbstractModelData** modelData,
                             bsccs::CyclicCoordinateDescent** ccd,
                             bsccs::AbstractModelSpecifics** model) override {
        *modelData = &source_;

        // The ModelData object already carries its ModelType, so unlike the R
        // binding there is no model-name string to re-parse here.
        const bsccs::ModelType modelType = source_.getModelType();
        const std::string& deviceName = arguments.computeDevice.name;
        const bsccs::DeviceType deviceType =
            (deviceName == "native") ? bsccs::DeviceType::CPU : bsccs::DeviceType::GPU;

        *model = bsccs::AbstractModelSpecifics::factory(modelType, **modelData,
                                                        deviceType, deviceName);
        if (*model == nullptr) {
            throw CyclopsError("Model type is not supported on compute device '" +
                               deviceName + "'");
        }

        *ccd = new bsccs::CyclicCoordinateDescent(**modelData, **model,
                                                  /*prior=*/nullptr, logger, error);
        (*ccd)->setNoiseLevel(arguments.noiseLevel);
    }

    // Results are read directly from the optimizer; see the class comment.
    void predictModelImpl(bsccs::CyclicCoordinateDescent*, bsccs::AbstractModelData*) override {}

    void logModelImpl(bsccs::CyclicCoordinateDescent*, bsccs::AbstractModelData*,
                      bsccs::ProfileInformationMap&, bool) override {}

    void diagnoseModelImpl(bsccs::CyclicCoordinateDescent*, bsccs::AbstractModelData*,
                           double, double) override {}

private:
    bsccs::AbstractModelData& source_;
    bsccs::CyclicCoordinateDescent* ccd_ = nullptr;
    bsccs::AbstractModelSpecifics* specifics_ = nullptr;
};

}  // namespace

// ---------------------------------------------------------------------------
// ModelData
// ---------------------------------------------------------------------------

struct ModelData::Impl {
    Precision precision = Precision::Fp64;
    std::shared_ptr<BufferedLogger> logger;
    std::shared_ptr<ThrowingErrorHandler> error;
    std::unique_ptr<bsccs::AbstractModelData> data;

    /// Number of columns present before any offset/intercept manipulation, used
    /// to validate coefficient-length arguments.
    void requireOutcome() const {
        if (data->getNumberOfRows() == 0) {
            throw CyclopsError("set_outcome() must be called before adding covariates");
        }
    }

    void requireMutable() const {
        if (data->getIsFinalized()) {
            throw CyclopsError("Model data is already finalized");
        }
    }
};

ModelData::ModelData() : impl_(new Impl()) {}
ModelData::~ModelData() = default;

std::unique_ptr<ModelData> ModelData::create(ModelKind kind, Precision precision,
                                             bool silent) {
    std::unique_ptr<ModelData> self(new ModelData());
    self->impl_->precision = precision;
    self->impl_->logger = std::make_shared<BufferedLogger>(silent);
    self->impl_->error = std::make_shared<ThrowingErrorHandler>();

    const auto modelType = toModelType(kind);
    if (precision == Precision::Fp32) {
        self->impl_->data.reset(new bsccs::ModelData<float>(
            modelType, self->impl_->logger, self->impl_->error));
    } else {
        self->impl_->data.reset(new bsccs::ModelData<double>(
            modelType, self->impl_->logger, self->impl_->error));
    }
    return self;
}

void ModelData::set_outcome(const std::vector<double>& y,
                            const std::vector<double>& time,
                            const std::vector<std::int64_t>& stratum_id,
                            const std::vector<std::int64_t>& row_id) {
    impl_->requireMutable();
    if (y.empty()) {
        throw CyclopsError("Outcome vector is empty");
    }
    impl_->data->loadY(stratum_id, row_id, y, time);
}

void ModelData::add_covariates_csc(const std::vector<std::int64_t>& indptr,
                                   const std::vector<std::int64_t>& row_indices,
                                   const std::vector<double>& values,
                                   const std::vector<std::int64_t>& covariate_ids,
                                   bool force_sparse) {
    impl_->requireMutable();
    impl_->requireOutcome();

    if (indptr.empty() || indptr.size() != covariate_ids.size() + 1) {
        throw CyclopsError("indptr must have length equal to the number of "
                           "covariate ids plus one");
    }
    if (indptr.front() != 0) {
        throw CyclopsError("indptr must start at 0");
    }
    const auto nnz = static_cast<std::int64_t>(row_indices.size());
    if (indptr.back() != nnz) {
        throw CyclopsError("indptr must end at the number of stored entries");
    }
    const bool hasValues = !values.empty();
    if (hasValues && values.size() != row_indices.size()) {
        throw CyclopsError("values and row_indices must have the same length");
    }

    const auto rows = static_cast<std::int64_t>(impl_->data->getNumberOfRows());
    for (const auto row : row_indices) {
        if (row < 0 || row >= rows) {
            throw CyclopsError("Row index out of range");
        }
    }

    std::vector<std::int64_t> columnRows;
    std::vector<double> columnValues;

    for (std::size_t column = 0; column < covariate_ids.size(); ++column) {
        const auto begin = indptr[column];
        const auto end = indptr[column + 1];
        if (end < begin) {
            throw CyclopsError("indptr must be non-decreasing");
        }

        columnRows.assign(row_indices.begin() + begin, row_indices.begin() + end);
        if (hasValues) {
            columnValues.assign(values.begin() + begin, values.begin() + end);
        } else {
            columnValues.assign(columnRows.size(), 1.0);
        }

        if (columnRows.empty()) {
            // loadX() reads an empty (rows, values) pair as an intercept column.
            // A single explicit zero yields the intended all-zero indicator
            // column instead, because the loaders drop zero entries.
            columnRows.assign(1, 0);
            columnValues.assign(1, 0.0);
        }

        impl_->data->loadX(covariate_ids[column], columnRows, columnValues,
                           /*reload=*/false, /*append=*/false, force_sparse);
    }
}

void ModelData::add_dense_covariate(std::int64_t covariate_id,
                                    const std::vector<double>& values) {
    impl_->requireMutable();
    impl_->requireOutcome();
    if (values.size() != impl_->data->getNumberOfRows()) {
        throw CyclopsError("A dense covariate must supply one value per row");
    }
    // An empty row-id vector with values present selects DENSE storage.
    impl_->data->loadX(covariate_id, /*rowId=*/{}, values,
                       /*reload=*/false, /*append=*/false, /*forceSparse=*/false);
}

void ModelData::add_intercept() {
    impl_->requireMutable();
    impl_->requireOutcome();
    if (impl_->data->getHasInterceptCovariate()) {
        throw CyclopsError("Model data already has an intercept");
    }
    // addIntercept() inserts at column 0. A promoted offset must stay there --
    // every index derived from getHasOffsetCovariate() assumes it, so inserting
    // ahead of it would silently report the offset's fixed coefficient in place
    // of the intercept.
    if (impl_->data->getHasOffsetCovariate()) {
        throw CyclopsError("Add the intercept before promoting the offset "
                           "covariate: the offset must occupy column 0");
    }
    impl_->data->addIntercept();
}

void ModelData::set_offset_covariate(std::int64_t covariate_id,
                                     bool already_on_log_scale) {
    impl_->requireMutable();
    if (impl_->data->getHasOffsetCovariate()) {
        throw CyclopsError("Model data already has an offset");
    }
    impl_->data->setOffsetCovariate(covariate_id);
    if (!already_on_log_scale) {
        impl_->data->logTransformCovariate(0);
    }
}

void ModelData::make_dense(const std::vector<std::int64_t>& covariate_ids) {
    impl_->requireMutable();
    for (const auto id : covariate_ids) {
        impl_->data->convertCovariateToDense(id);
    }
}

std::vector<double> ModelData::normalize(NormalizationKind kind) {
    impl_->requireMutable();
    return impl_->data->normalizeCovariates(toNormalizationType(kind));
}

void ModelData::finalize() {
    if (impl_->data->getIsFinalized()) return;
    impl_->data->setIsFinalized(true);
}

bool ModelData::is_finalized() const { return impl_->data->getIsFinalized(); }
ModelKind ModelData::kind() const { return fromModelType(impl_->data->getModelType()); }
Precision ModelData::precision() const { return impl_->precision; }
std::size_t ModelData::row_count() const { return impl_->data->getNumberOfRows(); }
std::size_t ModelData::covariate_count() const { return impl_->data->getNumberOfCovariates(); }
std::size_t ModelData::stratum_count() const { return impl_->data->getNumberOfPatients(); }
bool ModelData::has_intercept() const { return impl_->data->getHasInterceptCovariate(); }
bool ModelData::has_offset() const { return impl_->data->getHasOffsetCovariate(); }

std::int64_t ModelData::intercept_label() const {
    if (!impl_->data->getHasInterceptCovariate()) {
        throw CyclopsError("Model data has no intercept");
    }
    const std::size_t index = impl_->data->getHasOffsetCovariate() ? 1 : 0;
    return impl_->data->getColumnNumericalLabel(index);
}

std::vector<std::int64_t> ModelData::covariate_ids() const {
    const std::size_t count = impl_->data->getNumberOfCovariates();
    std::vector<std::int64_t> ids;
    ids.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        ids.push_back(impl_->data->getColumnNumericalLabel(i));
    }
    return ids;
}

std::vector<ColumnFormat> ModelData::covariate_formats() const {
    const std::size_t count = impl_->data->getNumberOfCovariates();
    std::vector<ColumnFormat> formats;
    formats.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        formats.push_back(fromFormatType(impl_->data->getColumnType(i)));
    }
    return formats;
}

std::vector<double> ModelData::outcome() const { return impl_->data->copyYVector(); }
std::vector<double> ModelData::time() const { return impl_->data->copyTimeVector(); }

std::vector<int> ModelData::stratum_index() const {
    return impl_->data->getPidVectorSTL();
}


std::vector<double> ModelData::univariable_correlation(
    const std::vector<std::int64_t>& ids) const {
    return impl_->data->univariableCorrelation(ids);
}

double ModelData::column_sum(std::int64_t covariate_id, int power) const {
    return impl_->data->sum(covariate_id, power);
}

std::vector<double> ModelData::sum_by_stratum(std::int64_t covariate_id,
                                              int power) const {
    std::vector<double> result(impl_->data->getNumberOfPatients(), 0.0);
    impl_->data->sumByPid(result, covariate_id, power);
    return result;
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct Model::Impl {
    ModelData* data = nullptr;
    std::unique_ptr<ApiCcdInterface> interface;
    PriorOptions prior;
    FitOptions options;
    bool startValuesSupplied = false;

    bsccs::AbstractModelData& raw() { return *data->impl_->data; }
    const bsccs::AbstractModelData& raw() const { return *data->impl_->data; }
    bsccs::CyclicCoordinateDescent& ccd() { return interface->ccd(); }
    const bsccs::CyclicCoordinateDescent& ccd() const { return interface->ccd(); }

    /// Index of the first estimated coefficient: column 0 holds the offset when
    /// one is present, and is never reported.
    std::size_t betaOffset() const { return raw().getHasOffsetCovariate() ? 1u : 0u; }

    std::size_t columnIndexOf(std::int64_t covariateId) const {
        const int index = raw().getColumnIndexByName(covariateId);
        if (index < 0) {
            std::ostringstream stream;
            stream << "Variable " << covariateId << " not found";
            throw CyclopsError(stream.str());
        }
        return static_cast<std::size_t>(index);
    }

    std::vector<std::size_t> columnIndicesOf(
        const std::vector<std::int64_t>& ids) const {
        std::vector<std::size_t> indices;
        if (ids.empty()) {
            const std::size_t count = raw().getNumberOfCovariates();
            indices.reserve(count);
            for (std::size_t i = 0; i < count; ++i) indices.push_back(i);
        } else {
            indices.reserve(ids.size());
            for (const auto id : ids) indices.push_back(columnIndexOf(id));
        }
        return indices;
    }

    void applyPrior();
    void checkJeffreysIsApplicable() const;
    void applyOptions();
    void applyInterceptWarmStart();
    FitResult collectResult(double fitSeconds);
};

void Model::Impl::applyPrior() {
    using namespace bsccs::priors;

    const std::size_t length = raw().getNumberOfCovariates();

    // Resolve the exclusion set. fitCyclopsModel() drops the intercept from
    // regularisation unless force_intercept is set; replicate that here so a
    // Python fit and an R fit penalise the same coefficients.
    std::vector<std::int64_t> exclude = prior.exclude;
    if (prior.kind != PriorKind::None && !prior.force_intercept &&
        raw().getHasInterceptCovariate()) {
        const std::size_t index = raw().getHasOffsetCovariate() ? 1 : 0;
        const auto interceptId = raw().getColumnNumericalLabel(index);
        if (std::find(exclude.begin(), exclude.end(), interceptId) == exclude.end()) {
            exclude.push_back(interceptId);
        }
    }

    const bool perCovariate = !prior.kinds.empty() || !prior.variances.empty();
    if (perCovariate) {
        if (prior.kinds.size() != length || prior.variances.size() != length) {
            throw CyclopsError("Per-covariate priors must supply one kind and one "
                               "variance per covariate");
        }
    }

    const bool anyJeffreys =
        prior.kind == PriorKind::Jeffreys ||
        std::find(prior.kinds.begin(), prior.kinds.end(), PriorKind::Jeffreys) !=
            prior.kinds.end();
    if (anyJeffreys) checkJeffreysIsApplicable();

    JointPriorPtr jointPrior;
    if (perCovariate) {
        auto first = CovariatePrior::makePrior(toPriorType(prior.kinds[0]),
                                               prior.variances[0]);
        auto mixture = bsccs::make_shared<MixtureJointPrior>(first,
                                                             static_cast<int>(length));
        for (std::size_t i = 1; i < length; ++i) {
            mixture->changePrior(
                CovariatePrior::makePrior(toPriorType(prior.kinds[i]),
                                          prior.variances[i]),
                static_cast<int>(i));
        }
        // The exclusion set still applies. R reaches its per-column branch only
        // when the exclusion list is empty and silently uses baseVariance[0]
        // otherwise; honouring both here keeps `exclude` and the automatic
        // intercept exclusion meaningful whichever prior form was used.
        auto noPrior = bsccs::make_shared<NoPrior>();
        for (const auto id : exclude) {
            mixture->changePrior(noPrior, static_cast<int>(columnIndexOf(id)));
        }
        jointPrior = mixture;
    } else if (exclude.empty()) {
        jointPrior = bsccs::make_shared<FullyExchangeableJointPrior>(
            CovariatePrior::makePrior(toPriorType(prior.kind), prior.variance));
    } else {
        auto single = CovariatePrior::makePrior(toPriorType(prior.kind),
                                                prior.variance);
        auto mixture = bsccs::make_shared<MixtureJointPrior>(single,
                                                             static_cast<int>(length));
        auto noPrior = bsccs::make_shared<NoPrior>();
        for (const auto id : exclude) {
            mixture->changePrior(noPrior, static_cast<int>(columnIndexOf(id)));
        }
        jointPrior = mixture;
    }

    ccd().setPrior(jointPrior);
}

void Model::Impl::checkJeffreysIsApplicable() const {
    // The Jeffreys prior is only implemented for a single binary covariate.
    // fitCyclopsModel() enforces this in R before reaching the optimizer;
    // without the same guard the C++ silently produces an undefined result.
    if (raw().getNumberOfCovariates() > 1) {
        throw CyclopsError(
            "Jeffreys prior is currently only implemented for 1 covariate");
    }

    const std::size_t index = betaOffset();
    if (raw().getColumnType(index) == bsccs::INDICATOR) return;

    const auto covariateId = raw().getColumnNumericalLabel(index);
    const double count = raw().sum(covariateId, 0);
    const double total = raw().sum(covariateId, 1);
    const double mean = (count > 0.0) ? total / count : 0.0;
    if (mean != 0.0 && mean != 1.0) {
        throw CyclopsError(
            "Jeffreys prior is currently only implemented for indicator "
            "covariates");
    }
}

void Model::Impl::applyOptions() {
    auto& arguments = interface->getArguments();

    arguments.modeFinding.maxIterations = options.max_iterations;
    arguments.modeFinding.tolerance = options.tolerance;
    arguments.modeFinding.convergenceType = toConvergenceType(options.convergence);
    arguments.modeFinding.useKktSwindle = options.use_kkt_swindle;
    arguments.modeFinding.swindleMultipler = options.swindle_multiplier;
    arguments.modeFinding.initialBound = options.initial_bound;
    arguments.modeFinding.maxBoundCount = options.max_bound_count;
    arguments.modeFinding.doItAll = options.do_it_all;
    arguments.modeFinding.algorithmType = (options.algorithm == AlgorithmKind::Mm)
        ? bsccs::AlgorithmType::MM : bsccs::AlgorithmType::CCD;

    arguments.crossValidation.useAutoSearchCV = options.auto_search;
    arguments.crossValidation.fold = options.fold;
    arguments.crossValidation.foldToCompute = options.fold * options.cv_repetitions;
    arguments.crossValidation.lowerLimit = options.lower_limit;
    arguments.crossValidation.upperLimit = options.upper_limit;
    arguments.crossValidation.gridSteps = options.grid_steps;
    arguments.crossValidation.startingVariance = options.starting_variance;
    arguments.crossValidation.syncCV = options.sync_cv;
    // createControl()'s cvWarmStart default under fitter = "default": each fold
    // fit starts from that fold's solution at the nearest variance already tried.
    arguments.crossValidation.warmStart = !options.reset_coefficients;

    // "auto" selector resolution, copied from fitCyclopsModel(): row-wise for
    // the unstratified likelihoods, otherwise whichever partition has more
    // units. CcdInterface::getDefaultSelectorTypeOrOverride() makes a different
    // choice for logistic/Poisson, so DEFAULT is never forwarded.
    if (options.selector == SelectorKind::Auto) {
        const auto kind = fromModelType(raw().getModelType());
        if (kind == ModelKind::Poisson || kind == ModelKind::Logistic) {
            arguments.crossValidation.selectorType = bsccs::SelectorType::BY_ROW;
        } else {
            const auto rows = static_cast<double>(raw().getNumberOfRows());
            const auto strata = static_cast<double>(raw().getNumberOfPatients());
            const double rowsPerStratum = (strata > 0) ? rows / strata : rows;
            arguments.crossValidation.selectorType = (rowsPerStratum < strata)
                ? bsccs::SelectorType::BY_PID : bsccs::SelectorType::BY_ROW;
        }
    } else {
        arguments.crossValidation.selectorType = toSelectorType(options.selector);
    }

    arguments.threads = options.threads;
    arguments.seed = options.seed;
    arguments.resetCoefficients = options.reset_coefficients;

    interface->setNoiseLevel(toNoiseLevel(options.noise));
}

void Model::Impl::applyInterceptWarmStart() {
    // fitCyclopsModel() warm-starts the intercept at the value that reproduces
    // the marginal outcome mean and leaves every other coefficient at zero.
    // CCD is only locally convergent for these likelihoods, so skipping this
    // changes the iterate path (and occasionally the mode that is reached).
    if (!raw().getHasInterceptCovariate() || startValuesSupplied) return;

    const auto kind = fromModelType(raw().getModelType());
    if (kind != ModelKind::Logistic && kind != ModelKind::Poisson &&
        kind != ModelKind::Normal) {
        return;
    }

    const auto y = raw().copyYVector();
    if (y.empty()) return;
    const double mean = std::accumulate(y.begin(), y.end(), 0.0) /
                        static_cast<double>(y.size());

    double intercept = 0.0;
    switch (kind) {
        case ModelKind::Logistic: intercept = std::log(mean / (1.0 - mean)); break;
        case ModelKind::Poisson:  intercept = std::log(mean); break;
        default:                  intercept = mean; break;
    }

    const std::size_t offset = betaOffset();
    std::vector<double> beta(raw().getNumberOfCovariates(), 0.0);
    if (offset == 1) beta[0] = 1.0;   // offset coefficient is fixed at 1
    beta[offset] = intercept;

    ccd().setBeta(beta);
    ccd().setStartingBeta(beta);
}

FitResult Model::Impl::collectResult(double fitSeconds) {
    // Field-for-field the payload DiagnosticsOutputWriter writes for R.
    FitResult result;
    auto& optimizer = ccd();

    result.log_likelihood = optimizer.getLogLikelihood();
    result.log_prior = optimizer.getLogPrior();
    result.iterations = optimizer.getIterationCount();
    result.prior_info = optimizer.getPriorInfo();
    result.variance = optimizer.getHyperprior();
    result.cross_validation_info = optimizer.getCrossValidationInfo();
    result.covariate_count = optimizer.getBetaSize();

    auto flag = optimizer.getUpdateReturnFlag();
    if (result.covariate_count == 0) flag = bsccs::MISSING_COVARIATES;
    result.return_flag = returnFlagString(flag);

    const int size = optimizer.getBetaSize();
    for (int index = static_cast<int>(betaOffset()); index < size; ++index) {
        result.covariate_ids.push_back(raw().getColumnNumericalLabel(index));
        result.coefficients.push_back(optimizer.getBeta(index));
    }

    result.fit_seconds = fitSeconds;
    return result;
}

Model::Model() : impl_(new Impl()) {}
Model::~Model() = default;

std::unique_ptr<Model> Model::create(ModelData& data,
                                     const std::string& compute_device) {
    if (!data.is_finalized()) {
        throw CyclopsError("Model data must be finalized before fitting");
    }
    if (data.row_count() == 0 || data.covariate_count() == 0) {
        throw CyclopsError("Model data are incompletely loaded");
    }

    std::unique_ptr<Model> self(new Model());
    self->impl_->data = &data;
    self->impl_->interface.reset(new ApiCcdInterface(
        *data.impl_->data, data.impl_->logger, data.impl_->error, compute_device));
    self->impl_->interface->initialize();

    // CyclicCoordinateDescent is constructed without a prior, so install the
    // default immediately: getPriorInfo() and getHyperprior() dereference it.
    self->impl_->applyPrior();
    self->impl_->applyOptions();
    return self;
}

void Model::set_prior(const PriorOptions& prior) {
    impl_->prior = prior;
    impl_->applyPrior();
}

void Model::set_options(const FitOptions& options) {
    impl_->options = options;
    impl_->applyOptions();
}

void Model::set_weights(const std::vector<double>& weights) {
    if (weights.size() != impl_->raw().getNumberOfRows()) {
        throw CyclopsError("Must provide a weight for each data row");
    }
    if (std::any_of(weights.begin(), weights.end(),
                    [](double w) { return w < 0.0; })) {
        throw CyclopsError("Only non-negative weights are allowed");
    }
    std::vector<double> copy(weights);
    impl_->ccd().setWeights(copy.data());
}

void Model::set_censor_weights(const std::vector<double>& weights) {
    if (weights.size() != impl_->raw().getNumberOfRows()) {
        throw CyclopsError("Must provide a censoring weight for each data row");
    }
    if (std::any_of(weights.begin(), weights.end(),
                    [](double w) { return w < 0.0 || w > 1.0; })) {
        throw CyclopsError("Censoring weights must lie in [0, 1]");
    }
    std::vector<double> copy(weights);
    impl_->ccd().setCensorWeights(copy.data());
}

void Model::set_start_values(const std::vector<double>& beta) {
    // Sized like coefficients() and FitResult: the offset column is never the
    // caller's to set, since its coefficient is fixed at 1.
    const std::size_t expected =
        impl_->raw().getNumberOfCovariates() - impl_->betaOffset();
    if (beta.size() != expected) {
        std::ostringstream stream;
        stream << "Must provide a starting value for each estimated coefficient ("
               << expected << " expected, " << beta.size() << " given)";
        throw CyclopsError(stream.str());
    }
    std::vector<double> full;
    full.reserve(beta.size() + impl_->betaOffset());
    if (impl_->betaOffset() == 1) full.push_back(1.0);
    full.insert(full.end(), beta.begin(), beta.end());

    impl_->ccd().setBeta(full);
    impl_->ccd().setStartingBeta(full);
    impl_->startValuesSupplied = true;
}

void Model::set_fixed(const std::vector<bool>& fixed) {
    const std::size_t expected =
        impl_->raw().getNumberOfCovariates() - impl_->betaOffset();
    if (fixed.size() != expected) {
        std::ostringstream stream;
        stream << "Must provide a flag for each estimated coefficient ("
               << expected << " expected, " << fixed.size() << " given)";
        throw CyclopsError(stream.str());
    }
    const auto offset = static_cast<int>(impl_->betaOffset());
    for (std::size_t i = 0; i < fixed.size(); ++i) {
        impl_->ccd().setFixedBeta(offset + static_cast<int>(i), fixed[i]);
    }
}

FitResult Model::fit() {
    impl_->applyInterceptWarmStart();

    double seconds = 0.0;
    if (impl_->prior.use_cross_validation) {
        if (impl_->prior.kind == PriorKind::None) {
            throw CyclopsError("Cross-validation requires a regularising prior");
        }
        const auto& options = impl_->options;
        const std::size_t units =
            (impl_->interface->getArguments().crossValidation.selectorType ==
             bsccs::SelectorType::BY_ROW)
                ? impl_->raw().getNumberOfRows()
                : impl_->raw().getNumberOfPatients();
        if (static_cast<std::size_t>(options.min_cv_data) > units) {
            throw CyclopsError("Insufficient data count for cross validation");
        }
        impl_->interface->getArguments().crossValidation.doFitAtOptimal = true;
        seconds = impl_->interface->crossValidate();
    } else {
        seconds = impl_->interface->fit();
    }

    auto result = impl_->collectResult(seconds);

    // fitCyclopsModel() retries a failed Bayesian-logistic-regression step under
    // the Lange criterion before giving up. R does this by recursing with a
    // modified *local* control, so the caller's configuration is untouched
    // afterwards; restore it here for the same reason -- otherwise a second
    // fit() on this object would silently keep using Lange.
    if (impl_->options.retry_on_poor_blr_step &&
        result.return_flag == "POOR_BLR_STEP" &&
        impl_->options.convergence == ConvergenceKind::Gradient) {
        const auto callerOptions = impl_->options;
        auto retryOptions = callerOptions;
        retryOptions.convergence = ConvergenceKind::Lange;
        set_options(retryOptions);
        try {
            seconds = impl_->prior.use_cross_validation
                ? impl_->interface->crossValidate()
                : impl_->interface->fit();
            result = impl_->collectResult(seconds);
        } catch (...) {
            set_options(callerOptions);
            throw;
        }
        set_options(callerOptions);
    }

    return result;
}

std::vector<double> Model::predict() const {
    auto& optimizer = const_cast<bsccs::CyclicCoordinateDescent&>(impl_->ccd());
    std::vector<double> predictions(optimizer.getPredictionSize(), 0.0);
    if (!predictions.empty()) {
        optimizer.getPredictiveEstimates(predictions.data(), nullptr);
    }
    return predictions;
}

double Model::log_likelihood() const {
    return const_cast<bsccs::CyclicCoordinateDescent&>(impl_->ccd()).getLogLikelihood();
}

double Model::log_prior() const {
    return const_cast<bsccs::CyclicCoordinateDescent&>(impl_->ccd()).getLogPrior();
}

std::vector<double> Model::coefficients() const {
    auto& optimizer = const_cast<bsccs::CyclicCoordinateDescent&>(impl_->ccd());
    const int size = optimizer.getBetaSize();
    std::vector<double> beta;
    beta.reserve(size - impl_->betaOffset());
    for (int index = static_cast<int>(impl_->betaOffset()); index < size; ++index) {
        beta.push_back(optimizer.getBeta(index));
    }
    return beta;
}

std::vector<double> Model::gradient() const {
    auto& optimizer = const_cast<bsccs::CyclicCoordinateDescent&>(impl_->ccd());
    const auto offset = static_cast<int>(impl_->betaOffset());
    const int length = optimizer.getBetaSize() - offset;
    std::vector<double> result;
    result.reserve(std::max(length, 0));
    for (int i = 0; i < length; ++i) {
        result.push_back(optimizer.getLogLikelihoodGradient(i + offset));
    }
    return result;
}

std::vector<double> Model::hessian_diagonal(
    const std::vector<std::int64_t>& ids) const {
    auto& optimizer = const_cast<bsccs::CyclicCoordinateDescent&>(impl_->ccd());
    const auto indices = impl_->columnIndicesOf(ids);
    std::vector<double> result;
    result.reserve(indices.size());
    for (const auto index : indices) {
        result.push_back(-optimizer.getHessianDiagonal(static_cast<int>(index)));
    }
    return result;
}

std::vector<double> Model::fisher_information(
    const std::vector<std::int64_t>& ids) const {
    const auto indices = impl_->columnIndicesOf(ids);
    std::vector<bsccs::IdType> asIds(indices.begin(), indices.end());
    const auto matrix = impl_->ccd().computeFisherInformation(asIds);

    std::vector<double> flat(static_cast<std::size_t>(matrix.size()));
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
            flat[static_cast<std::size_t>(row * matrix.cols() + col)] =
                matrix(row, col);
        }
    }
    return flat;
}

std::vector<double> Model::standard_errors(
    const std::vector<std::int64_t>& ids) const {
    const auto indices = impl_->columnIndicesOf(ids);
    std::vector<bsccs::IdType> asIds(indices.begin(), indices.end());
    const auto information = impl_->ccd().computeFisherInformation(asIds);

    // Matches R's `sqrt(diag(solve(fisherInformation)))`, but reports a singular
    // information matrix rather than propagating inf/NaN. R surfaces the same
    // condition as a LAPACK error out of solve(); silently returning NaN would
    // let a caller treat a degenerate fit as if it had standard errors.
    const Eigen::FullPivLU<Eigen::MatrixXd> decomposition(information);
    if (!decomposition.isInvertible()) {
        throw CyclopsError(
            "Fisher information matrix is singular, so no asymptotic standard "
            "errors exist. This is expected when a coefficient is unidentified "
            "or its estimate has run to infinity.");
    }

    const Eigen::MatrixXd covariance = decomposition.inverse();
    std::vector<double> result;
    result.reserve(indices.size());
    for (Eigen::Index i = 0; i < covariance.rows(); ++i) {
        const double variance = covariance(i, i);
        if (!(variance >= 0.0)) {   // also catches NaN
            throw CyclopsError(
                "Asymptotic variance is not positive; the fit is degenerate.");
        }
        result.push_back(std::sqrt(variance));
    }
    return result;
}

std::vector<bool> Model::is_regularized() const {
    const auto& optimizer = impl_->ccd();
    const int size = const_cast<bsccs::CyclicCoordinateDescent&>(optimizer).getBetaSize();
    std::vector<bool> result;
    result.reserve(size - impl_->betaOffset());
    for (int index = static_cast<int>(impl_->betaOffset()); index < size; ++index) {
        result.push_back(optimizer.getIsRegularized(index));
    }
    return result;
}

std::vector<ProfileInterval> Model::profile(const std::vector<std::int64_t>& ids,
                                            int threads, double threshold,
                                            bool include_penalty) {
    bsccs::ProfileVector covariates(ids.begin(), ids.end());
    if (covariates.empty()) {
        for (const auto id : impl_->data->covariate_ids()) covariates.push_back(id);
    }

    bsccs::ProfileInformationMap map;
    impl_->interface->profile(covariates, map, threads, threshold, include_penalty);

    std::vector<ProfileInterval> intervals;
    intervals.reserve(covariates.size());
    for (const auto id : covariates) {
        const auto& info = map[id];
        ProfileInterval interval;
        interval.covariate_id = id;
        interval.lower = info.lower95Bound;
        interval.upper = info.upper95Bound;
        interval.evaluations = info.evaluations;
        intervals.push_back(interval);
    }
    return intervals;
}

ProfileCurve Model::profile_curve(std::int64_t covariate_id,
                                  const std::vector<double>& points,
                                  int threads, bool include_penalty,
                                  bool with_derivatives) {
    ProfileCurve curve;
    curve.points = points;
    curve.values.assign(points.size(), 0.0);
    if (with_derivatives) curve.derivatives.assign(points.size(), 0.0);

    impl_->interface->evaluateProfile(
        covariate_id, points, curve.values,
        with_derivatives ? &curve.derivatives : nullptr, threads, include_penalty);
    return curve;
}

std::vector<std::string> Model::take_log() {
    return impl_->data->impl_->logger->drain();
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

std::string version() { return CYCLOPS_VERSION; }

std::vector<std::string> list_gpu_devices() {
    // Populated only by the GPU-enabled builds, which supply their own
    // enumeration entry point.
    return {};
}

bool removes_intercept(ModelKind kind) {
    // `bsccs::Models::removeIntercept` omits the survival models, but their
    // intercept is absorbed into the baseline hazard just as the conditional
    // models' is absorbed into the strata. The authoritative user-facing list is
    // R's `.cyclopsGetRemoveInterceptNames()`, reproduced here.
    switch (kind) {
        case ModelKind::ConditionalLogistic:
        case ModelKind::TiedConditionalLogistic:
        case ModelKind::EfronConditionalLogistic:
        case ModelKind::ConditionalPoisson:
        case ModelKind::SelfControlledCaseSeries:
        case ModelKind::Cox:
        case ModelKind::CoxRaw:
        case ModelKind::TimeVaryingCox:
        case ModelKind::FineGray:
            return true;
        default:
            return false;
    }
}

bool requires_strata(ModelKind kind) {
    return bsccs::Models::requiresStratumID(toModelType(kind));
}

bool requires_time(ModelKind kind) {
    const auto type = toModelType(kind);
    return bsccs::Models::requiresCensoredData(type) ||
           type == bsccs::ModelType::TIME_VARYING_COX ||
           type == bsccs::ModelType::FINE_GRAY ||
           type == bsccs::ModelType::SELF_CONTROLLED_MODEL;
}

bool requires_offset(ModelKind kind) {
    return bsccs::Models::requiresOffset(toModelType(kind));
}

}  // namespace api
}  // namespace cyclops
