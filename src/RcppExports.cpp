// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// cyclopsGetModelTypeNames
std::vector<std::string> cyclopsGetModelTypeNames();
RcppExport SEXP Cyclops_cyclopsGetModelTypeNames() {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    __result = Rcpp::wrap(cyclopsGetModelTypeNames());
    return __result;
END_RCPP
}
// cyclopsGetRemoveInterceptNames
std::vector<std::string> cyclopsGetRemoveInterceptNames();
RcppExport SEXP Cyclops_cyclopsGetRemoveInterceptNames() {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    __result = Rcpp::wrap(cyclopsGetRemoveInterceptNames());
    return __result;
END_RCPP
}
// cyclopsGetIsSurvivalNames
std::vector<std::string> cyclopsGetIsSurvivalNames();
RcppExport SEXP Cyclops_cyclopsGetIsSurvivalNames() {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    __result = Rcpp::wrap(cyclopsGetIsSurvivalNames());
    return __result;
END_RCPP
}
// cyclopsGetUseOffsetNames
std::vector<std::string> cyclopsGetUseOffsetNames();
RcppExport SEXP Cyclops_cyclopsGetUseOffsetNames() {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    __result = Rcpp::wrap(cyclopsGetUseOffsetNames());
    return __result;
END_RCPP
}
// cyclopsSetBeta
void cyclopsSetBeta(SEXP inRcppCcdInterface, const std::vector<double>& beta);
RcppExport SEXP Cyclops_cyclopsSetBeta(SEXP inRcppCcdInterfaceSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type beta(betaSEXP);
    cyclopsSetBeta(inRcppCcdInterface, beta);
    return R_NilValue;
END_RCPP
}
// cyclopsSetFixedBeta
void cyclopsSetFixedBeta(SEXP inRcppCcdInterface, int beta, bool fixed);
RcppExport SEXP Cyclops_cyclopsSetFixedBeta(SEXP inRcppCcdInterfaceSEXP, SEXP betaSEXP, SEXP fixedSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< int >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< bool >::type fixed(fixedSEXP);
    cyclopsSetFixedBeta(inRcppCcdInterface, beta, fixed);
    return R_NilValue;
END_RCPP
}
// cyclopsGetIsRegularized
bool cyclopsGetIsRegularized(SEXP inRcppCcdInterface, const int index);
RcppExport SEXP Cyclops_cyclopsGetIsRegularized(SEXP inRcppCcdInterfaceSEXP, SEXP indexSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< const int >::type index(indexSEXP);
    __result = Rcpp::wrap(cyclopsGetIsRegularized(inRcppCcdInterface, index));
    return __result;
END_RCPP
}
// cyclopsSetWeights
void cyclopsSetWeights(SEXP inRcppCcdInterface, NumericVector& weights);
RcppExport SEXP Cyclops_cyclopsSetWeights(SEXP inRcppCcdInterfaceSEXP, SEXP weightsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type weights(weightsSEXP);
    cyclopsSetWeights(inRcppCcdInterface, weights);
    return R_NilValue;
END_RCPP
}
// cyclopsGetPredictiveLogLikelihood
double cyclopsGetPredictiveLogLikelihood(SEXP inRcppCcdInterface, NumericVector& weights);
RcppExport SEXP Cyclops_cyclopsGetPredictiveLogLikelihood(SEXP inRcppCcdInterfaceSEXP, SEXP weightsSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type weights(weightsSEXP);
    __result = Rcpp::wrap(cyclopsGetPredictiveLogLikelihood(inRcppCcdInterface, weights));
    return __result;
END_RCPP
}
// cyclopsGetLogLikelihood
double cyclopsGetLogLikelihood(SEXP inRcppCcdInterface);
RcppExport SEXP Cyclops_cyclopsGetLogLikelihood(SEXP inRcppCcdInterfaceSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    __result = Rcpp::wrap(cyclopsGetLogLikelihood(inRcppCcdInterface));
    return __result;
END_RCPP
}
// cyclopsGetFisherInformation
Eigen::MatrixXd cyclopsGetFisherInformation(SEXP inRcppCcdInterface, const SEXP sexpCovariates);
RcppExport SEXP Cyclops_cyclopsGetFisherInformation(SEXP inRcppCcdInterfaceSEXP, SEXP sexpCovariatesSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type sexpCovariates(sexpCovariatesSEXP);
    __result = Rcpp::wrap(cyclopsGetFisherInformation(inRcppCcdInterface, sexpCovariates));
    return __result;
END_RCPP
}
// cyclopsSetPrior
void cyclopsSetPrior(SEXP inRcppCcdInterface, const std::vector<std::string>& priorTypeName, const std::vector<double>& variance, SEXP excludeNumeric, SEXP sexpGraph);
RcppExport SEXP Cyclops_cyclopsSetPrior(SEXP inRcppCcdInterfaceSEXP, SEXP priorTypeNameSEXP, SEXP varianceSEXP, SEXP excludeNumericSEXP, SEXP sexpGraphSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< const std::vector<std::string>& >::type priorTypeName(priorTypeNameSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type variance(varianceSEXP);
    Rcpp::traits::input_parameter< SEXP >::type excludeNumeric(excludeNumericSEXP);
    Rcpp::traits::input_parameter< SEXP >::type sexpGraph(sexpGraphSEXP);
    cyclopsSetPrior(inRcppCcdInterface, priorTypeName, variance, excludeNumeric, sexpGraph);
    return R_NilValue;
END_RCPP
}
// cyclopsProfileModel
List cyclopsProfileModel(SEXP inRcppCcdInterface, SEXP sexpCovariates, double threshold, bool override, bool includePenalty);
RcppExport SEXP Cyclops_cyclopsProfileModel(SEXP inRcppCcdInterfaceSEXP, SEXP sexpCovariatesSEXP, SEXP thresholdSEXP, SEXP overrideSEXP, SEXP includePenaltySEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< SEXP >::type sexpCovariates(sexpCovariatesSEXP);
    Rcpp::traits::input_parameter< double >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< bool >::type override(overrideSEXP);
    Rcpp::traits::input_parameter< bool >::type includePenalty(includePenaltySEXP);
    __result = Rcpp::wrap(cyclopsProfileModel(inRcppCcdInterface, sexpCovariates, threshold, override, includePenalty));
    return __result;
END_RCPP
}
// cyclopsPredictModel
List cyclopsPredictModel(SEXP inRcppCcdInterface);
RcppExport SEXP Cyclops_cyclopsPredictModel(SEXP inRcppCcdInterfaceSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    __result = Rcpp::wrap(cyclopsPredictModel(inRcppCcdInterface));
    return __result;
END_RCPP
}
// cyclopsSetControl
void cyclopsSetControl(SEXP inRcppCcdInterface, int maxIterations, double tolerance, const std::string& convergenceType, bool useAutoSearch, int fold, int foldToCompute, double lowerLimit, double upperLimit, int gridSteps, const std::string& noiseLevel, int threads, int seed, bool resetCoefficients, double startingVariance, bool useKKTSwindle, int swindleMultipler, const std::string& selectorType);
RcppExport SEXP Cyclops_cyclopsSetControl(SEXP inRcppCcdInterfaceSEXP, SEXP maxIterationsSEXP, SEXP toleranceSEXP, SEXP convergenceTypeSEXP, SEXP useAutoSearchSEXP, SEXP foldSEXP, SEXP foldToComputeSEXP, SEXP lowerLimitSEXP, SEXP upperLimitSEXP, SEXP gridStepsSEXP, SEXP noiseLevelSEXP, SEXP threadsSEXP, SEXP seedSEXP, SEXP resetCoefficientsSEXP, SEXP startingVarianceSEXP, SEXP useKKTSwindleSEXP, SEXP swindleMultiplerSEXP, SEXP selectorTypeSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    Rcpp::traits::input_parameter< int >::type maxIterations(maxIterationsSEXP);
    Rcpp::traits::input_parameter< double >::type tolerance(toleranceSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type convergenceType(convergenceTypeSEXP);
    Rcpp::traits::input_parameter< bool >::type useAutoSearch(useAutoSearchSEXP);
    Rcpp::traits::input_parameter< int >::type fold(foldSEXP);
    Rcpp::traits::input_parameter< int >::type foldToCompute(foldToComputeSEXP);
    Rcpp::traits::input_parameter< double >::type lowerLimit(lowerLimitSEXP);
    Rcpp::traits::input_parameter< double >::type upperLimit(upperLimitSEXP);
    Rcpp::traits::input_parameter< int >::type gridSteps(gridStepsSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type noiseLevel(noiseLevelSEXP);
    Rcpp::traits::input_parameter< int >::type threads(threadsSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type resetCoefficients(resetCoefficientsSEXP);
    Rcpp::traits::input_parameter< double >::type startingVariance(startingVarianceSEXP);
    Rcpp::traits::input_parameter< bool >::type useKKTSwindle(useKKTSwindleSEXP);
    Rcpp::traits::input_parameter< int >::type swindleMultipler(swindleMultiplerSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type selectorType(selectorTypeSEXP);
    cyclopsSetControl(inRcppCcdInterface, maxIterations, tolerance, convergenceType, useAutoSearch, fold, foldToCompute, lowerLimit, upperLimit, gridSteps, noiseLevel, threads, seed, resetCoefficients, startingVariance, useKKTSwindle, swindleMultipler, selectorType);
    return R_NilValue;
END_RCPP
}
// cyclopsRunCrossValidationl
List cyclopsRunCrossValidationl(SEXP inRcppCcdInterface);
RcppExport SEXP Cyclops_cyclopsRunCrossValidationl(SEXP inRcppCcdInterfaceSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    __result = Rcpp::wrap(cyclopsRunCrossValidationl(inRcppCcdInterface));
    return __result;
END_RCPP
}
// cyclopsFitModel
List cyclopsFitModel(SEXP inRcppCcdInterface);
RcppExport SEXP Cyclops_cyclopsFitModel(SEXP inRcppCcdInterfaceSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    __result = Rcpp::wrap(cyclopsFitModel(inRcppCcdInterface));
    return __result;
END_RCPP
}
// cyclopsLogModel
List cyclopsLogModel(SEXP inRcppCcdInterface);
RcppExport SEXP Cyclops_cyclopsLogModel(SEXP inRcppCcdInterfaceSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inRcppCcdInterface(inRcppCcdInterfaceSEXP);
    __result = Rcpp::wrap(cyclopsLogModel(inRcppCcdInterface));
    return __result;
END_RCPP
}
// cyclopsInitializeModel
List cyclopsInitializeModel(SEXP inModelData, const std::string& modelType, bool computeMLE);
RcppExport SEXP Cyclops_cyclopsInitializeModel(SEXP inModelDataSEXP, SEXP modelTypeSEXP, SEXP computeMLESEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type inModelData(inModelDataSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type modelType(modelTypeSEXP);
    Rcpp::traits::input_parameter< bool >::type computeMLE(computeMLESEXP);
    __result = Rcpp::wrap(cyclopsInitializeModel(inModelData, modelType, computeMLE));
    return __result;
END_RCPP
}
// isSorted
bool isSorted(const DataFrame& dataFrame, const std::vector<std::string>& indexes, const std::vector<bool>& ascending);
RcppExport SEXP Cyclops_isSorted(SEXP dataFrameSEXP, SEXP indexesSEXP, SEXP ascendingSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const DataFrame& >::type dataFrame(dataFrameSEXP);
    Rcpp::traits::input_parameter< const std::vector<std::string>& >::type indexes(indexesSEXP);
    Rcpp::traits::input_parameter< const std::vector<bool>& >::type ascending(ascendingSEXP);
    __result = Rcpp::wrap(isSorted(dataFrame, indexes, ascending));
    return __result;
END_RCPP
}
// isSortedVectorList
bool isSortedVectorList(const List& vectorList, const std::vector<bool>& ascending);
RcppExport SEXP Cyclops_isSortedVectorList(SEXP vectorListSEXP, SEXP ascendingSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const List& >::type vectorList(vectorListSEXP);
    Rcpp::traits::input_parameter< const std::vector<bool>& >::type ascending(ascendingSEXP);
    __result = Rcpp::wrap(isSortedVectorList(vectorList, ascending));
    return __result;
END_RCPP
}
// cyclopsPrintRowIds
void cyclopsPrintRowIds(Environment object);
RcppExport SEXP Cyclops_cyclopsPrintRowIds(SEXP objectSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type object(objectSEXP);
    cyclopsPrintRowIds(object);
    return R_NilValue;
END_RCPP
}
// isRcppPtrNull
bool isRcppPtrNull(SEXP x);
RcppExport SEXP Cyclops_isRcppPtrNull(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type x(xSEXP);
    __result = Rcpp::wrap(isRcppPtrNull(x));
    return __result;
END_RCPP
}
// cyclopsGetNumberOfStrata
int cyclopsGetNumberOfStrata(Environment object);
RcppExport SEXP Cyclops_cyclopsGetNumberOfStrata(SEXP objectSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type object(objectSEXP);
    __result = Rcpp::wrap(cyclopsGetNumberOfStrata(object));
    return __result;
END_RCPP
}
// cyclopsGetCovariateIds
std::vector<int64_t> cyclopsGetCovariateIds(Environment object);
RcppExport SEXP Cyclops_cyclopsGetCovariateIds(SEXP objectSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type object(objectSEXP);
    __result = Rcpp::wrap(cyclopsGetCovariateIds(object));
    return __result;
END_RCPP
}
// cyclopsGetCovariateType
CharacterVector cyclopsGetCovariateType(Environment object, const std::vector<int64_t>& covariateLabel);
RcppExport SEXP Cyclops_cyclopsGetCovariateType(SEXP objectSEXP, SEXP covariateLabelSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type object(objectSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type covariateLabel(covariateLabelSEXP);
    __result = Rcpp::wrap(cyclopsGetCovariateType(object, covariateLabel));
    return __result;
END_RCPP
}
// cyclopsGetNumberOfColumns
int cyclopsGetNumberOfColumns(Environment object);
RcppExport SEXP Cyclops_cyclopsGetNumberOfColumns(SEXP objectSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type object(objectSEXP);
    __result = Rcpp::wrap(cyclopsGetNumberOfColumns(object));
    return __result;
END_RCPP
}
// cyclopsGetNumberOfRows
int cyclopsGetNumberOfRows(Environment object);
RcppExport SEXP Cyclops_cyclopsGetNumberOfRows(SEXP objectSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type object(objectSEXP);
    __result = Rcpp::wrap(cyclopsGetNumberOfRows(object));
    return __result;
END_RCPP
}
// cyclopsGetNumberOfTypes
int cyclopsGetNumberOfTypes(Environment object);
RcppExport SEXP Cyclops_cyclopsGetNumberOfTypes(SEXP objectSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type object(objectSEXP);
    __result = Rcpp::wrap(cyclopsGetNumberOfTypes(object));
    return __result;
END_RCPP
}
// cyclopsUnivariableCorrelation
std::vector<double> cyclopsUnivariableCorrelation(Environment x, const std::vector<long>& covariateLabel);
RcppExport SEXP Cyclops_cyclopsUnivariableCorrelation(SEXP xSEXP, SEXP covariateLabelSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<long>& >::type covariateLabel(covariateLabelSEXP);
    __result = Rcpp::wrap(cyclopsUnivariableCorrelation(x, covariateLabel));
    return __result;
END_RCPP
}
// cyclopsSumByGroup
List cyclopsSumByGroup(Environment x, const std::vector<long>& covariateLabel, const long groupByLabel, const int power);
RcppExport SEXP Cyclops_cyclopsSumByGroup(SEXP xSEXP, SEXP covariateLabelSEXP, SEXP groupByLabelSEXP, SEXP powerSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<long>& >::type covariateLabel(covariateLabelSEXP);
    Rcpp::traits::input_parameter< const long >::type groupByLabel(groupByLabelSEXP);
    Rcpp::traits::input_parameter< const int >::type power(powerSEXP);
    __result = Rcpp::wrap(cyclopsSumByGroup(x, covariateLabel, groupByLabel, power));
    return __result;
END_RCPP
}
// cyclopsSumByStratum
List cyclopsSumByStratum(Environment x, const std::vector<long>& covariateLabel, const int power);
RcppExport SEXP Cyclops_cyclopsSumByStratum(SEXP xSEXP, SEXP covariateLabelSEXP, SEXP powerSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<long>& >::type covariateLabel(covariateLabelSEXP);
    Rcpp::traits::input_parameter< const int >::type power(powerSEXP);
    __result = Rcpp::wrap(cyclopsSumByStratum(x, covariateLabel, power));
    return __result;
END_RCPP
}
// cyclopsSum
std::vector<double> cyclopsSum(Environment x, const std::vector<long>& covariateLabel, const int power);
RcppExport SEXP Cyclops_cyclopsSum(SEXP xSEXP, SEXP covariateLabelSEXP, SEXP powerSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<long>& >::type covariateLabel(covariateLabelSEXP);
    Rcpp::traits::input_parameter< const int >::type power(powerSEXP);
    __result = Rcpp::wrap(cyclopsSum(x, covariateLabel, power));
    return __result;
END_RCPP
}
// cyclopsNewSqlData
List cyclopsNewSqlData(const std::string& modelTypeName, const std::string& noiseLevel);
RcppExport SEXP Cyclops_cyclopsNewSqlData(SEXP modelTypeNameSEXP, SEXP noiseLevelSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const std::string& >::type modelTypeName(modelTypeNameSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type noiseLevel(noiseLevelSEXP);
    __result = Rcpp::wrap(cyclopsNewSqlData(modelTypeName, noiseLevel));
    return __result;
END_RCPP
}
// cyclopsSetHasIntercept
void cyclopsSetHasIntercept(Environment x, bool hasIntercept);
RcppExport SEXP Cyclops_cyclopsSetHasIntercept(SEXP xSEXP, SEXP hasInterceptSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< bool >::type hasIntercept(hasInterceptSEXP);
    cyclopsSetHasIntercept(x, hasIntercept);
    return R_NilValue;
END_RCPP
}
// cyclopsGetHasIntercept
bool cyclopsGetHasIntercept(Environment x);
RcppExport SEXP Cyclops_cyclopsGetHasIntercept(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    __result = Rcpp::wrap(cyclopsGetHasIntercept(x));
    return __result;
END_RCPP
}
// cyclopsGetHasOffset
bool cyclopsGetHasOffset(Environment x);
RcppExport SEXP Cyclops_cyclopsGetHasOffset(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    __result = Rcpp::wrap(cyclopsGetHasOffset(x));
    return __result;
END_RCPP
}
// cyclopsGetMeanOffset
double cyclopsGetMeanOffset(Environment x);
RcppExport SEXP Cyclops_cyclopsGetMeanOffset(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    __result = Rcpp::wrap(cyclopsGetMeanOffset(x));
    return __result;
END_RCPP
}
// cyclopsFinalizeData
void cyclopsFinalizeData(Environment x, bool addIntercept, SEXP sexpOffsetCovariate, bool offsetAlreadyOnLogScale, bool sortCovariates, SEXP sexpCovariatesDense, bool magicFlag);
RcppExport SEXP Cyclops_cyclopsFinalizeData(SEXP xSEXP, SEXP addInterceptSEXP, SEXP sexpOffsetCovariateSEXP, SEXP offsetAlreadyOnLogScaleSEXP, SEXP sortCovariatesSEXP, SEXP sexpCovariatesDenseSEXP, SEXP magicFlagSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< bool >::type addIntercept(addInterceptSEXP);
    Rcpp::traits::input_parameter< SEXP >::type sexpOffsetCovariate(sexpOffsetCovariateSEXP);
    Rcpp::traits::input_parameter< bool >::type offsetAlreadyOnLogScale(offsetAlreadyOnLogScaleSEXP);
    Rcpp::traits::input_parameter< bool >::type sortCovariates(sortCovariatesSEXP);
    Rcpp::traits::input_parameter< SEXP >::type sexpCovariatesDense(sexpCovariatesDenseSEXP);
    Rcpp::traits::input_parameter< bool >::type magicFlag(magicFlagSEXP);
    cyclopsFinalizeData(x, addIntercept, sexpOffsetCovariate, offsetAlreadyOnLogScale, sortCovariates, sexpCovariatesDense, magicFlag);
    return R_NilValue;
END_RCPP
}
// cyclopsLoadDataY
void cyclopsLoadDataY(Environment x, const std::vector<int64_t>& stratumId, const std::vector<int64_t>& rowId, const std::vector<double>& y, const std::vector<double>& time);
RcppExport SEXP Cyclops_cyclopsLoadDataY(SEXP xSEXP, SEXP stratumIdSEXP, SEXP rowIdSEXP, SEXP ySEXP, SEXP timeSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type stratumId(stratumIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type rowId(rowIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type time(timeSEXP);
    cyclopsLoadDataY(x, stratumId, rowId, y, time);
    return R_NilValue;
END_RCPP
}
// cyclopsLoadDataMultipleX
int cyclopsLoadDataMultipleX(Environment x, const std::vector<int64_t>& covariateId, const std::vector<int64_t>& rowId, const std::vector<double>& covariateValue, const bool checkCovariateIds, const bool checkCovariateBounds, const bool append, const bool forceSparse);
RcppExport SEXP Cyclops_cyclopsLoadDataMultipleX(SEXP xSEXP, SEXP covariateIdSEXP, SEXP rowIdSEXP, SEXP covariateValueSEXP, SEXP checkCovariateIdsSEXP, SEXP checkCovariateBoundsSEXP, SEXP appendSEXP, SEXP forceSparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type covariateId(covariateIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type rowId(rowIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type covariateValue(covariateValueSEXP);
    Rcpp::traits::input_parameter< const bool >::type checkCovariateIds(checkCovariateIdsSEXP);
    Rcpp::traits::input_parameter< const bool >::type checkCovariateBounds(checkCovariateBoundsSEXP);
    Rcpp::traits::input_parameter< const bool >::type append(appendSEXP);
    Rcpp::traits::input_parameter< const bool >::type forceSparse(forceSparseSEXP);
    __result = Rcpp::wrap(cyclopsLoadDataMultipleX(x, covariateId, rowId, covariateValue, checkCovariateIds, checkCovariateBounds, append, forceSparse));
    return __result;
END_RCPP
}
// cyclopsLoadDataX
int cyclopsLoadDataX(Environment x, const int64_t covariateId, const std::vector<int64_t>& rowId, const std::vector<double>& covariateValue, const bool replace, const bool append, const bool forceSparse);
RcppExport SEXP Cyclops_cyclopsLoadDataX(SEXP xSEXP, SEXP covariateIdSEXP, SEXP rowIdSEXP, SEXP covariateValueSEXP, SEXP replaceSEXP, SEXP appendSEXP, SEXP forceSparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< const int64_t >::type covariateId(covariateIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type rowId(rowIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type covariateValue(covariateValueSEXP);
    Rcpp::traits::input_parameter< const bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< const bool >::type append(appendSEXP);
    Rcpp::traits::input_parameter< const bool >::type forceSparse(forceSparseSEXP);
    __result = Rcpp::wrap(cyclopsLoadDataX(x, covariateId, rowId, covariateValue, replace, append, forceSparse));
    return __result;
END_RCPP
}
// cyclopsAppendSqlData
int cyclopsAppendSqlData(Environment x, const std::vector<int64_t>& oStratumId, const std::vector<int64_t>& oRowId, const std::vector<double>& oY, const std::vector<double>& oTime, const std::vector<int64_t>& cRowId, const std::vector<int64_t>& cCovariateId, const std::vector<double>& cCovariateValue);
RcppExport SEXP Cyclops_cyclopsAppendSqlData(SEXP xSEXP, SEXP oStratumIdSEXP, SEXP oRowIdSEXP, SEXP oYSEXP, SEXP oTimeSEXP, SEXP cRowIdSEXP, SEXP cCovariateIdSEXP, SEXP cCovariateValueSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type oStratumId(oStratumIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type oRowId(oRowIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type oY(oYSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type oTime(oTimeSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type cRowId(cRowIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<int64_t>& >::type cCovariateId(cCovariateIdSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type cCovariateValue(cCovariateValueSEXP);
    __result = Rcpp::wrap(cyclopsAppendSqlData(x, oStratumId, oRowId, oY, oTime, cRowId, cCovariateId, cCovariateValue));
    return __result;
END_RCPP
}
// cyclopsGetInterceptLabel
SEXP cyclopsGetInterceptLabel(Environment x);
RcppExport SEXP Cyclops_cyclopsGetInterceptLabel(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Environment >::type x(xSEXP);
    __result = Rcpp::wrap(cyclopsGetInterceptLabel(x));
    return __result;
END_RCPP
}
// cyclopsReadFileData
List cyclopsReadFileData(const std::string& fileName, const std::string& modelTypeName);
RcppExport SEXP Cyclops_cyclopsReadFileData(SEXP fileNameSEXP, SEXP modelTypeNameSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const std::string& >::type fileName(fileNameSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type modelTypeName(modelTypeNameSEXP);
    __result = Rcpp::wrap(cyclopsReadFileData(fileName, modelTypeName));
    return __result;
END_RCPP
}
// cyclopsModelData
List cyclopsModelData(SEXP pid, SEXP y, SEXP z, SEXP offs, SEXP dx, SEXP sx, SEXP ix, const std::string& modelTypeName, bool useTimeAsOffset, int numTypes);
RcppExport SEXP Cyclops_cyclopsModelData(SEXP pidSEXP, SEXP ySEXP, SEXP zSEXP, SEXP offsSEXP, SEXP dxSEXP, SEXP sxSEXP, SEXP ixSEXP, SEXP modelTypeNameSEXP, SEXP useTimeAsOffsetSEXP, SEXP numTypesSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type pid(pidSEXP);
    Rcpp::traits::input_parameter< SEXP >::type y(ySEXP);
    Rcpp::traits::input_parameter< SEXP >::type z(zSEXP);
    Rcpp::traits::input_parameter< SEXP >::type offs(offsSEXP);
    Rcpp::traits::input_parameter< SEXP >::type dx(dxSEXP);
    Rcpp::traits::input_parameter< SEXP >::type sx(sxSEXP);
    Rcpp::traits::input_parameter< SEXP >::type ix(ixSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type modelTypeName(modelTypeNameSEXP);
    Rcpp::traits::input_parameter< bool >::type useTimeAsOffset(useTimeAsOffsetSEXP);
    Rcpp::traits::input_parameter< int >::type numTypes(numTypesSEXP);
    __result = Rcpp::wrap(cyclopsModelData(pid, y, z, offs, dx, sx, ix, modelTypeName, useTimeAsOffset, numTypes));
    return __result;
END_RCPP
}
