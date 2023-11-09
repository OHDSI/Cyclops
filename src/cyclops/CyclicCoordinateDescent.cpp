/*
 * CyclicCoordinateDescent.cpp
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>
#include <list>

//#include "Rcpp.h"

#include "CyclicCoordinateDescent.h"
#include "Iterators.h"
#include "Timing.h"

#include "priors/CovariatePrior.h"

namespace bsccs {

using namespace std; // TODO Bad form

CyclicCoordinateDescent::CyclicCoordinateDescent(
			//ModelData* reader,
			const AbstractModelData& reader,
			AbstractModelSpecifics& specifics,
			priors::JointPriorPtr prior,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error
		) : privateModelSpecifics(nullptr), modelSpecifics(specifics), jointPrior(prior),
		 hXI(reader), // hXBeta(modelSpecifics.getXBeta()), hXBetaSave(modelSpecifics.getXBetaSave()), // TODO Remove
		 logger(_logger), error(_error) {
	N = hXI.getNumberOfPatients();
	K = hXI.getNumberOfRows();
	J = hXI.getNumberOfCovariates();

// 	hXI = reader;
	// hY = hXI.getYVector(); // TODO Delegate all data to ModelSpecifics
//	hOffs = reader->getOffsetVector();
	// hPid = hXI.getPidVector();

	//conditionId = hXI.getConditionId();

	updateCount = 0;
	likelihoodCount = 0;
	noiseLevel = NOISY;
	initialBound = 2.0;

	priorType = priors::NONE;

	init(hXI.getHasOffsetCovariate());
}

CyclicCoordinateDescent* CyclicCoordinateDescent::clone(ComputeDeviceArguments computeDevice) {
	return new (std::nothrow) CyclicCoordinateDescent(*this, computeDevice);
}

//template <typename T>
//struct GetType<T>;
CyclicCoordinateDescent::CyclicCoordinateDescent(const CyclicCoordinateDescent& copy, ComputeDeviceArguments computeDevice)
	: privateModelSpecifics(
			bsccs::unique_ptr<AbstractModelSpecifics>(
				copy.modelSpecifics.clone(computeDevice))), // deep copy
	  modelSpecifics(*privateModelSpecifics),
      jointPrior(copy.jointPrior), // swallow
      hXI(copy.hXI), // swallow
// 	  hXBeta(modelSpecifics.getXBeta()), hXBetaSave(modelSpecifics.getXBetaSave()), // TODO Remove
// 	  jointPrior(priors::JointPriorPtr(copy.jointPrior->clone())), // deep copy
	  logger(copy.logger), error(copy.error) {

	N = hXI.getNumberOfPatients();
	K = hXI.getNumberOfRows();
	J = hXI.getNumberOfCovariates();

	// hY = hXI.getYVector(); // TODO Delegate all data to ModelSpecifics
	// hPid = hXI.getPidVector();

	// conditionId = hXI.getConditionId();

	updateCount = 0;
	likelihoodCount = 0;
	noiseLevel = copy.noiseLevel;
	initialBound = copy.initialBound;

	priorType = copy.priorType;

	init(hXI.getHasOffsetCovariate());

	if (copy.hWeights.size() > 0) {

	    std::vector<double> buffer;
	    buffer.resize(copy.hWeights.size());
	    std::copy(std::begin(copy.hWeights), std::end(copy.hWeights), std::begin(buffer));

	    setWeights(buffer.data());
	    checkAllLazyFlags();
	}

	// ESK: Copy hWeights functionality for cWeights
	if (copy.cWeights.size() > 0) {

	    std::vector<double> buffer;
	    buffer.resize(copy.cWeights.size());
	    std::copy(std::begin(copy.cWeights), std::end(copy.cWeights), std::begin(buffer));

	    setCensorWeights(buffer.data());
	    checkAllLazyFlags();
	}

	// Copy over exisiting beta;
	bool allBetaZero = true;
	for (int j = 0; j < J; ++j) {
	    hBeta[j] = copy.hBeta[j];
	    if (copy.hBeta[j] != 0.0) allBetaZero = false;
	}
	xBetaKnown = allBetaZero;
}

CyclicCoordinateDescent::~CyclicCoordinateDescent(void) {

//	free(hPid);
//	free(hNEvents);
//	free(hY);
//	free(hOffs);

//	free(hBeta);
//	free(hXBeta);
//	free(hXBetaSave);
//	free(hDelta);

// #ifdef TEST_ROW_INDEX
// 	for (int j = 0; j < J; ++j) {
// 		if (hXColumnRowIndicators[j]) {
// 			free(hXColumnRowIndicators[j]);
// 		}
// 	}
// 	free(hXColumnRowIndicators);
// #endif

//	free(hXjY);
//	free(offsExpXBeta);
//	free(xOffsExpXBeta);
//	free(denomPid);  // Nested in numerPid allocation
//	free(numerPid);
//	free(t1);

// #ifdef NO_FUSE
// 	free(wPid);
// #endif

// 	if (hWeights) {
// 		free(hWeights);
// 	}

#ifdef SPARSE_PRODUCT
// 	for (std::vector<std::vector<int>* >::iterator it = sparseIndices.begin();
// 			it != sparseIndices.end(); ++it) {
// 		if (*it) {
// 			delete *it;
// 		}
// 	}
#endif
}

void CyclicCoordinateDescent::setNoiseLevel(NoiseLevels noise) {
	noiseLevel = noise;
}

string CyclicCoordinateDescent::getPriorInfo() const {
	return jointPrior->getDescription();
}

void CyclicCoordinateDescent::setCrossValidationInfo(string info) {
    crossValidationInfo = info;
}

string CyclicCoordinateDescent::getCrossValidationInfo() const {
    return crossValidationInfo;
}

void CyclicCoordinateDescent::setPrior(priors::JointPriorPtr newPrior) {
    jointPrior = newPrior;
}

void CyclicCoordinateDescent::setInitialBound(double bound) {
    initialBound = bound;
}

void CyclicCoordinateDescent::resetBounds() {
	for (int j = 0; j < J; j++) {
		hDelta[j] = initialBound;
	}
}

void CyclicCoordinateDescent::init(bool offset) {

	// Set parameters and statistics space
	hDelta.resize(J, static_cast<double>(initialBound));
	hBeta.resize(J, static_cast<double>(0.0));

	// initialize starting betas to default value 0.0
	startingBeta.resize(J, static_cast<double>(0.0));

// 	hXBeta.resize(K, static_cast<double>(0.0));
// 	hXBetaSave.resize(K, static_cast<double>(0.0));

	fixBeta.resize(J, false);

	hWeights.resize(0);
    cWeights.resize(0); // ESK: For censor weights

	useCrossValidation = false;
	validWeights = false;
	sufficientStatisticsKnown = false;
	fisherInformationKnown = false;
	varianceKnown = false;
	if (offset) {
		hBeta[0] = static_cast<double>(1);
		startingBeta[0] = static_cast<double>(1);
		fixBeta[0] = true;
		xBetaKnown = false;
	} else {
		xBetaKnown = true; // all beta = 0 => xBeta = 0
	}
	doLogisticRegression = false;

	modelSpecifics.initialize(N, K, J,
                           //&hXI,
                           nullptr,
                           NULL, NULL, NULL,
			NULL, NULL,
			NULL, // hPid,
			NULL,
			NULL, NULL,
			NULL,
			NULL
		//	hY
			);

	usingGPU = modelSpecifics.isGPU();
	usingCUDA = modelSpecifics.isCUDA();
	resetBeta();
}

int CyclicCoordinateDescent::getAlignedLength(int N) {
	return (N / 16) * 16 + (N % 16 == 0 ? 0 : 16);
}

void CyclicCoordinateDescent::computeNEvents() {
	if (syncCV) {
		for (int i=0; i<syncCVFolds; i++) {
			modelSpecifics.setWeights(
					hWeightsPool[i].size() > 0 ? hWeightsPool[i].data() : nullptr,
							useCrossValidation, i);
		}
	} else {
		modelSpecifics.setWeights(
				hWeights.size() > 0 ? hWeights.data() : nullptr,
				cWeights.size() > 0 ? cWeights.data() : nullptr,
				useCrossValidation);
	}
/*
	//modelSpecifics.setWeights(
	//	hWeights.size() > 0 ? hWeights.data() : nullptr,
	//	useCrossValidation);
	// ESK: My version to incorporate censor weights as well
	modelSpecifics.setWeights(
	    hWeights.size() > 0 ? hWeights.data() : nullptr,
	    cWeights.size() > 0 ? cWeights.data() : nullptr,
	    useCrossValidation);
*/
}

void CyclicCoordinateDescent::resetBeta(void) {

    auto start = hXI.getHasOffsetCovariate() ? 1 : 0;
    for (auto j = start; j < J; j++) {

    	// check if a non-zero starting beta is present
    	if (startingBeta[j] != 0.0) {
    		hBeta[j] = startingBeta[j];
    	} else {
    		hBeta[j] = 0.0;
    	}
    }
	if (usingGPU || usingCUDA) {
		modelSpecifics.resetBeta();
	}
	computeXBeta();
	sufficientStatisticsKnown = false;
}

void CyclicCoordinateDescent::logResults(const char* fileName, bool withASE) {

	ofstream outLog(fileName);
	if (!outLog) {
	    std::ostringstream stream;
		stream << "Unable to open log file: " << fileName;
		error->throwError(stream);
	}
	string sep(","); // TODO Make option

	outLog << "label"
			<< sep << "estimate"
			//<< sep << "score"
			;
	if (withASE) {
		outLog << sep << "ASE";
	}
	outLog << endl;

	for (int i = 0; i < J; i++) {
		outLog << hXI.getColumnLabel(i)
//				<< sep << conditionId
				<< sep << hBeta[i];
		if (withASE) {
			double ASE = sqrt(getAsymptoticVariance(i,i));
			outLog << sep << ASE;
		}
		outLog << endl;
	}
	outLog.flush();
	outLog.close();
}

// double CyclicCoordinateDescent::getPredictiveLogLikelihood(double* weights) {
//
//     xBetaKnown = false;
//
//     if (!xBetaKnown) {
//         computeXBeta();
//         xBetaKnown = true;
//         sufficientStatisticsKnown = false;
//     }
//
//     if (!sufficientStatisticsKnown) {
//         computeRemainingStatistics(true, 0); // TODO Remove index????
//         sufficientStatisticsKnown = true;
//     }
//
//     getDenominators();
//
//     return modelSpecifics.getPredictiveLogLikelihood(weights); // TODO Pass double
// }

double CyclicCoordinateDescent::getNewPredictiveLogLikelihood(double* weights) {

//     std::vector<double> savedWeights;
//
// 	if (weights != nullptr) {
// 	    savedWeights = hWeights; // Save original
// 	    setWeights(weights);
// 	    // modelSpecifics.setWeights(weights, true);
// 	}
//
// 	auto result = getLogLikelihood();
// 	// auto result = modelSpecifics.getPredictiveLogLikelihood(weights);
//
// 	if (weights != nullptr) {
// 	    setWeights(savedWeights.data());
// 	    // modelSpecifics.setWeights(savedWeights.data(), true);
// 	}
//
// 	return result;

    xBetaKnown = false;

    if (!xBetaKnown) {
        computeXBeta();
        xBetaKnown = true;
        sufficientStatisticsKnown = false;
    }

    if (!sufficientStatisticsKnown) {
        computeRemainingStatistics(true, 0); // TODO Remove index????
        sufficientStatisticsKnown = true;
    }

    getDenominators();

    return modelSpecifics.getPredictiveLogLikelihood(weights); // TODO Pass double
}

void CyclicCoordinateDescent::getPredictiveEstimates(double* y, double* weights) const {
	modelSpecifics.getPredictiveEstimates(y, weights);
}

int CyclicCoordinateDescent::getBetaSize(void) {
	return J;
}

int CyclicCoordinateDescent::getPredictionSize(void) const {
	return K;
}

bool CyclicCoordinateDescent::getIsRegularized(int i) const {
    return jointPrior->getIsRegularized(i);
}

double CyclicCoordinateDescent::getBeta(int i) {
	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, i);
	}
	return static_cast<double>(hBeta[i]);
}

bool CyclicCoordinateDescent::getFixedBeta(int i) {
	return fixBeta[i];
}

void CyclicCoordinateDescent::setFixedBeta(int i, bool value) {
	fixBeta[i] = value;
}

void CyclicCoordinateDescent::checkAllLazyFlags(void) {
	if (!xBetaKnown) {
		computeXBeta();
		xBetaKnown = true;
		sufficientStatisticsKnown = false;
	}

	if (!validWeights) {
		computeNEvents();
		computeFixedTermsInLogLikelihood();
		computeFixedTermsInGradientAndHessian();
		validWeights = true;
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Check index?
		sufficientStatisticsKnown = true;
	}
}

double CyclicCoordinateDescent::getLogLikelihood(void) {

	checkAllLazyFlags();

	getDenominators();
	likelihoodCount += 1;

	return modelSpecifics.getLogLikelihood(useCrossValidation);
}

void CyclicCoordinateDescent::getDenominators() {
	// Do nothing
}

double convertVarianceToHyperparameter(double value) {
	return sqrt(2.0 / value);
}

double convertHyperparameterToVariance(double value) {
	return 2.0 / (value * value);
}

void CyclicCoordinateDescent::setHyperprior(int index, double value) {
    jointPrior->setVariance(index, value);
    if (usingGPU) {
		std::vector<double> varianceList = jointPrior->getVariance();
		std::vector<double> temp;
		temp.resize(J, 0.0);
		for (int i=0; i<J; i++) {
			int type = jointPrior->getPriorType(i);
			if (type == 1) {
				temp[i] = convertVarianceToHyperparameter(varianceList[0]);
			}
			if (type == 2) {
				temp[i] = varianceList[0];
			}
		}
		modelSpecifics.setPriorParams(temp);
    }
}

// TODO Depricate
void CyclicCoordinateDescent::setHyperprior(double value) {
	jointPrior->setVariance(0, value);
}

//Hierarchical Support
void CyclicCoordinateDescent::setClassHyperprior(double value) {
	jointPrior->setVariance(1,value);
}

std::vector<double> CyclicCoordinateDescent::getHyperprior(void) const {
	return jointPrior->getVariance();
}

void CyclicCoordinateDescent::makeDirty(void) {
	xBetaKnown = false;
	validWeights = false;
	sufficientStatisticsKnown = false;
}

void CyclicCoordinateDescent::setPriorType(int iPriorType) {
	if (iPriorType < priors::NONE || iPriorType > priors::NORMAL) {
	    std::ostringstream stream;
		stream << "Unknown prior type";
		error->throwError(stream);
	}
	priorType = iPriorType;
}

void CyclicCoordinateDescent::setBeta(const std::vector<double>& beta) {
	for (int j = 0; j < J; ++j) {
		hBeta[j] = beta[j]; // TODO Use std::copy
	}
	xBetaKnown = false;
	sufficientStatisticsKnown = false;
	fisherInformationKnown = false;
	varianceKnown = false;
}

void CyclicCoordinateDescent::getHBeta(void) {
	if (usingCUDA) {
		hBeta = modelSpecifics.getBeta();
	}
}

void CyclicCoordinateDescent::setHXBeta(void) {
	if (usingCUDA) {
		modelSpecifics.setHXBeta();
	}
}

void CyclicCoordinateDescent::setStartingBeta(const std::vector<double>& inStartingBeta) {
	// ToDo: This functionality could be merged into setBeta()
	for (int j = 0; j < J; ++j) {
		startingBeta[j] = inStartingBeta[j];
	}
}

void CyclicCoordinateDescent::setBeta(int i, double beta) {
#define PROCESS_IN_MS
#ifdef PROCESS_IN_MS
	double delta = beta - hBeta[i];
	updateXBeta(delta, i);
#else // Delay and then call computeSufficientStatistics
	SetBetaEntry entry(i, hBeta[i]);
	setBetaList.push_back(entry); // save old value and index
	hBeta[i] = beta;
	xBetaKnown = false;
#endif
	fisherInformationKnown = false;
	varianceKnown = false;
}

std::vector<double> CyclicCoordinateDescent::getWeights() {
    return hWeights; // Makes copy
}

std::vector<double> CyclicCoordinateDescent::getCensorWeights() {
    return cWeights; // Makes copy
}

void CyclicCoordinateDescent::setWeights(double* iWeights) {

	if (iWeights == NULL) {
		if (hWeights.size() != 0) {
			hWeights.resize(0);
		}

		// Turn off weights
		useCrossValidation = false;
		validWeights = false;
		sufficientStatisticsKnown = false;
	} else {

		if (hWeights.size() != static_cast<size_t>(K)) {
			hWeights.resize(K); // = (double*) malloc(sizeof(double) * K);
		}
		for (int i = 0; i < K; ++i) {
			hWeights[i] = iWeights[i];
		}
		useCrossValidation = true;
		validWeights = false;
		sufficientStatisticsKnown = false;
	}
}

void CyclicCoordinateDescent::setCensorWeights(double* dWeights) {
    if (dWeights == NULL) {
        if (cWeights.size() != 0) {
            cWeights.resize(0);
        }
    } else {

        if (cWeights.size() != static_cast<size_t>(K)) {
            cWeights.resize(K);
        }
        for (int i = 0; i < K; ++i) {
            cWeights[i] = dWeights[i];
        }
    }
}

double CyclicCoordinateDescent::getLogPrior(void) {
	return jointPrior->logDensity(hBeta, *this);
}

double CyclicCoordinateDescent::getObjectiveFunction(int convergenceType) {
	if (convergenceType == GRADIENT) {
	    return modelSpecifics.getGradientObjective(useCrossValidation);
	} else
	if (convergenceType == MITTAL) {
		return getLogLikelihood();
	} else
	if (convergenceType == LANGE) {
		return getLogLikelihood() + getLogPrior();
	} else if (convergenceType == ONE_STEP) {
	    return 0.0;
	} else {
    	std::ostringstream stream;
    	stream << "Invalid convergence type: " << convergenceType;
    	error->throwError(stream);
    }
    return 0.0;
}

double CyclicCoordinateDescent::computeZhangOlesConvergenceCriterion(void) {
	double sumAbsDiffs = 0;
	double sumAbsResiduals = 0;

	auto& hXBeta = modelSpecifics.getXBeta();
	auto& hXBetaSave = modelSpecifics.getXBetaSave();

	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			sumAbsDiffs += abs(hXBeta[i] - hXBetaSave[i]) * hWeights[i];
			sumAbsResiduals += abs(hXBeta[i]) * hWeights[i];
		}
	} else {
		for (int i = 0; i < K; i++) {
			sumAbsDiffs += abs(hXBeta[i] - hXBetaSave[i]);
			sumAbsResiduals += abs(hXBeta[i]);
		}
	}
	return sumAbsDiffs / (1.0 + sumAbsResiduals);
}

void CyclicCoordinateDescent::saveXBeta(void) {
    modelSpecifics.saveXBeta();
	// memcpy(hXBetaSave.data(), hXBeta.data(), K * sizeof(double));
}

void CyclicCoordinateDescent::update(const ModeFindingArguments& arguments) {

	const auto maxIterations = arguments.maxIterations;
	const auto convergenceType = arguments.convergenceType;
	const auto epsilon = arguments.tolerance;
	const int maxCount = arguments.maxBoundCount;
	const auto algorthmType = arguments.algorithmType;
	const bool doItAll = arguments.doItAll;
	const int qnQ = 0;

	initialBound = arguments.initialBound;

	int count = 0;
	bool done = false;

	if (syncCV) {
		donePool.resize(syncCVFolds);
		for (int i=0; i<syncCVFolds; i++) {
			donePool[i] = false;
		}
		if (usingGPU) modelSpecifics.updateDoneFolds(donePool);
	}
//	for (int i=0; i<J; i++) {
//		fixBeta[i] = false;
//	}


	while (!done) {
 	    if (arguments.useKktSwindle && jointPrior->getSupportsKktSwindle()) {
		    kktSwindle(arguments);
	    } else {
		    findMode(maxIterations, convergenceType, epsilon, algorthmType, qnQ, doItAll);
	    }
	    ++count;

	    if (lastReturnFlag == ILLCONDITIONED && count < maxCount) {
	        // Reset beta and shrink bounding box
	        initialBound /= 10.0;
            resetBeta();
	    } else {
	        done = true;
	    }
	}
}

typedef std::tuple<
	int,    // index
	double, // gradient
	bool    // force active
> ScoreTuple;

template <typename Iterator>
void CyclicCoordinateDescent::findMode(Iterator begin, Iterator end,
		const int maxIterations, const int convergenceType, const double epsilon,
		const AlgorithmType algorithmType, const int qnQ, const bool doItAll) {

	std::fill(fixBeta.begin(), fixBeta.end(), true);
 	std::for_each(begin, end, [this] (ScoreTuple& tuple) {
 		fixBeta[std::get<0>(tuple)] = false;
 	});
	findMode(maxIterations, convergenceType, epsilon, algorithmType, qnQ, doItAll);
    // fixBeta is no longer valid
}

void CyclicCoordinateDescent::kktSwindle(const ModeFindingArguments& arguments) {

	const auto maxIterations = arguments.maxIterations;
	const auto convergenceType = arguments.convergenceType;
	const auto epsilon = arguments.tolerance;
	const auto algorithmType = arguments.algorithmType;
	const auto doItAll = arguments.doItAll;
	const int qnQ = 0;

	// Make sure internal state is up-to-date
	checkAllLazyFlags();


	std::list<ScoreTuple> activeSet;
	std::list<ScoreTuple> inactiveSet;
	std::list<int> excludeSet;

	// Initialize sets
	int intercept = -1;
	if (hXI.getHasInterceptCovariate()) {
		intercept = hXI.getHasOffsetCovariate() ? 1 : 0;
	}

	for (int index = 0; index < J; ++index) {
		if (fixBeta[index]) {
			excludeSet.push_back(index);
		} else {
			if (index == intercept || // Always place intercept into active set
                !jointPrior->getSupportsKktSwindle(index)) {
// 				activeSet.push_back(index);
				activeSet.push_back(std::make_tuple(index, 0.0, true));
			} else {
				inactiveSet.push_back(std::make_tuple(index, 0.0, false));
			}
		}
	}

	bool done = false;
	int swindleIterationCount = 1;

//	int initialActiveSize = activeSet.size();
	// int perPassSize = arguments.swindleMultipler;

	while (!done) {

		if (noiseLevel >= QUIET) {
			std::ostringstream stream;
			stream << "\nKKT Swindle count " << swindleIterationCount << ", activeSet size =  " << activeSet.size();
			logger->writeLine(stream);
		}

		// Enforce all beta[inactiveSet] = 0
		for (auto& inactive : inactiveSet) {
			if (getBeta(std::get<0>(inactive)) != 0.0) { // Touch only if necessary
				setBeta(std::get<0>(inactive), 0.0);
			}
		}

		double updateTime = 0.0;
		lastReturnFlag = SUCCESS;
		if (activeSet.size() > 0) { // find initial mode
			auto start = bsccs::chrono::steady_clock::now();

			findMode(begin(activeSet), end(activeSet), maxIterations, convergenceType, epsilon,
            algorithmType, qnQ, doItAll);

			auto end = bsccs::chrono::steady_clock::now();
			bsccs::chrono::duration<double> elapsed_seconds = end-start;
			updateTime = elapsed_seconds.count();
		}

		if (noiseLevel >= QUIET) {
			std::ostringstream stream;
			stream << "update time: " << updateTime << " in " << lastIterationCount << " iterations.";
			logger->writeLine(stream);
		}

		if (inactiveSet.size() == 0 || lastReturnFlag != SUCCESS) { // Computed global mode, nothing more to do, or failed

			done = true;

		} else { // still inactive covariates


			if (swindleIterationCount == maxIterations) {
				lastReturnFlag = MAX_ITERATIONS;
				done = true;
				if (noiseLevel > SILENT) {
					std::ostringstream stream;
					stream << "Reached maximum swindle iterations";
					logger->writeLine(stream);
				}
			} else {

				auto checkConditions = [this] (const ScoreTuple& score) {
					return (std::get<1>(score) <= jointPrior->getKktBoundary(std::get<0>(score)));
				};

//				auto checkAlmostConditions = [this] (const ScoreTuple& score) {
//					return (std::get<1>(score) < 0.9 * jointPrior->getKktBoundary(std::get<0>(score)));
//				};

				// Check KKT conditions

				computeKktConditions(inactiveSet);

				bool satisfied = std::all_of(begin(inactiveSet), end(inactiveSet), checkConditions);

				if (satisfied) {
					done = true;
				} else {
//					auto newActiveSize = initialActiveSize + perPassSize;

					auto count1 = std::distance(begin(inactiveSet), end(inactiveSet));
					auto count2 = std::count_if(begin(inactiveSet), end(inactiveSet), checkConditions);


					// Remove elements from activeSet if less than 90% of boundary
// 					computeKktConditions(activeSet); // TODO Already computed in findMode
//
// // 					for (auto& active : activeSet) {
// // 						std::ostringstream stream;
// // 						stream << "Active: " << std::get<0>(active) << " : " << std::get<1>(active) << " : " << std::get<2>(active);
// // 						logger->writeLine(stream);
// // 					}
//
// 					int countActiveViolations = 0;
// 					while(activeSet.size() > 0 &&
// 						checkAlmostConditions(activeSet.back())
// 					) {
// 						auto& back = activeSet.back();
//
// // 						std::ostringstream stream;
// // 						stream << "Remove: " << std::get<0>(back) << ":" << std::get<1>(back)
// // 						       << " cut @ " << jointPrior->getKktBoundary(std::get<0>(back))
// // 						       << " diff = " << (std::get<1>(back) - jointPrior->getKktBoundary(std::get<0>(back)));
// // 						logger->writeLine(stream);
//
// 						inactiveSet.push_back(back);
// 						activeSet.pop_back();
// 						++countActiveViolations;
// 					}
					// end

					// Move inactive elements into active if KKT conditions are not met
					while (inactiveSet.size() > 0
							&& !checkConditions(inactiveSet.front())
//  							&& activeSet.size() < newActiveSize
					) {
						auto& front = inactiveSet.front();

// 						std::ostringstream stream;
// 						stream << std::get<0>(front) << ":" << std::get<1>(front);
// 						logger->writeLine(stream);

						activeSet.push_back(front);
						inactiveSet.pop_front();
					}

					if (noiseLevel >= QUIET) {
						std::ostringstream stream;
// 						stream << "  Active set violations: " << countActiveViolations << std::endl;
						stream << "Inactive set violations: " << (count1 - count2);
						logger->writeLine(stream);
					}
				}
			}
		}
		++swindleIterationCount;
		// perPassSize *= 2;

		logger->yield();			// This is not re-entrant safe
	}

	// restore fixBeta
	std::fill(fixBeta.begin(), fixBeta.end(), false);
	for (auto index : excludeSet) {
		fixBeta[index] = true;
	}
}

template <typename Container>
void CyclicCoordinateDescent::computeKktConditions(Container& scoreSet) {

    for (auto& score : scoreSet) {
        const auto index = std::get<0>(score);

		computeNumeratorForGradient(index);

		priors::GradientHessian gh;
		computeGradientAndHessian(index, &gh.first, &gh.second);

		std::get<1>(score) = std::abs(gh.first);
    }

    scoreSet.sort([] (ScoreTuple& lhs, ScoreTuple& rhs) -> bool {
    	if (std::get<2>(rhs) == std::get<2>(lhs)) {
			return (std::get<1>(rhs) < std::get<1>(lhs));
    	} else {
    		return(std::get<2>(lhs));
    	}
    });
}

bool CyclicCoordinateDescent::performCheckConvergence(int convergenceType,
                                                      double epsilon,
                                                      int maxIterations,
                                                      int iteration,
                                                      double* lastObjFunc) {
    bool done = false;
    double conv;
    bool illconditioned = false;
    bool poorBlr = false;
    if (convergenceType < ZHANG_OLES) {
        double thisObjFunc = getObjectiveFunction(convergenceType);
        if (thisObjFunc != thisObjFunc) {
            std::ostringstream stream;
            stream << "\nWarning: problem is ill-conditioned for this choice of\n"
                   << "\t prior (" << jointPrior->getDescription() << ") or\n"
                   << "\t initial bounding box (" << initialBound << ")\n"
                   << "Enforcing convergence!";
            logger->writeLine(stream);
            conv = 0.0;
            illconditioned = true;
        } else {
            conv = computeConvergenceCriterion(thisObjFunc, *lastObjFunc);
            if (conv == 0.0 && iteration == 1 && convergenceType != ONE_STEP &&
                !std::all_of(std::begin(fixBeta), std::end(fixBeta), [](bool fixed) { return fixed; })
            ) {
                std::ostringstream stream;
                stream << "\nWarning: BLR gradient is ill-conditioned\n"
                       << "Enforcing convergence!";
                logger->writeLine(stream);
                illconditioned = true;
                poorBlr = true;
            }
        }
        *lastObjFunc = thisObjFunc;
    } else { // ZHANG_OLES
        conv = computeZhangOlesConvergenceCriterion();
        saveXBeta();
    } // Necessary to call getObjFxn or computeZO before getLogLikelihood,
    // since these copy over XBeta

    double thisLogLikelihood = getLogLikelihood();
    double thisLogPrior = getLogPrior();
    double thisLogPost = thisLogLikelihood + thisLogPrior;

    std::ostringstream stream;
    if (noiseLevel > QUIET) {
        // stream << "\n";
        // printVector(&hBeta[0], J, stream);
        stream << "\n";
        stream << "log post: " << thisLogPost
               << " (" << thisLogLikelihood << " + " << thisLogPrior
               << ") (iter:" << iteration << ", conv: " << conv << ") ";
    }

    if (epsilon > 0 && conv < epsilon) {
        if (poorBlr) {
            lastReturnFlag = POOR_BLR_STEP;
        } else if (illconditioned) {
            lastReturnFlag = ILLCONDITIONED;
        } else {
            if (noiseLevel > SILENT) {
                stream << "Reached convergence criterion";
            }
            lastReturnFlag = SUCCESS;
        }
        done = true;
    } else if (iteration == maxIterations) {
        if (noiseLevel > SILENT) {
            stream << "Reached maximum iterations";
        }
        done = true;
        lastReturnFlag = MAX_ITERATIONS;
    }
    if (noiseLevel > QUIET) {
        logger->writeLine(stream);
    }

    logger->yield();

    return done;
}


void CyclicCoordinateDescent::findMode(
		int maxIterations,
		int convergenceType,
		double epsilon,
		AlgorithmType algorithmType,
		int qnQ, bool doItAll
		) {

	if (convergenceType < GRADIENT || convergenceType > ZHANG_OLES) {
	    std::ostringstream stream;
		stream << "Unknown convergence criterion: " << convergenceType;
		error->throwError(stream);
	}

	if (!validWeights || hXI.getTouchedY() // || hXI.getTouchedX()
		) {
		computeNEvents();
		computeFixedTermsInLogLikelihood();
		computeFixedTermsInGradientAndHessian();
		validWeights = true;
		hXI.clean();
	}

	if (!xBetaKnown) {
		computeXBeta();
		xBetaKnown = true;
		sufficientStatisticsKnown = false;
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Check index?
		sufficientStatisticsKnown = true;
	}

	resetBounds();

	bool done = false;
	int iteration = 0;
	double lastObjFunc = 0.0;
	std::vector<double> lastObjFuncVec;

	if (convergenceType < ZHANG_OLES) {
		if (syncCV) {
			lastObjFuncVec = getObjectiveFunctions(convergenceType);
		} else {
			lastObjFunc = getObjectiveFunction(convergenceType);
		}
	} else { // ZHANG_OLES
		saveXBeta();
	}

	std::vector<double> allDelta;
    double lastLogPosterior; // = getLogLikelihood() + getLogPrior();

	if (algorithmType == AlgorithmType::MM) {
	    // Any further initialization necessary?
	    if (allDelta.size() < J) allDelta.resize(J);
	    lastLogPosterior = -10E10;
	}

	if (usingGPU || usingCUDA) {
		modelSpecifics.setBounds(initialBound);
		std::vector<int> priorList;
		std::vector<double> varianceList = jointPrior->getVariance();
		std::vector<double> temp;
		temp.resize(J, 0.0);

		for (int i=0; i<J; i++) {
			int type = jointPrior->getPriorType(i);
			priorList.push_back(type);
			if (type == 1) {
				temp[i] = convertVarianceToHyperparameter(varianceList[0]);
			}
			if (type == 2) {
				temp[i] = varianceList[0];
			}
		}
		modelSpecifics.setPriorTypes(priorList);
		modelSpecifics.setPriorParams(temp);
		//modelSpecifics.resetBeta();
	}
	
	auto cycle = [this,&lastObjFunc,&lastObjFuncVec,&iteration,algorithmType,&allDelta,&doItAll] {
/*
		if (iteration%10==0) {
			std::cout<<"iteration " << iteration << " ";

			if (!syncCV) {
				std::cout << "lastObjFunc: " << lastObjFunc << " ";
			}

			if (syncCV) {
				for (int i=0; i<syncCVFolds; i++) {
					if (!donePool[i]) {
						std::cout << lastObjFuncVec[i] << " ";
					}
				}
			}

			std::cout << "\n";
		}
*/
	    auto log = [this](const int index) {
	        if ( (noiseLevel > QUIET) && ((index+1) % 100 == 0)) {
	            std::ostringstream stream;
	            stream << "Finished variable " << (index+1);
	            logger->writeLine(stream);
	        }
	    };

	    if (algorithmType == AlgorithmType::MM) {
            // Do delta computation in parallel
            mmUpdateAllBeta(allDelta, fixBeta);

	        // for (auto x : allDelta) {
	        //     std::cerr << " " << x;
	        // }
	        // std::cerr << "\n";
	        //Rcpp::stop("A");

            for (int index = 0; index < J; ++index) {
                if (!fixBeta[index]) {
                    double delta = allDelta[index];
                   // delta = applyBounds(delta, index);
                    if (delta != 0.0) {
                        sufficientStatisticsKnown = false;
                        hBeta[index] += delta;

                        // std::cerr << " : " << index << " " << hBeta[index] << " " << delta;

                    // modelSpecifics.axpyXBeta(delta, index); // TODO Do single spMV
                    }
                }
            }
            // std::cerr << "\n";

            // sufficientStatisticsKnown = false;
            modelSpecifics.computeXBeta(hBeta.data(), useCrossValidation);
            computeRemainingStatistics(true, 0);
            sufficientStatisticsKnown = true;


	    } else {

	    	if (usingGPU) {
			//if (usingGPU && syncCV) {
	    		modelSpecifics.runCCD(useCrossValidation, doItAll);
	    		if (!syncCV) {
	    			hBeta = modelSpecifics.getBeta();
	    		}
	    	} else if (usingCUDA) {
	    		for(int index = 0; index < J; index++) {
	    			if (!fixBeta[index]) {
	    				modelSpecifics.updateBetaAndDelta(index, useCrossValidation);
	    			}
	    		}
	    		hBeta = modelSpecifics.getBeta();
	    	} else {

	    		// Do a complete cycle in serial
	    		for(int index = 0; index < J; index++) {

	    			if (!fixBeta[index]) {
						double delta = ccdUpdateBeta(index);
						delta = applyBounds(delta, index);
					//	std::cout << "index: " << index << " delta: " << delta << "\n";
						if (delta != 0.0) {
							sufficientStatisticsKnown = false;
							updateSufficientStatistics(delta, index);
						}	
				}
	    			log(index);
	    		}
	    	}
	    }
	    iteration++;
	};

	auto check = [this,&iteration,&lastObjFunc,algorithmType,&lastLogPosterior,&lastObjFuncVec,
               convergenceType, epsilon,maxIterations] {
                   bool done = false;
                   //		bool checkConvergence = (iteration % J == 0 || iteration == maxIterations);
                   bool checkConvergence = true; // Check after each complete cycle


                   if (algorithmType == AlgorithmType::MM) {
                       double thisLogPosterior = getLogLikelihood() + getLogPrior();
                       if (iteration > 1) {
                           double change = thisLogPosterior - lastLogPosterior;
                           //Rcpp::Rcout << lastLogPosterior << " -> " << thisLogPosterior << " == " << change;

                           if (change < 0.0) {
                               std::ostringstream stream;
                               stream << "Non-increasing!";
                               error->throwError(stream);
                           }
                       }
                       lastLogPosterior = thisLogPosterior;
                   }

                   if (checkConvergence) {
                	   if (syncCV) {
                		   done = performCheckConvergence(convergenceType, epsilon, maxIterations, iteration, lastObjFuncVec);
                	   } else {
                		   done = performCheckConvergence(convergenceType, epsilon, maxIterations, iteration, &lastObjFunc);
                	   }
                   }

                   return done;
               };

	qnQ = 0;

	if (qnQ > 0) { // Use quasi-Newton
	    // using namespace Eigen;
	    //
	    // Eigen::MatrixXd secantsU(J, qnQ);
	    // Eigen::MatrixXd secantsV(J, qnQ);
	    //
	    // VectorXd x(J);
	    //
	    // // Fill initial secants
	    // int countU = 0;
	    // int countV = 0;
	    //
	    // for (int q = 0; q < qnQ; ++q) {
	    //     x = Map<const VectorXd>(hBeta.data(), J); // Make copy
	    //
	    //     cycle();
	    //     done = check();
	    //     if (done) break;
	    //
	    //     if (countU == 0) { // First time through
	    //         secantsU.col(countU) = Map<const VectorXd>(hBeta.data(), J) - x;
	    //         ++countU;
	    //     } else if (countU < qnQ - 1) { // Middle case
	    //         secantsU.col(countU) = Map<const VectorXd>(hBeta.data(), J) - x;
	    //         secantsV.col(countV) = secantsU.col(countU);
	    //         ++countU;
	    //         ++countV;
	    //     } else { // Last time through
	    //         secantsV.col(countV) = Map<const VectorXd>(hBeta.data(), J) - x;
	    //         ++countV;
	    //     }
	    // }
	    //
	    // // 	    std::cerr << secantsU << std::endl << std::endl;
	    // // 	    std::cerr << secantsV << std::endl;
	    //
	    // int newestSecant = qnQ - 1;
	    // int previousSecant = newestSecant - 1;
	    //
	    // while (!done) {
	    //
	    //     // 2 cycles for each QN step
	    //     x = Map<const VectorXd>(hBeta.data(), J); // Make copy
	    //     cycle();
	    //     // 	        done = check();
	    //     // 	        if (done) {
	    //     // 	            std::cerr << "break A" << std::endl;
	    //     // 	            break;
	    //     // 	        }
	    //
	    //     secantsU.col(newestSecant) = Map<const VectorXd>(hBeta.data(), J) - x;
	    //
	    //     VectorXd Fx = Map<const VectorXd>(hBeta.data(), J); // TODO Can remove?
	    //
	    //     x = Map<const VectorXd>(hBeta.data(), J);
	    //     cycle();
	    //     // 	        done = check();
	    //     // 	        if (done) {
	    //     // 	            std::cerr << "break B" << std::endl;
	    //     // 	            break;
	    //     // 	        }
	    //
	    //     secantsV.col(newestSecant) = Map<const VectorXd>(hBeta.data(), J) - x;
	    //
	    //     // Do QN step here
	    //
	    //     //             MatrixXd UtU = secantsU.transpose() * secantsU;
	    //     //             MatrixXd UtV = secantsU.transpose() * secantsV;
	    //     //             MatrixXd M = UtU - UtV;
	    //
	    //     auto M = secantsU.transpose() * (secantsU - secantsV);
	    //
	    //     auto Minv = M.inverse();
	    //
	    //     auto A = secantsU.transpose() * secantsU.col(newestSecant);
	    //     auto B = Minv * A;
	    //     auto C = secantsV * B;
	    //
	    //     // MatrixXd A = secantsV * Minv;
	    //     // MatrixXd B = A * secantsU.transpose();
	    //     // MatrixXd C = B * secantsV.col(newestSecant); // - (x - Fx)
	    //     // MatrixXd C = B * secantsU.col(newestSecant); // TODO Can remove?
	    //
	    //     //     VectorXd xqn = Map<const VectorXd>(hBeta.data(), J) + C;
	    //     VectorXd xqn = Fx + C; // TODO Can remove?
	    //
	    //     // Save CCD solution
	    //     x = Map<const VectorXd>(hBeta.data(), J);
	    //     double ccdObjective = getLogLikelihood() + getLogPrior();
	    //     // double savedLastObjFunc = lastObjFunc;
	    //
	    //     Map<VectorXd>(hBeta.data(), J) = xqn; // Set QN solution
	    //     //             for (int j = 0; j < J; ++j) {
	    //     //                 if (sign(hBeta[j]) == xqn(j)) {
	    //     //                     hBeta[j] = xqn(j);
	    //     //                 } else {
	    //     //                     hBeta[j] = 0.0;
	    //     //                 }
	    //     //             }
	    //
	    //     // xBetaKnown = false;
	    //     // checkAllLazyFlags();
	    //     modelSpecifics.computeXBeta(hBeta.data(), useCrossValidation);
	    //
	    //
	    //     //             done = check();
	    //     //             if (done) {
	    //     //                 std::cerr << "break B" << std::endl;
	    //     //                 break;
	    //     //             }
	    //
	    //     double qnObjective = getLogLikelihood() + getLogPrior();
	    //
	    //     if (ccdObjective > qnObjective) { // Revert
	    //         Map<VectorXd>(hBeta.data(), J) = x; // Set CCD solution
	    //         modelSpecifics.computeXBeta(hBeta.data(), useCrossValidation);
	    //         // xBetaKnown = false;
	    //         // checkAllLazyFlags();
	    //
	    //         double ccd2Objective = getLogLikelihood() + getLogPrior();
	    //
	    //         if (ccdObjective != ccd2Objective) {
	    //             std::cerr << "Poor revert: " << ccdObjective << " != " << ccd2Objective << " diff: "<< (ccdObjective - ccd2Objective) << std::endl;
	    //         }
	    //         // lastObjFunc = savedLastObjFunc;
	    //         std::cerr << "revert" << std::endl;
	    //     } else {
	    //         std::cerr << "accept" << std::endl;
	    //         //                 done = check();
	    //         //                 if (done) {
	    //         //                     std::cerr << "break C" << std::endl;
	    //         //                     break;
	    //         //                 }
	    //     }
	    //
	    //     done = check();
	    //
	    //     // lastObjFunc = getLogLikelihood() + getLogPrior();
	    //
	    //     // Get ready for next secant-pair
	    //     previousSecant = newestSecant;
	    //     newestSecant = (newestSecant + 1) % qnQ;
	    // }
    } else { // No QN
        while (!done) {
            cycle();
            done = check();
        }
    }

	lastIterationCount = iteration;
//	std::cout << "iterations: " << lastIterationCount << " lastObjFunc: " << lastObjFunc << '\n';

	updateCount += 1;

	modelSpecifics.printTiming();

	fisherInformationKnown = false;
	varianceKnown = false;
}

/**
 * Computationally heavy functions
 */

void CyclicCoordinateDescent::computeGradientAndHessian(int index, double *ogradient,
		double *ohessian) {
	modelSpecifics.computeGradientAndHessian(index, ogradient, ohessian, useCrossValidation);
}

void CyclicCoordinateDescent::computeNumeratorForGradient(int index) {
	// Delegate
	modelSpecifics.computeNumeratorForGradient(index, useCrossValidation);
}

void CyclicCoordinateDescent::computeRatiosForGradientAndHessian(int index) {
    std::ostringstream stream;
	stream << "Error!";
	error->throwError(stream);
}

double CyclicCoordinateDescent::getHessianDiagonal(int index) {

	checkAllLazyFlags();
	double g_d1, g_d2;

	computeNumeratorForGradient(index);
	computeGradientAndHessian(index, &g_d1, &g_d2);

	return g_d2;
}

double CyclicCoordinateDescent::getJerkDiagonal(int index) {

    checkAllLazyFlags();

    double jerk;

    computeNumeratorForGradient(index); // TODO Is this necessary?
    modelSpecifics.computeThirdDerivative(index, &jerk, useCrossValidation);

    return jerk;
}

double CyclicCoordinateDescent::getAsymptoticVariance(int indexOne, int indexTwo) {
	checkAllLazyFlags();
	if (!fisherInformationKnown) {
		computeAsymptoticPrecisionMatrix();
		fisherInformationKnown = true;
	}

	if (!varianceKnown) {
		computeAsymptoticVarianceMatrix();
		varianceKnown = true;
	}

	IndexMap::iterator itOne = hessianIndexMap.find(indexOne);
	IndexMap::iterator itTwo = hessianIndexMap.find(indexTwo);

	if (itOne == hessianIndexMap.end() || itTwo == hessianIndexMap.end()) {
		return NAN;
	} else {
		return varianceMatrix(itOne->second, itTwo->second);
	}
}

// double CyclicCoordinateDescent::getAsymptoticPrecision(int indexOne, int indexTwo) {
// 	checkAllLazyFlags();
// 	if (!fisherInformationKnown) {
// 		computeAsymptoticPrecisionMatrix();
// 		fisherInformationKnown = true;
// 	}
//
// 	IndexMap::iterator itOne = hessianIndexMap.find(indexOne);
// 	IndexMap::iterator itTwo = hessianIndexMap.find(indexTwo);
//
// 	if (itOne == hessianIndexMap.end() || itTwo == hessianIndexMap.end()) {
// 		return NAN;
// 	} else {
// 		return hessianMatrix(itOne->second, itTwo->second);
// 	}
// }

CyclicCoordinateDescent::Matrix CyclicCoordinateDescent::computeFisherInformation(const std::vector<IdType>& indices) const {
    Matrix fisherInformation(indices.size(), indices.size());
    for (size_t ii = 0; ii < indices.size(); ++ii) {
        const int i = indices[ii];
        for (size_t jj = ii; jj < indices.size(); ++jj) {
            const int j = indices[jj];
            double element = 0.0;
             modelSpecifics.computeFisherInformation(i, j, &element, useCrossValidation);
            fisherInformation(jj, ii) = fisherInformation(ii, jj) = element;
        }
    }
    return fisherInformation;
}

void CyclicCoordinateDescent::computeAsymptoticPrecisionMatrix(void) {

	typedef std::vector<int> int_vec;
	int_vec indices;
	hessianIndexMap.clear();

	int index = 0;
	for (int j = 0; j < J; ++j) {
		if (!fixBeta[j] &&
				(priorType != priors::LAPLACE || getBeta(j) != 0.0)) {
			indices.push_back(j);
			hessianIndexMap[j] = index;
			index++;
		}
	}

	hessianMatrix.resize(indices.size(), indices.size());
	modelSpecifics.makeDirty(); // clear hessian terms

	for (size_t ii = 0; ii < indices.size(); ++ii) {
		for (size_t jj = ii; jj < indices.size(); ++jj) {
			const int i = indices[ii];
			const int j = indices[jj];
//			std::cerr << "(" << i << "," << j << ")" << std::endl;
			double fisherInformation = 0.0;
			modelSpecifics.computeFisherInformation(i, j, &fisherInformation, useCrossValidation);
//			if (fisherInformation != 0.0) {
				// Add tuple to sparse matrix
//				tripletList.push_back(Triplet<double>(ii,jj,fisherInformation));
//			}
			hessianMatrix(jj,ii) = hessianMatrix(ii,jj) = fisherInformation;

		}
	}

//	sm.setFromTriplets(tripletList.begin(), tripletList.end());
//	cout << sm;
//	auto inv = sm.triangularView<Upper>().solve(dv1);
//	cout << sm.inverse();
//	cout << hessianMatrix << endl;

	// Take inverse
//	hessianMatrix = hessianMatrix.inverse();

//	cout << hessianMatrix << endl;
}

void CyclicCoordinateDescent::computeAsymptoticVarianceMatrix(void) {
	varianceMatrix = hessianMatrix.inverse();
// 	cout << varianceMatrix << endl;
}

void CyclicCoordinateDescent::mmUpdateAllBeta(std::vector<double>& delta,
                                                const std::vector<bool>& fixedBeta) {

    if (!sufficientStatisticsKnown) {
        std::ostringstream stream;
        stream << "Error in state synchronization.";
        error->throwError(stream);
    }

    std::vector<priors::GradientHessian> gh(J); // TODO make once

    // computeNumeratorForGradient(index);

    // priors::GradientHessian gh;
    // computeGradientAndHessian(index, &gh.first, &gh.second);

    modelSpecifics.computeMMGradientAndHessian(gh, fixedBeta,
                                               useCrossValidation);

    double scale = 1.0; // TODO Tune, somehow?

    for (int j = 0; j < J; ++j) {
        if (!fixedBeta[j]) {
            if (gh[j].second < 0.0) {
                gh[j].first = 0.0;
                gh[j].second = 0.0;
                //Rcpp::stop("Bad hessian");
            }

            gh[j].second /= scale;

            delta[j] = jointPrior->getDelta(gh[j], hBeta, j, *this);
            // TODO this is only correct when joints are independent across dimensions
        } else {
            delta[j] = 0.0;
        }
    }
}


double CyclicCoordinateDescent::ccdUpdateBeta(int index) {

	if (!sufficientStatisticsKnown) {
	    std::ostringstream stream;
		stream << "Error in state synchronization.";
		error->throwError(stream);
	}

	computeNumeratorForGradient(index);

	priors::GradientHessian gh;
	computeGradientAndHessian(index, &gh.first, &gh.second);

//	std::cout << "index: " << index << " grad: " << gh.first << " hess: " << gh.second << "\n";

	if (gh.second < 0.0) {
	    gh.first = 0.0;
	    gh.second = 0.0;
	}

	return jointPrior->getDelta(gh, hBeta, index, *this);
}

void CyclicCoordinateDescent::axpyXBeta(const double beta, const int j) {
    modelSpecifics.axpyXBeta(beta, j);
}

// template <class IteratorType>
// void CyclicCoordinateDescent::axpy(double* y, const double alpha, const int index) {
// 	IteratorType it(hXI, index);
// 	for (; it; ++it) {
// 		const int k = it.index();
// 		y[k] += alpha * it.value();
// 	}
// }
//
// void CyclicCoordinateDescent::axpyXBeta(const double beta, const int j) {
// 	if (beta != static_cast<double>(0.0)) {
// 		switch (hXI.getFormatType(j)) {
// 		case INDICATOR:
// 			axpy < IndicatorIterator > (hXBeta.data(), beta, j);
// 			break;
// 		case INTERCEPT:
// 		    axpy < InterceptIterator > (hXBeta.data(), beta, j);
// 		    break;
// 		case DENSE:
// 			axpy < DenseIterator > (hXBeta.data(), beta, j);
// 			break;
// 		case SPARSE:
// 			axpy < SparseIterator > (hXBeta.data(), beta, j);
// 			break;
// 		default:
// 			// throw error
// 			std::ostringstream stream;
// 			stream << "Unknown vector type.";
// 			error->throwError(stream);
// 		}
// 	}
// }

void CyclicCoordinateDescent::computeXBeta(void) {
	// Note: X is current stored in (sparse) column-major format, which is
	// inefficient for forming X\beta.
	// TODO Make row-major version of X

	if (setBetaList.empty()) { // Update all
		// clear X\beta
		modelSpecifics.zeroXBeta();
		for (int j = 0; j < J; ++j) {
			axpyXBeta(hBeta[j], j);
		}
	} else {
		while (!setBetaList.empty()) {
			SetBetaEntry entry = setBetaList.front();
			axpyXBeta(hBeta[entry.first] - entry.second, entry.first);
			setBetaList.pop_front();
		}
	}
}

void CyclicCoordinateDescent::updateXBeta(double delta, int index) {
	// Update beta
	double realDelta = static_cast<double>(delta);
	hBeta[index] += delta;

	// Delegate
	modelSpecifics.updateXBeta(realDelta, index, useCrossValidation);
}

void CyclicCoordinateDescent::updateSufficientStatistics(double delta, int index) {
	updateXBeta(delta, index);
	sufficientStatisticsKnown = true;
}

void CyclicCoordinateDescent::computeRemainingStatistics(bool allStats, int index) { // TODO Rename
	// Separate function for benchmarking
	if (allStats) {
		// Delegate
		modelSpecifics.computeRemainingStatistics(useCrossValidation);
	}
}

/**
 * Updating and convergence functions
 */

double CyclicCoordinateDescent::computeConvergenceCriterion(double newObjFxn, double oldObjFxn) {
	// This is the stopping criterion that Ken Lange generally uses
	return abs(newObjFxn - oldObjFxn) / (abs(newObjFxn) + 1.0);
}

double CyclicCoordinateDescent::applyBounds(double delta, int index) {

    auto doBound = true;
    if (doBound) {
	    if (delta < -hDelta[index]) {
		    delta = -hDelta[index];
	    } else if (delta > hDelta[index]) {
		    delta = hDelta[index];
	    }

	    // TODO Remove magic numbers
	    auto intermediate = std::max(std::abs(delta) * 2, hDelta[index] / 2);
	    intermediate = std::max(intermediate, 1E-3);
	    hDelta[index] = intermediate;
    }

	return delta;
}

/**
 * Utility functions
 */

void CyclicCoordinateDescent::computeFixedTermsInLogLikelihood(void) {
	modelSpecifics.computeFixedTermsInLogLikelihood(useCrossValidation);
}

void CyclicCoordinateDescent::computeFixedTermsInGradientAndHessian(void) {
	// Delegate
	modelSpecifics.computeFixedTermsInGradientAndHessian(useCrossValidation);
}

template <class T>
void CyclicCoordinateDescent::printVector(T* vector, int length, ostream &os) {
	os << "(" << vector[0];
	for (int i = 1; i < length; i++) {
		os << ", " << vector[i];
	}
	os << ")";
}

template <class T>
T* CyclicCoordinateDescent::readVector(
		const char *fileName,
		int *length) {

	ifstream fin(fileName);
	T d;
	std::vector<T> v;

	while (fin >> d) {
		v.push_back(d);
	}

	T* ptr = (T*) malloc(sizeof(T) * v.size());
	for (int i = 0; i < v.size(); i++) {
		ptr[i] = v[i];
	}

	*length = v.size();
	return ptr;
}

void CyclicCoordinateDescent::testDimension(int givenValue, int trueValue, const char *parameterName) {
	if (givenValue != trueValue) {
	    std::ostringstream stream;
		stream << "Wrong dimension in " << parameterName << " vector.";
		error->throwError(stream);
	}
}

inline int CyclicCoordinateDescent::sign(double x) {
	if (x == 0) {
		return 0;
	}
	if (x < 0) {
		return -1;
	}
	return 1;
}

void CyclicCoordinateDescent::getFold(int fold) {
	modelSpecifics.setFold(fold);
}

void CyclicCoordinateDescent::turnOnStreamCV(int foldToCompute) {

	streamCV = true;
	streamCVFolds = foldToCompute;
//	std::cout << "foldToCompute: " << foldToCompute << "\n";
	modelSpecifics.turnOnStreamCV(foldToCompute);

}

void CyclicCoordinateDescent::turnOnSyncCV(int foldToCompute) {

	syncCV = true;
	syncCVFolds = foldToCompute;
//	std::cout << "foldToCompute: " << foldToCompute << "\n";
	modelSpecifics.turnOnSyncCV(foldToCompute);
	for(int i=0; i<foldToCompute; ++i) {
		hBetaPool.push_back(hBeta);
		hDeltaPool.push_back(hDelta);
		hWeightsPool.push_back(hWeights);
	}

	fixBetaPool.resize(foldToCompute);
	donePool.resize(foldToCompute, false);
	for (int i=0; i<foldToCompute; i++) {
		fixBetaPool[i].resize(J, false);
	}

	modelSpecifics.setBounds(initialBound);
	std::vector<int> priorList;
	std::vector<double> varianceList = jointPrior->getVariance();
	std::vector<double> temp;
	temp.resize(J, 0.0);

	for (int i=0; i<J; i++) {
		int type = jointPrior->getPriorType(i);
		priorList.push_back(type);
		if (type == 1) {
			temp[i] = convertVarianceToHyperparameter(varianceList[0]);
		}
		if (type == 2) {
			temp[i] = varianceList[0];
		}
	}
	modelSpecifics.setPriorTypes(priorList);
	modelSpecifics.setPriorParams(temp);
	modelSpecifics.resetBeta();

}

void CyclicCoordinateDescent::turnOffSyncCV() {
	syncCV = false;
	modelSpecifics.turnOffSyncCV();
}

std::vector<double> CyclicCoordinateDescent::getPredictiveLogLikelihood(std::vector<std::vector<double>>& weightsPool) {
	xBetaKnown = false;
	if (usingGPU && syncCV) xBetaKnown = true;

	if (!xBetaKnown) {
		computeXBeta();
		xBetaKnown = true;
		sufficientStatisticsKnown = false;
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Remove index????
		sufficientStatisticsKnown = true;
	}

	getDenominators();

	std::vector<double> result;
	result.resize(syncCVFolds);
	for (int cvIndex = 0; cvIndex < syncCVFolds; cvIndex++) {
		if (donePool[cvIndex]) {
			result[cvIndex] = modelSpecifics.getPredictiveLogLikelihood(&weightsPool[cvIndex][0], cvIndex); // TODO Pass double
		} else {
			result[cvIndex] = std::numeric_limits<double>::quiet_NaN();
		}
	}

//	std::cout << "iterations: " << lastIterationCount << " ";
	return result;
}

double getPredictiveLogLikelihood(double* weights, int cvIndex) {
	return 0;
}


void CyclicCoordinateDescent::setWeights(double* iWeights, int syncCVIndex) {
	if (hWeightsPool[syncCVIndex].size() != static_cast<size_t>(K)) {
		hWeightsPool[syncCVIndex].resize(K); // = (double*) malloc(sizeof(double) * K);
	}
	for (int i = 0; i < K; ++i) {
		hWeightsPool[syncCVIndex][i] = iWeights[i];
	}
	useCrossValidation = true;
	validWeights = false;
	sufficientStatisticsKnown = false;
}

std::vector<double> CyclicCoordinateDescent::getObjectiveFunctions(int convergenceType) {
	if (convergenceType == GRADIENT) {
		//std::cout << "grad obj \n";
	    return modelSpecifics.getGradientObjectives();
	} else
	if (convergenceType == MITTAL) {
		//std::cout << "mittal obj \n";
		return getLogLikelihoods();
	} else
	if (convergenceType == LANGE) {
		//std::cout << "lange obj: ";
	    std::vector<double> logLikelihoods = getLogLikelihoods();
	    std::vector<double> logPriors = getLogPriors();
	    for (int i = 0; i < syncCVFolds; i++) {
	        logLikelihoods[i] = logLikelihoods[i] + logPriors[i];
	    }
		return logLikelihoods;
	} else {
    	std::ostringstream stream;
    	stream << "Invalid convergence type: " << convergenceType;
    	error->throwError(stream);
    }
	std::vector<double> result;
    return result;
}

std::vector<double> CyclicCoordinateDescent::getLogLikelihoods(void) {

	checkAllLazyFlags();

	getDenominators();
	likelihoodCount += 1;

	std::vector<double> result;
	return result;

	//return modelSpecifics.getLogLikelihoods(useCrossValidation);
}
std::vector<double> CyclicCoordinateDescent::getLogPriors(void) {
	std::vector<double> result;
	for (int index = 0; index < syncCVFolds; index++) {
		result.push_back(jointPrior->logDensity(hBetaPool[index], *this));
	}
	return result;
}


bool CyclicCoordinateDescent::performCheckConvergence(int convergenceType,
                                                      double epsilon,
                                                      int maxIterations,
                                                      int iteration,
                                                      std::vector<double>& lastObjFuncVec) {

    std::ostringstream stream;

    bool done = true;
    std::vector<double> conv;
    conv.resize(syncCVFolds);
    //std::vector<bool> illconditioned;
    //illconditioned.resize(syncCVFolds, false);
    bool illconditioned = true;
    //bool allIllconditioned = true;

    if (convergenceType < ZHANG_OLES) {
    	std::vector<double> thisObjFuncVec = getObjectiveFunctions(convergenceType);

//    	std::cout << "thisObjFunc " << iteration << ": ";
//        for (auto x : thisObjFuncVec) {
//            std::cout << " " << x;
//        }
//        std::cout << '\n';

    	//std::cout << "ObjFunc" << convergenceType << ": " << thisObjFunc << '\n';
    	bool needsUpdate = false;
    	for (int cvIndex = 0; cvIndex < syncCVFolds; cvIndex++) {
    		if (thisObjFuncVec[cvIndex] != thisObjFuncVec[cvIndex]) {
    			std::ostringstream stream;
    			stream << "\nWarning: problem is ill-conditioned for this choice of\n"
    					<< "\t prior (" << jointPrior->getDescription() << ") or\n"
						<< "\t initial bounding box (" << initialBound << ")\n"
						<< "Enforcing convergence!";
    			logger->writeLine(stream);
    			conv[cvIndex] = 0.0;
    			//illconditioned[cvIndex] = true;
    			//illconditioned = true;
    			//done = false;
    		} else {
    			illconditioned = false;
    			conv[cvIndex] = computeConvergenceCriterion(thisObjFuncVec[cvIndex], lastObjFuncVec[cvIndex]);
    			if (!donePool[cvIndex] && epsilon > 0 && conv[cvIndex] >= epsilon) {
    				done = false;
    			}
    			if (!donePool[cvIndex] && epsilon > 0 && conv[cvIndex] < epsilon) {
    				donePool[cvIndex] = true;
    				needsUpdate = true;
    				//if (usingGPU) modelSpecifics.updateDoneFolds(donePool);
    				/*
    				for (int j = 0; j < J; j++) {
    					fixBetaPool[cvIndex][j] = true;
    				}
    				syncCVIterator.fix(cvIndex);
    				*/
    			}
    		}
    		lastObjFuncVec[cvIndex] = thisObjFuncVec[cvIndex];
    	}
    	if (usingGPU && needsUpdate) modelSpecifics.updateDoneFolds(donePool);
    } else { // TODO ZHANG_OLES
        //conv = computeZhangOlesConvergenceCriterion();
        //saveXBeta();
    } // Necessary to call getObjFxn or computeZO before getLogLikelihood,
    // since these copy over XBeta

    if (illconditioned) {
    	done = true;
		lastReturnFlag = ILLCONDITIONED;
    } else if (done) {
    	if (noiseLevel > SILENT) {
    		stream << "Reached convergence criterion";
    	}
    	lastReturnFlag = SUCCESS;
    } else if (iteration > maxIterations) {
    	if (noiseLevel > SILENT) {
    		stream << "Reached maximum iterations";
    	}
    	done = true;
    	lastReturnFlag = MAX_ITERATIONS;
    }

    if (noiseLevel > QUIET) {
        logger->writeLine(stream);
    }

    logger->yield();

    return done;
}

} // namespace
