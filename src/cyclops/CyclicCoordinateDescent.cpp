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

#include "Rcpp.h"

#include "CyclicCoordinateDescent.h"
#include "Iterators.h"
#include "Timing.h"

#include "priors/CovariatePrior.h"
#include <tbb/parallel_for.h>


namespace bsccs {

using namespace std; // TODO Bad form

CyclicCoordinateDescent::CyclicCoordinateDescent(
			//ModelData* reader,
			const ModelData& reader,
			AbstractModelSpecifics& specifics,
			priors::JointPriorPtr prior,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error
		) : privateModelSpecifics(nullptr), modelSpecifics(specifics), jointPrior(prior),
		 hXI(reader), // hXBeta(modelSpecifics.getXBeta()), hXBetaSave(modelSpecifics.getXBetaSave()), // TODO Remove
		 logger(_logger), error(_error) {
	N = hXI.getNumberOfPatients();
	K = hXI.getNumberOfRows();
	J = hXI.getNumberOfColumns();

// 	hXI = reader;
	// hY = hXI.getYVector(); // TODO Delegate all data to ModelSpecifics
//	hOffs = reader->getOffsetVector();
	// hPid = hXI.getPidVector();

	conditionId = hXI.getConditionId();

	updateCount = 0;
	likelihoodCount = 0;
	noiseLevel = NOISY;
	initialBound = 2.0;

	init(hXI.getHasOffsetCovariate());
}

CyclicCoordinateDescent* CyclicCoordinateDescent::clone() {
	return new (std::nothrow) CyclicCoordinateDescent(*this);
}

//template <typename T>
//struct GetType<T>;

CyclicCoordinateDescent::CyclicCoordinateDescent(const CyclicCoordinateDescent& copy)
	: privateModelSpecifics(
			bsccs::unique_ptr<AbstractModelSpecifics>(
				copy.modelSpecifics.clone())), // deep copy
	  modelSpecifics(*privateModelSpecifics),
      jointPrior(copy.jointPrior), // swallow
      hXI(copy.hXI), // swallow
// 	  hXBeta(modelSpecifics.getXBeta()), hXBetaSave(modelSpecifics.getXBetaSave()), // TODO Remove
// 	  jointPrior(priors::JointPriorPtr(copy.jointPrior->clone())), // deep copy
	  logger(copy.logger), error(copy.error) {

	N = hXI.getNumberOfPatients();
	K = hXI.getNumberOfRows();
	J = hXI.getNumberOfColumns();

	// hY = hXI.getYVector(); // TODO Delegate all data to ModelSpecifics
	// hPid = hXI.getPidVector();

	conditionId = hXI.getConditionId();

	updateCount = 0;
	likelihoodCount = 0;
	noiseLevel = copy.noiseLevel;
	initialBound = copy.initialBound;

	init(hXI.getHasOffsetCovariate());

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

string CyclicCoordinateDescent::getPriorInfo() {
	return jointPrior->getDescription();
}

void CyclicCoordinateDescent::setPrior(priors::JointPriorPtr newPrior) {
    jointPrior = newPrior;
}

void CyclicCoordinateDescent::setInitialBound(double bound) {
    initialBound = bound;
}

void CyclicCoordinateDescent::resetBounds() {
	if (syncCV) {
		for (int index = 0; index < syncCVFolds; index++) {
			for (int j = 0; j < J; j++) {
				hDeltaPool[index][j] = initialBound;
			}
		}
	} else {
		for (int j = 0; j < J; j++) {
			hDelta[j] = initialBound;
		}
	}
}

void CyclicCoordinateDescent::init(bool offset) {

	// Set parameters and statistics space
	hDelta.resize(J, static_cast<double>(initialBound));
	hBeta.resize(J, static_cast<double>(0.0));

// 	hXBeta.resize(K, static_cast<double>(0.0));
// 	hXBetaSave.resize(K, static_cast<double>(0.0));

	fixBeta.resize(J, false);
	hWeights.resize(0);

	useCrossValidation = false;
	validWeights = false;
	sufficientStatisticsKnown = false;
	fisherInformationKnown = false;
	varianceKnown = false;
	if (offset) {
		hBeta[0] = static_cast<double>(1);
		fixBeta[0] = true;
		xBetaKnown = false;
	} else {
		xBetaKnown = true; // all beta = 0 => xBeta = 0
	}
	doLogisticRegression = false;

	modelSpecifics.initialize(N, K, J, &hXI, NULL, NULL, NULL,
			NULL, NULL,
			NULL, // hPid,
			NULL,
			NULL, NULL,
			NULL,
			NULL
		//	hY
			);
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
						useCrossValidation);
	}
}

void CyclicCoordinateDescent::resetBeta(void) {
	auto start = hXI.getHasOffsetCovariate() ? 1 : 0;
	if (syncCV) {
		for (int i=0; i<syncCVFolds; ++i) {
			if (!donePool[i]) {
				for (auto j = start; j < J; j++) {
					hBetaPool[i][j] = 0.0;
				}
			}
		}
	} else {
		for (auto j = start; j < J; j++) {
			hBeta[j] = 0.0;
		}
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
		outLog << hXI.getColumn(i).getLabel()
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

double CyclicCoordinateDescent::getPredictiveLogLikelihood(double* weights) {

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

double CyclicCoordinateDescent::getLogPrior(void) {
	return jointPrior->logDensity(hBeta);
}

double CyclicCoordinateDescent::getObjectiveFunction(int convergenceType) {
	if (convergenceType == GRADIENT) {
		//std::cout << "grad obj\n";
	    return modelSpecifics.getGradientObjective(useCrossValidation);
	} else
	if (convergenceType == MITTAL) {
		//std::cout << "mittal obj\n";
		return getLogLikelihood();
	} else
	if (convergenceType == LANGE) {
		//std::cout << "lange obj\n";
    	//std::cout << getLogLikelihood() << " " << getLogPrior();
		return getLogLikelihood() + getLogPrior();
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
	const auto algorithmType = arguments.algorithmType;
	const int qnQ = 0;
	const auto mmepsilon = arguments.mmtolerance;
	modelSpecifics.setAlgorithmType(algorithmType);

	initialBound = arguments.initialBound;

	if (syncCV) {
		fixBetaPool.resize(syncCVFolds);
		for (int i=0; i<syncCVFolds; i++) {
			fixBetaPool[i].resize(J);
			for (int j=0; j<J; j++) {
				fixBetaPool[i][j] = false;
			}
		}
		donePool.resize(syncCVFolds);
		for (int i=0; i<syncCVFolds; i++) {
			donePool[i] = false;
		}
		nonZeros.resize(syncCVFolds);
		for (int i=0; i<syncCVFolds; i++) {
			nonZeros[i] = J;
		}
		syncCVIterator.reset(syncCVFolds, J);
	}

	for (int i=0; i<J; i++) {
		fixBeta[i] = false;
	}

	int count = 0;
	bool done = false;
	while (!done) {
 	    if (arguments.useKktSwindle && jointPrior->getSupportsKktSwindle()) {
		    kktSwindle(arguments);
	    } else {
		    findMode(maxIterations, convergenceType, epsilon, algorithmType, qnQ, mmepsilon);
	    }
	    ++count;

	    if (lastReturnFlag == ILLCONDITIONED && count < maxCount) {
	        // Reset beta and shrink bounding box
	        initialBound /= 10.0;
            resetBeta();
            resetFixBeta();
	    } else {
	        done = true;
	    }
	}
}

void CyclicCoordinateDescent::resetFixBeta() {
	for (int i=0; i<J; i++) {
		fixBeta[i] = false;
	}
	if (syncCV) {
		syncCVIterator.reset(syncCVFolds, J);
		std::cout << "J: " << J << "\n";
		for (int i=0; i<syncCVFolds; i++) {
			for (int j=0; j<J; j++) {
				fixBetaPool[i][j] = false;
			}
			nonZeros[i] = J;
			donePool[i] = false;
		}
	}
}

typedef std::tuple<
	int,    // index
	double, // gradient
	bool    // force active
> ScoreTuple;

//called by kkt swindle
template <typename Iterator>
void CyclicCoordinateDescent::findMode(Iterator begin, Iterator end,
		const int maxIterations, const int convergenceType, const double epsilon,
		const AlgorithmType algorithmType, const int qnQ, const double mmepsilon) {

	std::fill(fixBeta.begin(), fixBeta.end(), true);
 	std::for_each(begin, end, [this] (ScoreTuple& tuple) {
 		fixBeta[std::get<0>(tuple)] = false;
 	});
	findMode(maxIterations, convergenceType, epsilon, algorithmType, qnQ, mmepsilon);
    // fixBeta is no longer valid
}

void CyclicCoordinateDescent::kktSwindle(const ModeFindingArguments& arguments) {

	const auto maxIterations = arguments.maxIterations;
	const auto convergenceType = arguments.convergenceType;
	const auto epsilon = arguments.tolerance;
	const auto algorithmType = arguments.algorithmType;
	const auto mmepsilon = arguments.mmtolerance;
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
	int perPassSize = arguments.swindleMultipler;

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
            algorithmType, qnQ, mmepsilon);

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
		perPassSize *= 2;

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
    //if (iteration==1) std::cout << "sizeof real:  " << sizeof(real) << "\n";
	//std::cout << "" << *lastObjFunc << '\n';

    bool done = false;
    double conv;
    bool illconditioned = false;
    if (convergenceType < ZHANG_OLES) {
        double thisObjFunc = getObjectiveFunction(convergenceType);
        //std::cout << "ObjFunc" << convergenceType << ": " << thisObjFunc << '\n';
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
        }
        *lastObjFunc = thisObjFunc;
    } else { // ZHANG_OLES
        conv = computeZhangOlesConvergenceCriterion();
        saveXBeta();
    } // Necessary to call getObjFxn or computeZO before getLogLikelihood,
    // since these copy over XBeta

    //double thisLogLikelihood = getLogLikelihood();
    //double thisLogPrior = getLogPrior();
    //double thisLogPost = thisLogLikelihood + thisLogPrior;
    //std::cout << setprecision(15) << "hBeta: " << hBeta[0] << " | " << hBeta[1] << '\n';
    //std::cout << "logs: " << thisLogLikelihood << " | " << thisLogPrior << '\n';

    std::ostringstream stream;
    if (noiseLevel > QUIET) {
        // stream << "\n";
        // printVector(&hBeta[0], J, stream);
        stream << "\n";
        //stream << "log post: " << thisLogPost
        //       << " (" << thisLogLikelihood << " + " << thisLogPrior
         //      << ") (iter:" << iteration << ", conv: " << conv << ") ";
    }

    //std::cout << conv << "\n";

    if (epsilon > 0 && conv < epsilon) {
        if (illconditioned) {
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
		int qnQ,
		double mmepsilon
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

	/*
	if (initialBound == 2.0) {
		modelSpecifics.printStuff();
	}
	*/

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
	std::vector<double> allDeltaObj;

    double lastLogPosterior; // = getLogLikelihood() + getLogPrior();

	std::vector<std::vector<double>> allDeltaPool;
    std::vector<double> lastLogPosteriorVec;

	if (mm_original) {
	   algorithmType = AlgorithmType::MM;
	   modelSpecifics.setAlgorithmType(AlgorithmType::MM);
	   std::cout << "switch to mm \n";
	}

	if (algorithmType == AlgorithmType::MM) {
	    // Any further initialization necessary?
		if (syncCV) {
			allDeltaPool.resize(syncCVFolds);
			lastLogPosteriorVec.resize(syncCVFolds);
			for (int index = 0; index < syncCVFolds; index++) {
				if (allDeltaPool[index].size() < J) allDeltaPool[index].resize(J);
				lastLogPosteriorVec[index] = -10E10;
			}
		} else {
			if (allDelta.size() < J) allDelta.resize(J);
			lastLogPosterior = -10E10;
		}
	}

	if (algorithmType == AlgorithmType::CCDGREEDY) {
		if (allDelta.size() < J) allDelta.resize(J);
		if (allDeltaObj.size() < J) allDeltaObj.resize(J);
	}

	auto cycle = [this,&iteration,&algorithmType,&allDelta,&allDeltaPool,&lastObjFunc,&convergenceType] {

		if (iteration%10==0 && iteration > 0) {
			std::cout<<"iteration " << iteration << " ";

			if (syncCV) {
				std::cout << "nonzeros: ";
				for (int i=0; i<syncCVFolds; i++) {
					if (!donePool[i]) {
						std::cout << nonZeros[i] << " ";
					}
				}
			}
			std::cout << "\n";
		}

	    auto log = [this](const int index) {
	        if ( (noiseLevel > QUIET) && ((index+1) % 100 == 0)) {
	            std::ostringstream stream;
	            stream << "Finished variable " << (index+1);
	            logger->writeLine(stream);
	        }
	    };

	    if (algorithmType == AlgorithmType::MM) {
            // Do delta computation in parallel
	    	if (syncCV) {
	    		std::vector<std::pair<int,int>> indicesToUpdate = syncCVIterator.getAllIndicesMM();
	    		//std::cout << "indicesToUpdate length: " << indicesToUpdate.size() << "\n";
	    		allDelta.resize(indicesToUpdate.size());
	    		mmUpdateAllBeta(allDelta, indicesToUpdate);
	    		applyBounds(allDelta, indicesToUpdate);
	    		updateSufficientStatistics(allDelta, indicesToUpdate);
	            computeRemainingStatistics();

	    	} else {
	            mmUpdateAllBeta(allDelta);
	            updateSufficientStatistics(allDelta);
	            computeRemainingStatistics();
	    	}

	        //Rcpp::stop("A");
            //std::cout << setprecision (15) << "hBeta: " << hBeta[0] << " | " << hBeta[1] << " delta: " << allDelta[0] << " | " << allDelta[1] << '\n';



/*
            for (int index = 0; index < J; ++index) {
                if (!fixBeta[index]) {
                    double delta = allDelta[index];
                   // delta = applyBounds(delta, index);
                    if (delta != 0.0) {
                        sufficientStatisticsKnown = false;
                        //hBeta[index] += delta;
	                    updateSufficientStatistics(delta, index);
                        // std::cerr << " : " << index << " " << hBeta[index] << " " << delta;

                    // modelSpecifics.axpyXBeta(delta, index); // TODO Do single spMV
                    }
                }
            }
            */

            // sufficientStatisticsKnown = false;
            //computeRemainingStatistics(true, 0);
            //sufficientStatisticsKnown = true;

	    } else if (algorithmType == AlgorithmType::CCDGREEDY){
            ccdUpdateAllBeta(allDelta);
            /*
            std::cout << "allDelta: ";
            for (auto x:allDelta) {
            	std::cout << x << " ";
            }
            std::cout << "\n";
            */

            int maxIndex = 0;
            int maxDelta = 0;
            for (int j=0; j<J; ++j) {
            	if (allDelta[j] > maxDelta) {
            		maxDelta = allDelta[j];
            		maxIndex = j;
            	}
            }

            double delta = ccdUpdateBeta(maxIndex);
            //std::cout << "delta " << index << ": " << delta;
            //std::cerr << " " << delta;
            delta = applyBounds(delta, maxIndex);
            //std::cout << " " << delta;
            //std::cout << "delta" << index << ": " << delta << "\n";
            if (delta != 0.0) {
            	sufficientStatisticsKnown = false;
            	updateSufficientStatistics(delta, maxIndex);
            	computeRemainingStatistics();
            }

	    } else {

	    	if (syncCV) {
	    		std::vector<std::vector<std::pair<int,int>>> indicesToUpdate = syncCVIterator.getAllIndicesCCD();
	    		//std::cout << "indices updated: " << indicesToUpdate.size() << "\n";
	    		for (int i=0; i<indicesToUpdate.size(); i++) {
	    			//std::cout << "update " << i << " : " << indicesToUpdate[i].size() << "\n";
		    		allDelta.resize(indicesToUpdate[i].size());
	    			ccdUpdateBetaVec(allDelta, indicesToUpdate[i]);
/*
	    			std::cout << "allDelta" << i << " ";
	    			for (auto x:allDelta) {
	    				std::cout << x << " ";
	    			}
	    			std::cout << "\n";
*/

		    		applyBounds(allDelta, indicesToUpdate[i]);
		    		updateSufficientStatistics(allDelta, indicesToUpdate[i]);
		            computeRemainingStatistics(true, allDelta, indicesToUpdate[i]);
	    		}
	    	}



	    	//std::cout<<hBeta[0]<<"\n";
	        // Do a complete cycle in serial
	    	//std::cout << "starting cycle\n";
	        for(int index = 0; index < J; index++) {
	        	//std::cout << "index " << index << ": ";
		    	//std::cout << "hBeta[0]: " << hBeta[0] << " hBeta[1]: " << hBeta[1] << '\n';
	        	if (syncCV) {
/*
	        		std::vector<double> deltaVec = ccdUpdateBetaVec(index);

	        		std::cout << "allDelta: ";
	        		for (auto x:deltaVec) {
	        			std::cout << x << " ";
	        		}
	        		std::cout << "\n";

	        		deltaVec = applyBounds(deltaVec, index);
	        		updateSufficientStatistics(deltaVec, index);
	        		computeRemainingStatistics(true, deltaVec);
*/
	        	} else {
	        		if (!fixBeta[index]) {
	        			double delta = ccdUpdateBeta(index);
	        			//std::cout << "delta " << index << ": " << delta;
	        			//std::cerr << " " << delta;
	        			delta = applyBounds(delta, index);
	        			//std::cout << " " << delta;
	        			//std::cout << "delta" << index << ": " << delta << "\n";
	        			if (delta != 0.0) {
	        				sufficientStatisticsKnown = false;
	        				updateSufficientStatistics(delta, index);
	        				computeRemainingStatistics();
	        			}
	        		}

	            //log(index);
	        	}
		        //std::cout << "\n";
	        }

	        //std::cout << '\n';
            //std::cout << setprecision (15) << "hBeta: " << hBeta[0] << " | " << hBeta[1] << '\n';

	    }
	    iteration++;
	};

	auto check = [this,&iteration,&lastObjFunc,&algorithmType,&lastLogPosterior, &lastObjFuncVec, &lastLogPosteriorVec,
               convergenceType, epsilon,maxIterations,mmepsilon] {
                   bool done = false;
                   //		bool checkConvergence = (iteration % J == 0 || iteration == maxIterations);
                   bool checkConvergence = true; // Check after each complete cycle

                   //std::cout << "log likelihood: " << getLogLikelihood() << " log prior: " << getLogPrior() << " \n";
                   //double thisLogPosterior = getLogLikelihood() + getLogPrior();
                   //std::cout << thisLogPosterior << "\n";

                   if (algorithmType == AlgorithmType::MM) {
                       //std::cout << thisLogPosterior << "\n";
                       //std::cout << "log likelihood: " << getLogLikelihood() << " log prior: " << getLogPrior() << " total: ";

                	   if (syncCV) {
                		   /*
                		   int count = 0;
                		   std::vector<double> thisObjFuncVec = getObjectiveFunctions(convergenceType);
                		   for (int i=0; i<syncCVFolds; i++) {
                               double conv = computeConvergenceCriterion(thisObjFuncVec[i], lastObjFuncVec[i]);
                               if (conv < mmepsilon) {
                            	   count++;
                               }
                		   }
                		   if (count > syncCVFolds*2/3) {
                			   mm_original = true;
                		   }
                		   */
                	   } else {
                       if (iteration > 1) {
                    	   /*
                           double change = thisLogPosterior - lastLogPosterior;
                           if (abs(change) < 0.01) {
                        	   algorithmType = AlgorithmType::CCD;
                        	   modelSpecifics.setAlgorithmType(AlgorithmType::CCD);
                           }
                           */
                           //std::cout << setprecision (15) << lastLogPosterior << " -> " << thisLogPosterior << " == " << change << '\n';
                           //Rcpp::Rcout << lastLogPosterior << " -> " << thisLogPosterior << " == " << change << '\n';
                           //std::cout << '\n';


                           /*
                           if (change < 0.0) {
                               std::ostringstream stream;
                               stream << "Non-increasing!";
                               error->throwError(stream);
                           }
                           */
                           double thisObjFunc = getObjectiveFunction(convergenceType);
                           double conv = computeConvergenceCriterion(thisObjFunc, lastObjFunc);
                           if (conv < mmepsilon) {
                        	   mm_original = true;
                        	   algorithmType = AlgorithmType::CCDGREEDY;
                        	   modelSpecifics.setAlgorithmType(AlgorithmType::CCDGREEDY);
                        	   std::cout << "switch to greedy ccd \n";
                           }
                       }
                	   }

                   }
                   //lastLogPosterior = thisLogPosterior;


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
	    using namespace Eigen;

	    Eigen::MatrixXd secantsU(J, qnQ);
	    Eigen::MatrixXd secantsV(J, qnQ);

	    VectorXd x(J);

	    // Fill initial secants
	    int countU = 0;
	    int countV = 0;

	    for (int q = 0; q < qnQ; ++q) {
	        x = Map<const VectorXd>(hBeta.data(), J); // Make copy

	        cycle();
	        done = check();
	        if (done) break;

	        if (countU == 0) { // First time through
	            secantsU.col(countU) = Map<const VectorXd>(hBeta.data(), J) - x;
	            ++countU;
	        } else if (countU < qnQ - 1) { // Middle case
	            secantsU.col(countU) = Map<const VectorXd>(hBeta.data(), J) - x;
	            secantsV.col(countV) = secantsU.col(countU);
	            ++countU;
	            ++countV;
	        } else { // Last time through
	            secantsV.col(countV) = Map<const VectorXd>(hBeta.data(), J) - x;
	            ++countV;
	        }
	    }

	    // 	    std::cerr << secantsU << std::endl << std::endl;
	    // 	    std::cerr << secantsV << std::endl;

	    int newestSecant = qnQ - 1;
	    int previousSecant = newestSecant - 1;

	    while (!done) {

	        // 2 cycles for each QN step
	        x = Map<const VectorXd>(hBeta.data(), J); // Make copy
	        cycle();
	        // 	        done = check();
	        // 	        if (done) {
	        // 	            std::cerr << "break A" << std::endl;
	        // 	            break;
	        // 	        }

	        secantsU.col(newestSecant) = Map<const VectorXd>(hBeta.data(), J) - x;

	        VectorXd Fx = Map<const VectorXd>(hBeta.data(), J); // TODO Can remove?

	        x = Map<const VectorXd>(hBeta.data(), J);
	        cycle();
	        // 	        done = check();
	        // 	        if (done) {
	        // 	            std::cerr << "break B" << std::endl;
	        // 	            break;
	        // 	        }

	        secantsV.col(newestSecant) = Map<const VectorXd>(hBeta.data(), J) - x;

	        // Do QN step here

	        //             MatrixXd UtU = secantsU.transpose() * secantsU;
	        //             MatrixXd UtV = secantsU.transpose() * secantsV;
	        //             MatrixXd M = UtU - UtV;

	        auto M = secantsU.transpose() * (secantsU - secantsV);

	        auto Minv = M.inverse();

	        auto A = secantsU.transpose() * secantsU.col(newestSecant);
	        auto B = Minv * A;
	        auto C = secantsV * B;

	        // MatrixXd A = secantsV * Minv;
	        // MatrixXd B = A * secantsU.transpose();
	        // MatrixXd C = B * secantsV.col(newestSecant); // - (x - Fx)
	        // MatrixXd C = B * secantsU.col(newestSecant); // TODO Can remove?

	        //     VectorXd xqn = Map<const VectorXd>(hBeta.data(), J) + C;
	        VectorXd xqn = Fx + C; // TODO Can remove?

	        // Save CCD solution
	        x = Map<const VectorXd>(hBeta.data(), J);
	        double ccdObjective = getLogLikelihood() + getLogPrior();
	        // double savedLastObjFunc = lastObjFunc;

	        Map<VectorXd>(hBeta.data(), J) = xqn; // Set QN solution
	        //             for (int j = 0; j < J; ++j) {
	        //                 if (sign(hBeta[j]) == xqn(j)) {
	        //                     hBeta[j] = xqn(j);
	        //                 } else {
	        //                     hBeta[j] = 0.0;
	        //                 }
	        //             }

	        // xBetaKnown = false;
	        // checkAllLazyFlags();
	        modelSpecifics.computeXBeta(hBeta.data(), useCrossValidation);


	        //             done = check();
	        //             if (done) {
	        //                 std::cerr << "break B" << std::endl;
	        //                 break;
	        //             }

	        double qnObjective = getLogLikelihood() + getLogPrior();

	        if (ccdObjective > qnObjective) { // Revert
	            Map<VectorXd>(hBeta.data(), J) = x; // Set CCD solution
	            modelSpecifics.computeXBeta(hBeta.data(), useCrossValidation);
	            // xBetaKnown = false;
	            // checkAllLazyFlags();

	            double ccd2Objective = getLogLikelihood() + getLogPrior();

	            if (ccdObjective != ccd2Objective) {
	                std::cerr << "Poor revert: " << ccdObjective << " != " << ccd2Objective << " diff: "<< (ccdObjective - ccd2Objective) << std::endl;
	            }
	            // lastObjFunc = savedLastObjFunc;
	            std::cerr << "revert" << std::endl;
	        } else {
	            std::cerr << "accept" << std::endl;
	            //                 done = check();
	            //                 if (done) {
	            //                     std::cerr << "break C" << std::endl;
	            //                     break;
	            //                 }
	        }

	        done = check();

	        // lastObjFunc = getLogLikelihood() + getLogPrior();

	        // Get ready for next secant-pair
	        previousSecant = newestSecant;
	        newestSecant = (newestSecant + 1) % qnQ;
	    }
    } else { // No QN
        while (!done) {
            cycle();
            done = check();
        }
    }

	lastIterationCount = iteration;
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
	if (syncCV) {
		int count = 0;
		std::vector<int> temp;
		for (int i=0; i<syncCVFolds; ++i) {
			if (!fixBetaPool[i][index]) {
				++count;
				temp.push_back(i);
			}
		}

		auto func = [&](const tbb::blocked_range<int>& range) {
			for (int k = range.begin(); k < range.end(); ++k) {
				modelSpecifics.computeNumeratorForGradient(useCrossValidation, temp[k]);
			}
		};
		tbb::parallel_for(tbb::blocked_range<int>(0,count),func);

	} else {
		modelSpecifics.computeNumeratorForGradient(index);
	}
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

double CyclicCoordinateDescent::getAsymptoticPrecision(int indexOne, int indexTwo) {
	checkAllLazyFlags();
	if (!fisherInformationKnown) {
		computeAsymptoticPrecisionMatrix();
		fisherInformationKnown = true;
	}

	IndexMap::iterator itOne = hessianIndexMap.find(indexOne);
	IndexMap::iterator itTwo = hessianIndexMap.find(indexTwo);

	if (itOne == hessianIndexMap.end() || itTwo == hessianIndexMap.end()) {
		return NAN;
	} else {
		return hessianMatrix(itOne->second, itTwo->second);
	}
}

CyclicCoordinateDescent::Matrix CyclicCoordinateDescent::computeFisherInformation(const std::vector<size_t>& indices) const {
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

//			std::cerr << "Info = " << fisherInformation << std::endl;
//			std::cerr << "Hess = " << getHessianDiagonal(0) << std::endl;
//			exit(-1);
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

void CyclicCoordinateDescent::mmUpdateAllBeta(std::vector<double>& delta) {

    if (!sufficientStatisticsKnown) {
        std::ostringstream stream;
        stream << "Error in state synchronization.";
        error->throwError(stream);
    }

    std::vector<priors::GradientHessian> gh(J); // TODO make once

    // computeNumeratorForGradient(index);

    // priors::GradientHessian gh;
    // computeGradientAndHessian(index, &gh.first, &gh.second);

    modelSpecifics.computeMMGradientAndHessian(gh, fixBeta, useCrossValidation);

    double scale = 1.0; // TODO Tune, somehow?

    for (int j = 0; j < J; ++j) {
        if (!fixBeta[j]) {
            if (gh[j].second < 0.0) {
                gh[j].first = 0.0;
                gh[j].second = 0.0;
                Rcpp::stop("Bad hessian");
            }

            gh[j].second /= scale;

            delta[j] = jointPrior->getDelta(gh[j], hBeta, j);
            //std::cout << "j: " << j << " | " << gh[j].first << " | " << gh[j].second << " | " << delta[j] << '\n';
            // TODO this is only correct when joints are independent across dimensions
        } else {
            delta[j] = 0.0;
        }
    }

}

void CyclicCoordinateDescent::ccdUpdateAllBeta(std::vector<double>& delta) {
	 if (!sufficientStatisticsKnown) {
	        std::ostringstream stream;
	        stream << "Error in state synchronization.";
	        error->throwError(stream);
	    }

	    std::vector<priors::GradientHessian> gh(J); // TODO make once

	    // computeNumeratorForGradient(index);

	    // priors::GradientHessian gh;
	    // computeGradientAndHessian(index, &gh.first, &gh.second);

	    modelSpecifics.computeAllGradientAndHessian(gh, fixBeta, useCrossValidation);
	    for (int j=0; j<J; ++j) {
	    	if (fixBeta[j]) {
	    		delta[j] = 0.0;
	    	} else {
	    		gh[j].second = 1.0;
	    		delta[j] = abs(jointPrior->getDelta(gh[j],hBeta,j));
	    	}
	    }
	    /*
	    double scale = 1.0; // TODO Tune, somehow?

	    for (int j = 0; j < J; ++j) {
	        if (!fixBeta[j]) {
	            if (gh[j].second < 0.0) {
	                gh[j].first = 0.0;
	                gh[j].second = 0.0;
	                Rcpp::stop("Bad hessian");
	            }

	            gh[j].second /= scale;

	            delta[j] = jointPrior->getDelta(gh[j], hBeta, j);
	            //std::cout << "j: " << j << " | " << gh[j].first << " | " << gh[j].second << " | " << delta[j] << '\n';
	            // TODO this is only correct when joints are independent across dimensions
	        } else {
	            delta[j] = 0.0;
	        }
	    }
	    */
}

void CyclicCoordinateDescent::mmUpdateAllBeta(std::vector<double>& delta, std::vector<std::pair<int,int>>& indicesToUpdate) {

    if (!sufficientStatisticsKnown) {
        std::ostringstream stream;
        stream << "Error in state synchronization.";
        error->throwError(stream);
    }

    int length = indicesToUpdate.size();
    //std::vector<std::vector<int>> indicesToUpdate = syncCVIterator.getAllIndicesMM();
    if (length==0) {
    	return;
    }
    //delta.resize(length);
    std::vector<priors::GradientHessian> gh(length); // TODO make once

    modelSpecifics.computeMMGradientAndHessian(gh, indicesToUpdate);

    //std::cout << "length: " << length << " delta: " << delta.size() << " gh: " << gh.size() << " indices: " << indicesToUpdate.size() << "\n";
    double scale = 1.0;
    for (int i=0; i<length; i++) {
    	if (gh[i].second < 0.0) {
    		gh[i].first = 0.0;
    		gh[i].second = 0.0;
    		Rcpp::stop("Bad hessian");
    	}
    	gh[i].second /= scale;
    	delta[i] = jointPrior->getDelta(gh[i],hBetaPool[indicesToUpdate[i].second],indicesToUpdate[i].first);
    }
    /*
    auto func = [&](const tbb::blocked_range<int>& range) {
    	for (int k = range.begin(); k < range.end(); ++k) {
    	    //modelSpecifics.computeMMGradientAndHessian(ghPool[temp[k]], fixBetaPool[temp[k]], useCrossValidation, temp[k]);
    	}
    };
    tbb::parallel_for(tbb::blocked_range<int>(0,count),func);
    */
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
	//std::cout << "grad" << ": " << gh.first << " hess" << ": " << gh.second << " ";

	if (gh.second < 0.0) {
	    gh.first = 0.0;
	    gh.second = 0.0;
	}

	double blah = jointPrior->getDelta(gh, hBeta, index);
	//std::cout << " delta: " << blah;

	return blah;

	//return jointPrior->getDelta(gh, hBeta, index);
}

void CyclicCoordinateDescent::axpyXBeta(const double beta, const int j) {
    modelSpecifics.axpyXBeta(beta, j);
}

void CyclicCoordinateDescent::axpyXBeta(const double beta, const int j, const int index) {
    modelSpecifics.axpyXBeta(beta, j, index);
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
		if (syncCV) {
			for (int i = 0; i < syncCVFolds; ++i) {
				for (int j = 0; j < J; ++j) {
					axpyXBeta(hBetaPool[i][j], j, i);
				}
			}
		} else {
			for (int j = 0; j < J; ++j) {
				axpyXBeta(hBeta[j], j);
			}
		}
	} else { //TODO figure out for syncCV
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
	if (hBeta[index] == 0.0 && fixBeta[index] == false) {
		fixBeta[index] = true;
	}
	sufficientStatisticsKnown = true;
}

void CyclicCoordinateDescent::computeRemainingStatistics(bool allStats, std::vector<double>& deltaVec) { // TODO Rename
	// Separate function for benchmarking
	if (allStats) {
		// Delegate
		/*
		for (int i=0; i<syncCVFolds; ++i) {
			if (!donePool[i] || !fixBetaPool[index][i]) {
				if (deltaVec[i] != 0.0) {
					modelSpecifics.computeRemainingStatistics(useCrossValidation, i);
				}
			}
		}
		*/
		std::vector<bool> fixBetaTemp;
		for (int i=0; i<syncCVFolds; i++) {
			if (donePool[i]==false && deltaVec[i]!=0.0) {
				fixBetaTemp.push_back(false);
			} else {
				fixBetaTemp.push_back(true);
			}
		}
		modelSpecifics.computeRemainingStatistics(useCrossValidation, fixBetaTemp);
		/*
		int count = 0;
		std::vector<int> temp;

		for (int i=0; i<syncCVFolds; ++i) {
			if (!donePool[i] || !fixBetaPool[i][index]) {
				if (deltaVec[i] != 0.0) {
					++count;
					temp.push_back(i);
				}
			}
		}

		auto func = [&](const tbb::blocked_range<int>& range) {
			for (int k = range.begin(); k < range.end(); ++k) {
				modelSpecifics.computeRemainingStatistics(useCrossValidation, temp[k]);
			}
		};
		tbb::parallel_for(tbb::blocked_range<int>(0,count),func);
		*/

	}
}

void CyclicCoordinateDescent::computeRemainingStatistics(bool allStats, std::vector<double>& deltaVec, std::vector<std::pair<int,int>> updateIndices) { // TODO Rename
	// Separate function for benchmarking
	if (allStats) {
		std::vector<bool> fixBetaTemp;
		fixBetaTemp.resize(syncCVFolds, true);
		for (int i=0; i<updateIndices.size(); i++) {
			int cvIndex = updateIndices[i].second;
			if (donePool[cvIndex]==false && deltaVec[i]!=0.0) {
				fixBetaTemp[cvIndex] = false;
			}
		}
		modelSpecifics.computeRemainingStatistics(useCrossValidation, fixBetaTemp);
	}
}



void CyclicCoordinateDescent::computeRemainingStatistics(bool allStats, int index) { // TODO Rename
	// Separate function for benchmarking
	if (allStats) {
		if (syncCV) {
			std::vector<bool> fixBetaTemp;
			for (int i=0; i<syncCVFolds; i++) {
				if (donePool[i] || fixBetaPool[i][index]) {
					fixBetaTemp.push_back(true);
				} else {
					fixBetaTemp.push_back(false);
				}
			}
			modelSpecifics.computeRemainingStatistics(useCrossValidation, fixBetaTemp);

		}
		modelSpecifics.computeRemainingStatistics(useCrossValidation);
	}
}

void CyclicCoordinateDescent::computeRemainingStatistics() { // TODO Rename
	// Separate function for benchmarking
	if (syncCV) {
		std::vector<bool> fixBetaTemp;
		for (int i=0; i<syncCVFolds; i++) {
			if (donePool[i]) {
				fixBetaTemp.push_back(true);
			} else {
				fixBetaTemp.push_back(false);
			}
		}
		modelSpecifics.computeRemainingStatistics(useCrossValidation, fixBetaTemp);

	}
	modelSpecifics.computeRemainingStatistics(useCrossValidation);
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

void CyclicCoordinateDescent::turnOnSyncCV(int foldToCompute) {
	syncCV = true;
	syncCVFolds = foldToCompute;
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
}

void CyclicCoordinateDescent::turnOffSyncCV() {
	syncCV = false;
	modelSpecifics.turnOffSyncCV();
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

	return modelSpecifics.getLogLikelihoods(useCrossValidation);
}

std::vector<double> CyclicCoordinateDescent::getLogPriors(void) {
	std::vector<double> result;
	for (int index = 0; index < syncCVFolds; index++) {
		result.push_back(jointPrior->logDensity(hBetaPool[index]));
	}
	return result;
}

std::vector<double> CyclicCoordinateDescent::ccdUpdateBetaVec(int index) {

	if (!sufficientStatisticsKnown) {
	    std::ostringstream stream;
		stream << "Error in state synchronization.";
		error->throwError(stream);
	}

	computeNumeratorForGradient(index);

	priors::GradientHessian gh;
	std::vector<priors::GradientHessian> ghList;
	std::vector<bool> fixBetaTemp;

	for (int cvIndex = 0; cvIndex < syncCVFolds; ++cvIndex) {
		ghList.push_back(gh);
		fixBetaTemp.push_back(fixBetaPool[cvIndex][index]);
	}

	if (fixBetaTemp.size() > 0) modelSpecifics.computeGradientAndHessian(index, ghList, fixBetaTemp, useCrossValidation);

	/*
	int count = 0;
	std::vector<int> temp;

	for (int cvIndex = 0; cvIndex < syncCVFolds; ++cvIndex) {
		ghList.push_back(gh);
		if (!fixBetaPool[cvIndex][index]) {
			++count;
			temp.push_back(cvIndex);
		}
	}

	auto func = [&](const tbb::blocked_range<int>& range) {
		for (int k = range.begin(); k < range.end(); ++k) {
			modelSpecifics.computeGradientAndHessian(index, &ghList[temp[k]].first, &ghList[temp[k]].second, useCrossValidation, temp[k]);
		}
	};
	tbb::parallel_for(tbb::blocked_range<int>(0,count),func);
	*/

	/*
	std::vector<double> gradList;
	gradList.resize(syncCVFolds);
	std::vector<double> hessList;
	hessList.resize(syncCVFolds);
	*/




	//computeGradientAndHessian(index, gradList, hessList);

	/*
	for (int cvIndex = 0; cvIndex < syncCVFolds; cvIndex++) {
		std::cout << cvIndex << ": " << gradList[cvIndex] << " | " << hessList[cvIndex] << " ";
	}
	std::cout << "\n";
	*/

	//computeGradientAndHessian(index, &gh.first, &gh.second);
	//std::cout << "grad" << index << ": " << gh.first << " hess" << index << ": " << gh.second << " ";

	std::vector<double> result;
	for (int cvIndex = 0; cvIndex < syncCVFolds; cvIndex++) {
		if (donePool[cvIndex] || fixBetaPool[cvIndex][index]) {
			result.push_back(0.0);
		} else {
			if (ghList[cvIndex].second < 0.0) {
				ghList[cvIndex].first = 0.0;
				ghList[cvIndex].second = 0.0;
			}
			//gh = priors::GradientHessian(gradList[cvIndex], hessList[cvIndex]);
			gh = ghList[cvIndex];
			//std::cout << "gh " << cvIndex <<  ": " << gh.first << " | " << gh.second << " ";
			double blah = jointPrior->getDelta(gh, hBetaPool[cvIndex], index);
			//std::cout << "delta: " << blah << "\n";
			result.push_back(blah);
			//std::cout << "\n       ";
		}
	}
/*
	std::cout << "h" << index << ": ";
	for (int i=0; i<syncCVFolds; i++) {
		std::cout << hBetaPool[i][index] << " ";
	}
	std::cout << "\n";

	std::cout << "gh: ";
	for (int i=0; i<syncCVFolds; i++) {
		std::cout << ghList[i].first << " " << ghList[i].second << " ";
	}
	std::cout << "\n";
	*/

	return result;
}


void CyclicCoordinateDescent::ccdUpdateBetaVec(std::vector<double>& deltaVec, std::vector<std::pair<int,int>>& indicesToUpdate) {

	if (!sufficientStatisticsKnown) {
	    std::ostringstream stream;
		stream << "Error in state synchronization.";
		error->throwError(stream);
	}

    int length = indicesToUpdate.size();
    //std::vector<std::vector<int>> indicesToUpdate = syncCVIterator.getAllIndicesMM();
    if (length==0) {
    	return;
    }
    //delta.resize(length);
    std::vector<priors::GradientHessian> gh(length); // TODO make once

    modelSpecifics.computeGradientAndHessian(gh, indicesToUpdate);

    //std::cout << "length: " << length << " delta: " << delta.size() << " gh: " << gh.size() << " indices: " << indicesToUpdate.size() << "\n";
    double scale = 1.0;
    for (int i=0; i<length; i++) {
    	if (gh[i].second < 0.0) {
    		gh[i].first = 0.0;
    		gh[i].second = 0.0;
    		Rcpp::stop("Bad hessian");
    	}
    	gh[i].second /= scale;
    	deltaVec[i] = jointPrior->getDelta(gh[i],hBetaPool[indicesToUpdate[i].second],indicesToUpdate[i].first);
    }
}

void CyclicCoordinateDescent::computeGradientAndHessian(int index, double *ogradient, double *ohessian, int cvIndex) {
		modelSpecifics.computeGradientAndHessian(index, ogradient, ohessian, useCrossValidation, cvIndex);
}

std::vector<double> CyclicCoordinateDescent::applyBounds(std::vector<double> delta, int index) {

    auto doBound = true;
    if (doBound) {
    	for (int cvIndex = 0; cvIndex < syncCVFolds; cvIndex++) {
    		if (delta[cvIndex] < -hDeltaPool[cvIndex][index]) {
    			delta[cvIndex] = -hDeltaPool[cvIndex][index];
    		} else if (delta[cvIndex] > hDeltaPool[cvIndex][index]) {
    			delta[cvIndex] = hDeltaPool[cvIndex][index];
    		}


	    // TODO Remove magic numbers

	    auto intermediate = std::max(std::abs(delta[cvIndex]) * 2, hDeltaPool[cvIndex][index] / 2);
	    intermediate = std::max(intermediate, 1E-3);
	    hDeltaPool[cvIndex][index] = intermediate;

    	}
    }

	return delta;
}

void CyclicCoordinateDescent::applyBounds(std::vector<std::vector<double>>& delta) {

    auto doBound = true;
    if (doBound) {
		for (int cvIndex = 0; cvIndex < syncCVFolds; cvIndex++) {
			for (int index = 0; index < J; index++) {
				if (delta[cvIndex][index] < -hDeltaPool[cvIndex][index]) {
					delta[cvIndex][index] = -hDeltaPool[cvIndex][index];
				} else if (delta[cvIndex][index] > hDeltaPool[cvIndex][index]) {
					delta[cvIndex][index] = hDeltaPool[cvIndex][index];
				}
				auto intermediate = std::max(std::abs(delta[cvIndex][index]) * 2, hDeltaPool[cvIndex][index] / 2);
				intermediate = std::max(intermediate, 1E-3);
				hDeltaPool[cvIndex][index] = intermediate;
			}
    	}
    }
	//return delta;
}

void CyclicCoordinateDescent::applyBounds(std::vector<double>& delta, std::vector<std::pair<int,int>>& indicesToUpdate) {
    auto doBound = true;
    if (doBound) {
    	for (int i=0; i<delta.size(); i++) {
    		int index = indicesToUpdate[i].first;
    		int cvIndex = indicesToUpdate[i].second;
    		if (delta[i] < -hDeltaPool[cvIndex][index]) {
    			delta[i] = -hDeltaPool[cvIndex][index];
    		} else if (delta[i] > hDeltaPool[cvIndex][index]) {
    			delta[i] = hDeltaPool[cvIndex][index];
    		}
    		auto intermediate = std::max(std::abs(delta[i]) * 2, hDeltaPool[cvIndex][index] / 2);
    		intermediate = std::max(intermediate, 1E-3);
    		hDeltaPool[cvIndex][index] = intermediate;
    	}
    }
	//return delta;
}

void CyclicCoordinateDescent::updateXBeta(std::vector<double> delta, int index) {
	// Update beta
	std::vector<double> realDeltaVec;

	for (int cvIndex = 0; cvIndex < syncCVFolds; cvIndex++) {
		double realDelta = static_cast<double>(delta[cvIndex]);
		hBetaPool[cvIndex][index] += realDelta;
		realDeltaVec.push_back(realDelta);
		// Delegate
		//if (realDelta!=0.0) modelSpecifics.updateXBeta(realDelta, index, useCrossValidation, cvIndex);
	}
	modelSpecifics.updateXBeta(realDeltaVec, index, useCrossValidation);
}

void CyclicCoordinateDescent::updateSufficientStatistics(std::vector<double> delta, int index) {
	updateXBeta(delta, index);
	for (int i=0; i<syncCVFolds; i++) {
		if (hBetaPool[i][index] == 0.0 && fixBetaPool[i][index] == false) {
			nonZeros[i]--;
			fixBetaPool[i][index] = true;
		}
	}
	sufficientStatisticsKnown = true;
}

void CyclicCoordinateDescent::updateSufficientStatistics(std::vector<double> allDelta) {
	for (int index = 0; index < J; ++index) {
		if (!fixBeta[index]) {
			double delta = allDelta[index];
			double realDelta = static_cast<double>(applyBounds(delta, index));
			hBeta[index] += realDelta;
			allDelta[index] = realDelta;
			//updateSufficientStatistics(delta, index);
		}
	}
	//std::cout << "calling updateallxbeta \n";
	modelSpecifics.updateAllXBeta(allDelta, useCrossValidation);
}

void CyclicCoordinateDescent::updateSufficientStatistics(std::vector<double>& allDelta, std::vector<std::pair<int,int>>& indicesToUpdate) {
	for (int i=0; i<allDelta.size(); i++) {
		int index = indicesToUpdate[i].first;
		int cvIndex = indicesToUpdate[i].second;
		hBetaPool[cvIndex][index] += allDelta[i];
		if (hBetaPool[cvIndex][index] == 0.0 && fixBetaPool[cvIndex][index] == false) {
			nonZeros[cvIndex]--;
			fixBetaPool[cvIndex][index] = true;
			syncCVIterator.fix(index,cvIndex);
		}
	}
	/*
	    		auto func = [&](const tbb::blocked_range<int>& range) {
	    			for (int k = range.begin(); k < range.end(); ++k) {
	    				int current = temp[k];
	    				modelSpecifics.updateAllXBeta(allDeltaPool[current], useCrossValidation, current);
	    			}
	    		};

	    		tbb::parallel_for(tbb::blocked_range<int>(0,count),func);
	 */
	modelSpecifics.updateXBeta(allDelta, indicesToUpdate, useCrossValidation);
}


bool CyclicCoordinateDescent::performCheckConvergence(int convergenceType,
                                                      double epsilon,
                                                      int maxIterations,
                                                      int iteration,
                                                      std::vector<double>& lastObjFuncVec) {
	/*
	std::cout << "lastObjFunc: ";
    for (auto x : lastObjFuncVec) {
        std::cout << " " << x;
    }
    std::cout << '\n';
    */
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
    	/*
    	std::cout << "thisObjFunc: ";
        for (auto x : thisObjFuncVec) {
            std::cout << " " << x;
        }
        std::cout << '\n';
        */
    	//std::cout << "ObjFunc" << convergenceType << ": " << thisObjFunc << '\n';
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
    			if (conv[cvIndex] > epsilon) {
    				done = false;
    			}
    			if (!donePool[cvIndex] && epsilon > 0 && conv[cvIndex] < epsilon) {
    				donePool[cvIndex] = true;
    				for (int j = 0; j < J; j++) {
    					fixBetaPool[cvIndex][j] = true;
    				}
    				syncCVIterator.fix(cvIndex);
    			}
    		}
    		lastObjFuncVec[cvIndex] = thisObjFuncVec[cvIndex];
    	}
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

	/*
	std::cout << "weights" << syncCVIndex << ": ";
	for (auto x : hWeightsPool[syncCVIndex]) {
		std::cout << " " << x;
	}
	std::cout << '\n';
	*/

}

double CyclicCoordinateDescent::getPredictiveLogLikelihood(double* weights, int cvIndex) {

	/*
	std::cout << "weights: ";
	for (int x = 0; x < K; x++) {
		std::cout << " " << weights[x];
	}
	std::cout << "\n";
	*/

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

	if (donePool[cvIndex]) {
		return modelSpecifics.getPredictiveLogLikelihood(weights, cvIndex); // TODO Pass double
	} else {
		return std::numeric_limits<double>::quiet_NaN();
	}
}




} // namespace
