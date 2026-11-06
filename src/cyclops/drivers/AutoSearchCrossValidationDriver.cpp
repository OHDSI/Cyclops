/*
 * AutoSearchCrossValidationDriver.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: msuchard
 */

#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>
#include <iterator>
#include <algorithm>
#include <limits>
#include <memory>
#include <map>

#include "Types.h"
#include "Thread.h"
#include "AutoSearchCrossValidationDriver.h"
#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "../utils/HParSearch.h"

// #include "boost/iterator/counting_iterator.hpp"

namespace bsccs {

const static int MAX_STEPS = 50;

AutoSearchCrossValidationDriver::AutoSearchCrossValidationDriver(
			const AbstractModelData& _modelData,
			const CCDArguments& arguments,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
            vector<double>* wtsExclude
			) : AbstractCrossValidationDriver(_logger, _error, wtsExclude), modelData(_modelData),
			maxSteps(MAX_STEPS) {

	// Do anything???
}

AutoSearchCrossValidationDriver::~AutoSearchCrossValidationDriver() {
	// Do nothing
}

void AutoSearchCrossValidationDriver::logResults(const CCDArguments& allArguments) {
    const auto& arguments = allArguments.crossValidation;
	ofstream outLog(arguments.cvFileName.c_str());
	if (!outLog) {
	    std::ostringstream stream;
		stream << "Unable to open log file: " << arguments.cvFileName;
		error->throwError(stream);
	}
	outLog << std::scientific;
	for (int i = 0; i < maxPoint.point.size(); ++i) {
	    outLog << maxPoint.point[i] << " ";
	}
	outLog << std::endl;
	outLog.close();
}

// The search stops after a few points spaced by factors of 10 and returns the maximum of a quadratic that
// it has not evaluated, which can lie far from the optimum of the cross-validation curve. This evaluates
// that point and refits a quadratic to the best point and its two neighbours, until the next step is under
// 15% (at most three more evaluations, each warm-started when warm starts are on).
void AutoSearchCrossValidationDriver::refineSearch(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& allArguments,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool,
			int dim,
			int& step,
			std::vector<double>& triedValues,
			std::vector<std::vector<double>>& triedScores,
			std::vector<double>& currentOptimal,
			double& currentOptimalValue) {

	const int maxExtra = 3;
	const double stopByX = log(1.15);

	for (int extra = 0; extra < maxExtra; ++extra) {
		const double x = currentOptimal[dim];
		bool alreadyTried = false;
		for (double t : triedValues) {
			alreadyTried = alreadyTried || std::abs(log(t) - log(x)) < 1E-9;
		}
		if (alreadyTried) {
			break;
		}

		ccd.setHyperprior(dim, x);
		selector.reseed();
		std::vector<double> predLogLikelihood;
		doCrossValidationStep(ccd, selector, allArguments, step, nThreads, ccdPool, selectorPool,
		                      predLogLikelihood);
		++step;
		triedValues.push_back(x);
		triedScores.push_back(predLogLikelihood);

		std::vector<double> means, stdDevs;
		computeCommonFoldEstimates(triedScores, means, stdDevs);
		std::map<double, UniModalSearch::MS> byX;
		for (size_t p = 0; p < triedValues.size(); ++p) {
			if (means[p] == means[p]) {
				byX[triedValues[p]] = UniModalSearch::MS(means[p], stdDevs[p]);
			}
		}
		if (byX.empty()) {
			break;
		}
		auto best = byX.begin();
		for (auto it = byX.begin(); it != byX.end(); ++it) {
			if (it->second.m > best->second.m) {
				best = it;
			}
		}

		std::ostringstream stream;
		stream << "Refinement at " << x << ": AvgPred = " << means.back() << "; best " << best->first;

		if (best == byX.begin() || std::next(best) == byX.end()) { // the best point is not bracketed
			currentOptimal[dim] = best->first;
			currentOptimalValue = best->second.m;
			stream << "; stopping at " << currentOptimal[dim];
			logger->writeLine(stream);
			break;
		}

		std::map<double, UniModalSearch::MS> local(std::prev(best), std::next(std::next(best)));
		QuadrCoefs coefs = QuadrLogFit(local);
		if (!(coefs.c2 < 0.0)) { // not concave: keep the best point
			currentOptimal[dim] = best->first;
			currentOptimalValue = best->second.m;
			stream << "; not concave, stopping at " << currentOptimal[dim];
			logger->writeLine(stream);
			break;
		}
		double next = exp(-coefs.c1 / coefs.c2 / 2);
		next = std::min(std::max(next, std::prev(best)->first), std::next(best)->first);
		currentOptimal[dim] = next;
		currentOptimalValue = -coefs.c1 * coefs.c1 / coefs.c2 / 4 + coefs.c0;
		stream << "; next " << next;
		logger->writeLine(stream);
		if (std::abs(log(next) - log(best->first)) < stopByX) {
			break;
		}
	}
}

// This is specific to auto-search
MaxPoint AutoSearchCrossValidationDriver::doCrossValidationLoop(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& allArguments,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool) {

    const auto& arguments = allArguments.crossValidation;

	// With a Laplace prior, every penalized coefficient is 0 at variances below the null model's, and every
	// fold fit there is (nearly) the same model, so the search does not go below it: from a flat run of
	// equal scores it would stop at a model without covariates. Searching up from the default start begins
	// there (as the null model's variance moves with the scale of the outcome, and the default does not)
	const bool nullModelStart = arguments.startingVariance == -1 && arguments.searchDirection > 0;
	const double nullModelVariance = (arguments.nullModelBound || nullModelStart) ?
	    ccd.getNullModelVariance(allArguments.modeFinding) : -1.0;

	double tryvalue = (arguments.startingVariance > 0) ?
	    arguments.startingVariance :
		modelData.getNormalBasedDefaultVar();
	if (nullModelVariance > 0.0 && (nullModelStart || tryvalue < nullModelVariance)) {
	    tryvalue = nullModelVariance;
	}

	std::ostringstream stream;
	stream << "Starting var = " << tryvalue;
	if (arguments.startingVariance == -1) {
	    stream << ((nullModelVariance > 0.0 && tryvalue == nullModelVariance) ? " (null model)" : " (default)");
	}
	if (nullModelVariance > 0.0) {
	    stream << "; null-model variance " << nullModelVariance;
	}
	logger->writeLine(stream);

	const double tolerance = 1E-2; // TODO Make Cyclops argument

	int nDim = ccd.getHyperprior().size();

	std::vector<double> currentOptimal(nDim, tryvalue);
	double currentOptimalValue;

	bool globalFinished = false;
	std::vector<double> savedOptimal;

	while (!globalFinished) {

	    if (nDim > 1) {
	        savedOptimal = currentOptimal; // make copy
	    }

	    for (int dim = 0; dim < nDim; ++dim) {

	        // Local search. The searcher is rebuilt after each step, so that every hyperparameter tried is
	        // compared over the same folds (computeCommonFoldEstimates()). The first step goes up from a
	        // variance below firstCut and down otherwise.
	        const double firstCut = (arguments.searchDirection == 0) ? 1.0 :
	            (arguments.searchDirection > 0) ? std::numeric_limits<double>::infinity() : 0.0;
	        std::vector<double> triedValues;
	        std::vector<std::vector<double>> triedScores;
	        std::unique_ptr<UniModalSearch> searcher;
	        bool searchFailed = false;

	        int step = 0;
	        bool dimFinished = false;

	        while (!dimFinished) {

	            ccd.setHyperprior(dim, currentOptimal[dim]);
	            selector.reseed();

	            std::vector<double> predLogLikelihood;

	            // Newly re-located code
	            doCrossValidationStep(ccd, selector, allArguments, step,
                                      nThreads, ccdPool, selectorPool,
                                      predLogLikelihood);

	            const double tried = currentOptimal[dim];
	            size_t current = std::find(triedValues.begin(), triedValues.end(), tried) - triedValues.begin();
	            if (current == triedValues.size()) {
	                triedValues.push_back(tried);
	                triedScores.push_back(predLogLikelihood);
	            } else {
	                triedScores[current] = predLogLikelihood;
	            }

	            std::vector<double> means, stdDevs;
	            computeCommonFoldEstimates(triedScores, means, stdDevs);
	            // least squares: the cross-validated likelihood is far from quadratic over several decades of
	            // variance (flat towards no penalty, steep towards the null model), so fit it locally
	            searcher.reset(new UniModalSearch(10, 0.01, log(1.5), firstCut,
	                                              modelData.getModelType() == ModelType::NORMAL));
	            int nScored = 0;
	            for (size_t p = 0; p < triedValues.size(); ++p) {
	                if (means[p] == means[p]) { // a hyperparameter without any computed fold is left out
	                    searcher->tried(triedValues[p], means[p], stdDevs[p]);
	                    ++nScored;
	                }
	            }

	            double pointEstimate = means[current];
	            double stdDevEstimate = stdDevs[current];
/*
	            std::cout << "predLikelihoods: ";
	            for (auto x : predLogLikelihood) {
	            	std::cout << x << " ";
	            }
	            std::cout << "\n";
*/
	            std::ostringstream stream;
	            stream << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;

	            if (!(pointEstimate == pointEstimate)) {
	                // No fold fit converged at this hyperparameter: stop at the best one with converged folds
	                if (nScored == 0) {
	                    std::ostringstream errorStream;
	                    errorStream << "Cross-validation failed: no fold fit converged at " << tried;
	                    error->throwError(errorStream);
	                }
	                currentOptimal[dim] = searcher->bestx();
	                currentOptimalValue = searcher->besty();
	                stream << "No fold fit converged at " << tried << "; stopping at " << currentOptimal[dim];
	                logger->writeLine(stream);
	                dimFinished = true;
	                searchFailed = true;
	                step++;
	                continue;
	            }

	            StepValue next = searcher->step();
	            if (nullModelVariance > 0.0 && next.second < nullModelVariance) {
	                // Below the null-model variance: try it instead. Just above it, the first coefficients to
	                // enter can score below the null model, so when it is the best point the search looks further
	                // up (to 100 times its variance) before stopping there
	                double largest = 0.0;
	                bool nullTried = false;
	                for (double t : triedValues) {
	                    largest = std::max(largest, t);
	                    nullTried = nullTried || std::abs(log(t) - log(nullModelVariance)) < 1E-9;
	                }
	                if (!nullTried) {
	                    next = StepValue(true, nullModelVariance, 0.0);
	                } else if (std::abs(log(searcher->bestx()) - log(nullModelVariance)) < 1E-9 &&
	                           largest < 100.0 * nullModelVariance) {
	                    next = StepValue(true, largest * 10.0, 0.0);
	                } else {
	                    next = StepValue(false, searcher->bestx(), searcher->besty());
	                }
	            }
	            stream << "Completed at " << currentOptimal[dim] << std::endl;
	            stream << "Next point at " << next.second << " with value " << next.expected << " and continue = " << next.first;
	            logger->writeLine(stream);

	            if (next.second > 0.0 && next.second < std::numeric_limits<double>::infinity()) {
	                currentOptimal[dim] = next.second;
	                currentOptimalValue = next.expected;
	                if (!next.first) {
	                    dimFinished = true;
	                }
	            } else { // never search at a non-finite hyperparameter: stop at the best one tried
	                currentOptimal[dim] = searcher->bestx();
	                currentOptimalValue = searcher->besty();
	                dimFinished = true;
	                searchFailed = true;
	            }
	            std::ostringstream stream1;
	            stream1 << *searcher;
	            logger->writeLine(stream1);
	            step++;
	            if (step >= maxSteps) {
	                std::ostringstream stream;
	                stream << "Max steps reached!";
	                logger->writeLine(stream);
	                dimFinished = true;
	            }
	        }

	        if (arguments.refineSearch && !searchFailed) {
	            refineSearch(ccd, selector, allArguments, nThreads, ccdPool, selectorPool, dim, step,
	                         triedValues, triedScores, currentOptimal, currentOptimalValue);
	        }
	    }

	    if (nDim == 1) {
	        globalFinished = true;
	    } else {

	        double diff = 0.0;
	        for (int i = 0; i < nDim; ++i) {
	            diff += std::abs((currentOptimal[i] - savedOptimal[i]) / savedOptimal[i]);
	        }
	        std::ostringstream stream;
	        stream << "Absolute percent difference in cycle: " << diff << std::endl;

	        globalFinished = (diff < tolerance);
	    }
	}
	return MaxPoint{currentOptimal, currentOptimalValue};
}

} // namespace
