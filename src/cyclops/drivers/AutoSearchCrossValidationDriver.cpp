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

// This is specific to auto-search
MaxPoint AutoSearchCrossValidationDriver::doCrossValidationLoop(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& allArguments,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool) {

    const auto& arguments = allArguments.crossValidation;

	double tryvalue = (arguments.startingVariance > 0) ?
	    arguments.startingVariance :
		modelData.getNormalBasedDefaultVar();

	std::ostringstream stream;
	stream << "Starting var = " << tryvalue;
	if (arguments.startingVariance == -1) {
	    stream << " (default)";
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
	        // compared over the same folds (computeCommonFoldEstimates())
	        std::vector<double> triedValues;
	        std::vector<std::vector<double>> triedScores;
	        std::unique_ptr<UniModalSearch> searcher;

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
	            searcher.reset(new UniModalSearch(10, 0.01, log(1.5)));
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
	                step++;
	                continue;
	            }

	            StepValue next = searcher->step();
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
