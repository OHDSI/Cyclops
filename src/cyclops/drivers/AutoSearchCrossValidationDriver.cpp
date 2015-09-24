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

#include "Types.h"
#include "Thread.h"
#include "AutoSearchCrossValidationDriver.h"
#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "../utils/HParSearch.h"

#include "boost/iterator/counting_iterator.hpp"

namespace bsccs {

const static int MAX_STEPS = 50;

AutoSearchCrossValidationDriver::AutoSearchCrossValidationDriver(
			const ModelData& _modelData,
			const CCDArguments& arguments,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
            vector<real>* wtsExclude
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
	for (int i = 0; i < maxPoint.size(); ++i) {
	    outLog << maxPoint[i] << " ";
	}
	outLog << std::endl;
	outLog.close();
}

// This is specific to auto-search
std::vector<double> AutoSearchCrossValidationDriver::doCrossValidationLoop(
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

	bool globalFinished = false;
	std::vector<double> savedOptimal;

	while (!globalFinished) {

	    if (nDim > 1) {
	        savedOptimal = currentOptimal; // make copy
	    }

	    for (int dim = 0; dim < nDim; ++dim) {

	        // Local search
	        UniModalSearch searcher(10, 0.01, log(1.5));

	        int step = 0;
	        bool dimFinished = false;

	        while (!dimFinished) {

	            ccd.setHyperprior(dim, currentOptimal[dim]);
	            selector.reseed();

	            std::vector<double> predLogLikelihood;

	            // Newly re-located code
	            double pointEstimate = doCrossValidationStep(ccd, selector, allArguments, step,
                                                          nThreads, ccdPool, selectorPool,
                                                          predLogLikelihood);

	            double stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);

	            std::ostringstream stream;
	            stream << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;
	            searcher.tried(currentOptimal[dim], pointEstimate, stdDevEstimate);
	            pair<bool,double> next = searcher.step();
	            stream << "Completed at " << currentOptimal[dim] << std::endl;
	            stream << "Next point at " << next.second << " and " << next.first;
	            logger->writeLine(stream);

	            currentOptimal[dim] = next.second;
	            if (!next.first) {
	                dimFinished = true;
	            }
	            std::ostringstream stream1;
	            stream1 << searcher;
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
	return currentOptimal;
}

} // namespace
