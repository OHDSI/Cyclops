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
	outLog << std::scientific << maxPoint << std::endl;
	outLog.close();
}

// This is specific to auto-search
double AutoSearchCrossValidationDriver::doCrossValidationLoop(
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
		
	UniModalSearch searcher(10, 0.01, log(1.5));

	std::ostringstream stream;
	stream << "Starting var = " << tryvalue;
	if (arguments.startingVariance == -1) {
	    stream << " (default)";   
	}
	logger->writeLine(stream);
	
	int step = 0;
	bool finished = false;
	
	while (!finished) {
		ccd.setHyperprior(tryvalue);
		selector.reseed();		

		std::vector<double> predLogLikelihood;

		// Newly re-located code
		double pointEstimate = doCrossValidationStep(ccd, selector, allArguments, step, 
			nThreads, ccdPool, selectorPool,
			predLogLikelihood);

		double stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);

        std::ostringstream stream;
				stream << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;
        searcher.tried(tryvalue, pointEstimate, stdDevEstimate);
        pair<bool,double> next = searcher.step();
        stream << "Completed at " << tryvalue << std::endl;
        stream << "Next point at " << next.second << " and " << next.first;
        logger->writeLine(stream);

        tryvalue = next.second;
        if (!next.first) {
            finished = true;
        }
        std::ostringstream stream1;
        stream1 << searcher;
        logger->writeLine(stream1);
        step++;
        if (step >= maxSteps) {
          std::ostringstream stream;
        	stream << "Max steps reached!";
        	logger->writeLine(stream);
        	finished = true;
        }
	}
	return tryvalue;
}

} // namespace
