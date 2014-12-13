/*
 * AutoSearchCrossValidationDriver.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: msuchard
 */


// TODO Change from fixed grid to adaptive approach in BBR

#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>
#include <iterator>

#include "AutoSearchCrossValidationDriver.h"
#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
//#include "ccd.h"
#include "../utils/HParSearch.h"

namespace bsccs {

const static int MAX_STEPS = 50;

AutoSearchCrossValidationDriver::AutoSearchCrossValidationDriver(
			const ModelData& _modelData,
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,			
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,			
            vector<real>* wtsExclude			
			) : AbstractCrossValidationDriver(_logger, _error), modelData(_modelData), maxPoint(0), gridSize(iGridSize),
			lowerLimit(iLowerLimit), upperLimit(iUpperLimit), weightsExclude(wtsExclude),
			maxSteps(MAX_STEPS) {

	// Do anything???
}

AutoSearchCrossValidationDriver::~AutoSearchCrossValidationDriver() {
	// Do nothing
}

double AutoSearchCrossValidationDriver::computeGridPoint(int step) {
	if (gridSize == 1) {
		return upperLimit;
	}
	// Log uniform grid
	double stepSize = (log(upperLimit) - log(lowerLimit)) / (gridSize - 1);
	return exp(log(lowerLimit) + step * stepSize);
}

void AutoSearchCrossValidationDriver::logResults(const CCDArguments& arguments) {

	ofstream outLog(arguments.cvFileName.c_str());
	if (!outLog) {
	    std::ostringstream stream;
		stream << "Unable to open log file: " << arguments.cvFileName;
		error->throwError(stream);		
	}
	outLog << std::scientific << maxPoint << std::endl;
	outLog.close();
}

void AutoSearchCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		CrossValidationSelector& selector,
		const CCDArguments& arguments) {

	ccd.setWeights(NULL);
	ccd.setHyperprior(maxPoint);
	ccd.resetBeta(); // Cold-start
}


double AutoSearchCrossValidationDriver::doCrossValidation(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments,
		int step,
		std::vector<double> & predLogLikelihood){


	std::vector<real> weights;

	/* start code duplication */
	//std::vector<double> predLogLikelihood;
	for (int i = 0; i < arguments.foldToCompute; i++) { // TODO PARALLEL
		int fold = i % arguments.fold;
		if (fold == 0) {
			selector.permute(); // Permute every full cross-validation rep
		}

		// Get this fold and update
		selector.getWeights(fold, weights); // TODO THREAD-SAFE
		if(weightsExclude){
			for(int j = 0; j < (int)weightsExclude->size(); j++){
				if(weightsExclude->at(j) == 1.0){
					weights[j] = 0.0;
				}
			}
		}
		ccd.setWeights(&weights[0]); // TODO THREAD-SPECIFIC
		std::ostringstream stream;
		stream << "Running at " << ccd.getPriorInfo() << " ";
		 // TODO THREAD-SPECIFIC
		ccd.update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);

		// Compute predictive loglikelihood for this fold
		selector.getComplement(weights);  // TODO THREAD_SAFE
		if(weightsExclude){
			for(int j = 0; j < (int)weightsExclude->size(); j++){
				if(weightsExclude->at(j) == 1.0){
					weights[j] = 0.0;
				}
			}
		}

		double logLikelihood = ccd.getPredictiveLogLikelihood(&weights[0]);  // TODO THREAD-SPECIFIC

		stream << "Grid-point #" << (step + 1) << " at "; // << ccd.getHyperprior();
		std::vector<double> hyperprior = ccd.getHyperprior();
		std::copy(hyperprior.begin(), hyperprior.end(),
		    std::ostream_iterator<double>(stream, " "));
		
		stream << "\tFold #" << (fold + 1)
				  << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
				  << logLikelihood;
        logger->writeLine(stream);				  

		// Store value
		predLogLikelihood.push_back(logLikelihood); // TODO THREAD-SAFE
	}

	double pointEstimate = computePointEstimate(predLogLikelihood);
	/* end code duplication */

	return(pointEstimate);

}

void AutoSearchCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// TODO Check that selector is type of CrossValidationSelector

	double tryvalue = modelData.getNormalBasedDefaultVar();
	UniModalSearch searcher(10, 0.01, log(1.5));
//	const double eps = 0.05; //search stopper
	std::ostringstream stream;
	stream << "Default var = " << tryvalue;
	logger->writeLine(stream);

	bool finished = false;

	int step = 0;
	while (!finished) {
		ccd.setHyperprior(tryvalue);
		selector.reseed();		

		std::vector<double> predLogLikelihood;

		// Newly re-located code
		double pointEstimate = doCrossValidation(ccd, selector, arguments, step, predLogLikelihood);

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

	maxPoint = tryvalue;

	// Report results
	std::ostringstream stream1;
	stream1 << std::endl;
	stream1 << "Maximum predicted log likelihood estimated at:" << std::endl;
	stream1 << "\t" << maxPoint << " (variance)" << std::endl;
	if (!arguments.useNormalPrior) {
		double lambda = convertVarianceToHyperparameter(maxPoint);
		stream1 << "\t" << lambda << " (lambda)" << std::endl;
	}	
	logger->writeLine(stream1);
}

} // namespace
