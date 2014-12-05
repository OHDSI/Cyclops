/*
 * CrossValidationDriver.cpp
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>

#include "GridSearchCrossValidationDriver.h"

namespace bsccs {

using std::vector;

GridSearchCrossValidationDriver::GridSearchCrossValidationDriver(
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,			
			vector<real>* wtsExclude) : AbstractCrossValidationDriver(_logger, _error), gridSize(iGridSize),
			lowerLimit(iLowerLimit), upperLimit(iUpperLimit), weightsExclude(wtsExclude) {

	// Do anything???
}

GridSearchCrossValidationDriver::~GridSearchCrossValidationDriver() {
	// Do nothing
}

double GridSearchCrossValidationDriver::computeGridPoint(int step) {
	if (gridSize == 1) {
		return upperLimit;
	}
	// Linear grid
//	double stepSize = (upperLimit - lowerLimit) / (gridSize - 1);
//	return lowerLimit + step * stepSize;
	// Log uniform grid
	double stepSize = (log(upperLimit) - log(lowerLimit)) / (gridSize - 1);
	return exp(log(lowerLimit) + step * stepSize);
}
//double GridSearchCrossValidationDriver::computePointEstimate(const std::vector<double>& value) {
//	// Mean of log values
//	return accumulate(value.begin(), value.end(), 0.0);
//}


void GridSearchCrossValidationDriver::logResults(const CCDArguments& arguments) {

	ofstream outLog(arguments.cvFileName.c_str());
	if (!outLog) {
	    std::ostringstream stream;
		stream << "Unable to open log file: " << arguments.cvFileName;
		error->throwError(stream);		
	}

	string sep(","); // TODO Make option

	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);

	for (size_t i = 0; i < gridPoint.size(); i++) {
		outLog << std::setw(5) << std::setprecision(4) << std::fixed << gridPoint[i] << sep;
		if (!arguments.useNormalPrior) {
			outLog << convertVarianceToHyperparameter(gridPoint[i]) << sep;
		}
		outLog << std::scientific << gridValue[i] << sep;
		outLog << (maxValue - gridValue[i]) << std::endl;
	}

	outLog.close();
}

void GridSearchCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		CrossValidationSelector& selector,
		const CCDArguments& arguments) {

	ccd.setWeights(NULL);

	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);
	ccd.setHyperprior(maxPoint);
	ccd.resetBeta(); // Cold-start
}

void GridSearchCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// TODO Check that selector is type of CrossValidationSelector

	std::vector<real> weights;

	for (int step = 0; step < gridSize; step++) {

		std::vector<double> predLogLikelihood;
		double point = computeGridPoint(step);
		ccd.setHyperprior(point);
		selector.reseed();

		for (int i = 0; i < arguments.foldToCompute; i++) {
			int fold = i % arguments.fold;
			if (fold == 0) {
				selector.permute(); // Permute every full cross-validation rep
			}

			// Get this fold and update
			selector.getWeights(fold, weights);
			if(weightsExclude){
				for(int j = 0; j < (int)weightsExclude->size(); j++){
					if(weightsExclude->at(j) == 1.0){
						weights[j] = 0.0;
					}
				}
			}
			ccd.setWeights(&weights[0]);
			std::ostringstream stream;
			stream << "Running at " << ccd.getPriorInfo() << " ";
			ccd.update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);

			// Compute predictive loglikelihood for this fold
			selector.getComplement(weights);
			if(weightsExclude){
				for(int j = 0; j < (int)weightsExclude->size(); j++){
					if(weightsExclude->at(j) == 1.0){
						weights[j] = 0.0;
					}
				}
			}

			double logLikelihood = ccd.getPredictiveLogLikelihood(&weights[0]);

			stream << "Grid-point #" << (step + 1) << " at " << point;
			stream << "\tFold #" << (fold + 1)
			          << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
			          << logLikelihood;
            logger->writeLine(stream);			          

			// Store value
			predLogLikelihood.push_back(logLikelihood);
		}

		double value = computePointEstimate(predLogLikelihood) /
				(double(arguments.foldToCompute) / double(arguments.fold));
		gridPoint.push_back(point);
		gridValue.push_back(value);
	}

	// Report results
	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);

    std::ostringstream stream;
	stream << std::endl;
	stream << "Maximum predicted log likelihood (" << maxValue << ") found at:" << std::endl;
	stream << "\t" << maxPoint << " (variance)" << std::endl;
	if (!arguments.useNormalPrior) {
		double lambda = convertVarianceToHyperparameter(maxPoint);
		stream << "\t" << lambda << " (lambda)" << std::endl;
	}
	logger->writeLine(stream);	
}


void GridSearchCrossValidationDriver::findMax(double* maxPoint, double* maxValue) {

	*maxPoint = gridPoint[0];
	*maxValue = gridValue[0];
	for (size_t i = 1; i < gridPoint.size(); i++) {
		if (gridValue[i] > *maxValue) {
			*maxPoint = gridPoint[i];
			*maxValue = gridValue[i];
		}
	}
}

} // namespace
