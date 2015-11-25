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
            const CCDArguments& arguments,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
			vector<real>* wtsExclude) : AbstractCrossValidationDriver(_logger, _error, wtsExclude),
			gridSize(arguments.crossValidation.gridSteps),
			lowerLimit(arguments.crossValidation.lowerLimit),
			upperLimit(arguments.crossValidation.upperLimit)
// 			weightsExclude(wtsExclude)
			{

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


void GridSearchCrossValidationDriver::logResults(const CCDArguments& allArguments) {
    const auto& arguments = allArguments.crossValidation;
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
		if (!allArguments.useNormalPrior) {
			outLog << convertVarianceToHyperparameter(gridPoint[i]) << sep;
		}
		outLog << std::scientific << gridValue[i] << sep;
		outLog << (maxValue - gridValue[i]) << std::endl;
	}

	outLog.close();
}

// void GridSearchCrossValidationDriver::resetForOptimal(
// 		CyclicCoordinateDescent& ccd,
// 		CrossValidationSelector& selector,
// 		const CCDArguments& allArguments) {
//
// 	ccd.setWeights(NULL);
//
// 	double maxPoint;
// 	double maxValue;
// 	findMax(&maxPoint, &maxValue);
// 	ccd.setHyperprior(maxPoint);
// 	ccd.resetBeta(); // Cold-start
// }

std::vector<double> GridSearchCrossValidationDriver::doCrossValidationLoop(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& allArguments,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool) {

    const auto& arguments = allArguments.crossValidation;

// 	std::vector<real> weights;
	for (int step = 0; step < gridSize; step++) {

		std::vector<double> predLogLikelihood;
		double point = computeGridPoint(step);
		ccd.setHyperprior(point);
		selector.reseed();

		double pointEstimate = doCrossValidationStep(ccd, selector, allArguments, step,
			nThreads, ccdPool, selectorPool,
			predLogLikelihood);
		double value = pointEstimate / (double(arguments.foldToCompute) / double(arguments.fold));

		gridPoint.push_back(point);
		gridValue.push_back(value);
	}

	// Report results
	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);

//     std::ostringstream stream;
// 	stream << std::endl;
// 	stream << "Maximum predicted log likelihood (" << maxValue << ") found at:" << std::endl;
// 	stream << "\t" << maxPoint << " (variance)" << std::endl;
// 	if (!allArguments.useNormalPrior) {
// 		double lambda = convertVarianceToHyperparameter(maxPoint);
// 		stream << "\t" << lambda << " (lambda)" << std::endl;
// 	}
// 	logger->writeLine(stream);
    return std::vector<double>(1, maxPoint);
}


// void GridSearchCrossValidationDriver::drive(
// 		CyclicCoordinateDescent& ccd,
// 		AbstractSelector& selector,
// 		const CCDArguments& allArguments) {
//
// 	// TODO Check that selector is type of CrossValidationSelector
//
// 	const auto& arguments = allArguments.crossValidation;
//
// 	std::vector<real> weights;
//
// 	for (int step = 0; step < gridSize; step++) {
//
// 		std::vector<double> predLogLikelihood;
// 		double point = computeGridPoint(step);
// 		ccd.setHyperprior(point);
// 		selector.reseed();
//
// 		for (int i = 0; i < arguments.foldToCompute; i++) {
// 			int fold = i % arguments.fold;
// 			if (fold == 0) {
// 				selector.permute(); // Permute every full cross-validation rep
// 			}
//
// 			// Get this fold and update
// 			selector.getWeights(fold, weights);
// 			if(weightsExclude){
// 				for(int j = 0; j < (int)weightsExclude->size(); j++){
// 					if(weightsExclude->at(j) == 1.0){
// 						weights[j] = 0.0;
// 					}
// 				}
// 			}
// 			ccd.setWeights(&weights[0]);
//
// 			std::ostringstream stream;
// 			stream << "Running at " << ccd.getPriorInfo() << " ";
// 			stream << "Grid-point #" << (step + 1) << " at " << point;
// 			stream << "\tFold #" << (fold + 1)
// 					  << " Rep #" << (i / arguments.fold + 1) << " pred log like = ";
//
// 			ccd.update(allArguments.modeFinding);
//
// 			if (ccd.getUpdateReturnFlag() == SUCCESS) {
//
// 				// Compute predictive loglikelihood for this fold
// 				selector.getComplement(weights);
// 				if(weightsExclude){
// 					for(int j = 0; j < (int)weightsExclude->size(); j++){
// 						if(weightsExclude->at(j) == 1.0){
// 							weights[j] = 0.0;
// 						}
// 					}
// 				}
//
// 				double logLikelihood = ccd.getPredictiveLogLikelihood(&weights[0]);
//
// 				stream << logLikelihood;
// 				predLogLikelihood.push_back(logLikelihood);
// 			} else {
// 				ccd.resetBeta(); // cold start for stability
// 				stream << "Not computed";
// 				predLogLikelihood.push_back(std::numeric_limits<double>::quiet_NaN());
// 			}
//
// 			logger->writeLine(stream);
// 		}
//
// 		double value = computePointEstimate(predLogLikelihood) /
// 				(double(arguments.foldToCompute) / double(arguments.fold));
// 		gridPoint.push_back(point);
// 		gridValue.push_back(value);
// 	}
//
// 	// Report results
// 	double maxPoint;
// 	double maxValue;
// 	findMax(&maxPoint, &maxValue);
//
//     std::ostringstream stream;
// 	stream << std::endl;
// 	stream << "Maximum predicted log likelihood (" << maxValue << ") found at:" << std::endl;
// 	stream << "\t" << maxPoint << " (variance)" << std::endl;
// 	if (!allArguments.useNormalPrior) {
// 		double lambda = convertVarianceToHyperparameter(maxPoint);
// 		stream << "\t" << lambda << " (lambda)" << std::endl;
// 	}
// 	logger->writeLine(stream);
// }


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
