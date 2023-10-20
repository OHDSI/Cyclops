/*
 * HierarchyGridSearchCrossValidationDriver.cpp
 *
 *  Created on: April 10, 2014
 *      Author: Trevor Shaddox
 */


// TODO Change from fixed grid to adaptive approach in BBR

#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>

#include "HierarchyGridSearchCrossValidationDriver.h"

namespace bsccs {

using std::vector;

HierarchyGridSearchCrossValidationDriver::HierarchyGridSearchCrossValidationDriver(
			const CCDArguments& arguments,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
			vector<double>* wtsExclude) : GridSearchCrossValidationDriver(
			        arguments,
					_logger,
					_error,
					wtsExclude)
{
	// Do anything???
}

HierarchyGridSearchCrossValidationDriver::~HierarchyGridSearchCrossValidationDriver() {
	// Do nothing
}

void HierarchyGridSearchCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {


	ccd.setWeights(NULL);
	ccd.setHyperprior(maxPoint);
	ccd.setClassHyperprior(maxPointClass);
	ccd.resetBeta(); // Cold-start
}

/* Author: tshaddox
 * Changes one parameter in the ccd
 */

void HierarchyGridSearchCrossValidationDriver::changeParameter(CyclicCoordinateDescent &ccd, int varianceIndex, double varianceValue) {
	if (varianceIndex == 0) {
		ccd.setHyperprior(varianceValue);

	}
	if (varianceIndex == 1) {
		ccd.setClassHyperprior(varianceValue);
	}
}


void HierarchyGridSearchCrossValidationDriver::drive(CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& allArguments) {

    const auto& arguments = allArguments.crossValidation;

	std::vector<double> weights;
	std::vector<double> outerPoints;
	std::vector<double> innerPoints;
	std::vector<double> outerValues;
	// std::vector<double> minValues;

	for (int outerStep = 0; outerStep < gridSize; outerStep++){
		std::vector<double> predLogLikelihoodOuter;
		double outerPoint = computeGridPoint(outerStep);
		ccd.setClassHyperprior(outerPoint);

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
				ccd.setWeights(&weights[0]);

				ccd.update(allArguments.modeFinding);
				// Compute predictive loglikelihood for this fold
				selector.getComplement(weights);
				double logLikelihood = ccd.getNewPredictiveLogLikelihood(&weights[0]);

                std::ostringstream stream;
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
		// double minValue; removed for v3.3.0; TODO why was this here?
		findMax(&maxPoint, &maxValue);
		// minValues.push_back(minValue);

		innerPoints.push_back(maxPoint);
		outerPoints.push_back(outerPoint);
		outerValues.push_back(maxValue);

		if (!allArguments.useNormalPrior) {
			double lambda = convertVarianceToHyperparameter(maxPoint);
			std::ostringstream stream;
			stream << "\t" << lambda << " (lambda)";
			logger->writeLine(stream);
		}

	}
	maxPointClass = outerPoints[0];
	maxPoint = innerPoints[0];
	double outerMaxValue = outerValues[0];
	for (size_t i = 0; i < outerPoints.size(); i++) {
		if (outerValues[i] > outerMaxValue) {
			outerMaxValue = outerValues[i];
			maxPointClass = outerPoints[i];
			maxPoint = innerPoints[i];
		}
	}

	std::ostringstream stream;
	stream << std::endl;
	stream << "Maximum predicted log likelihood (" << outerMaxValue << ") found at:" << std::endl;
	stream << "\t" << maxPoint << " (drug variance) and at " << maxPointClass << " (class variance)";
	logger->writeLine(stream);

}



} // namespace
