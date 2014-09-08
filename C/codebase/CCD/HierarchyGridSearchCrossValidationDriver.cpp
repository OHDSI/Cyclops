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

HierarchyGridSearchCrossValidationDriver::HierarchyGridSearchCrossValidationDriver(
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,
			vector<real>* wtsExclude) : GridSearchCrossValidationDriver(iGridSize,
					iLowerLimit,
					iUpperLimit,
					wtsExclude)
{
	// Do anything???
}

HierarchyGridSearchCrossValidationDriver::~HierarchyGridSearchCrossValidationDriver() {
	// Do nothing
}

void HierarchyGridSearchCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		CrossValidationSelector& selector,
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
		const CCDArguments& arguments) {

	std::vector<bsccs::real> weights;
	std::vector<double> outerPoints;
	std::vector<double> innerPoints;
	std::vector<double> outerValues;
	std::vector<double> minValues;

	for (int outerStep = 0; outerStep < gridSize; outerStep++){
		std::vector<double> predLogLikelihoodOuter;
		double outerPoint = computeGridPoint(outerStep);
		ccd.setClassHyperprior(outerPoint);

		for (int step = 0; step < gridSize; step++) {

			std::vector<double> predLogLikelihood;
			double point = computeGridPoint(step);
			ccd.setHyperprior(point);

			for (int i = 0; i < arguments.foldToCompute; i++) {

				int fold = i % arguments.fold;
				if (fold == 0) {
					selector.permute(); // Permute every full cross-validation rep
				}

				// Get this fold and update
				selector.getWeights(fold, weights);
				ccd.setWeights(&weights[0]);

				ccd.update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);
				// Compute predictive loglikelihood for this fold
				selector.getComplement(weights);
				double logLikelihood = ccd.getPredictiveLogLikelihood(&weights[0]);

			//	std::cout << "Grid-point #" << (step + 1) << " at " << point;
			//	std::cout << "\tFold #" << (fold + 1)
			//	          << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
			//	          << logLikelihood << std::endl;

				// Store value
				predLogLikelihood.push_back(logLikelihood);
			}

			double value = computePointEstimate(predLogLikelihood) /
					(double(arguments.foldToCompute) / double(arguments.fold));
			gridPoint.push_back(point);
			std::cout << "hyperprior point = " << point;
			std::cout << " class hyperprior point = " << outerPoint;
			cout << " value = " << value << endl;
			gridValue.push_back(value);
		}

		// Report results
		double maxPoint;
		double maxValue;
		double minValue;
		findMax(&maxPoint, &maxValue);
		minValues.push_back(minValue);

		innerPoints.push_back(maxPoint);
		outerPoints.push_back(outerPoint);
		outerValues.push_back(maxValue);

		if (!arguments.useNormalPrior) {
			double lambda = convertVarianceToHyperparameter(maxPoint);
			std::cout << "\t" << lambda << " (lambda)" << std::endl;
		}

	}
	maxPointClass = outerPoints[0];
	maxPoint = innerPoints[0];
	double outerMaxValue = outerValues[0];
	for (int i = 0; i < outerPoints.size(); i++) {
		if (outerValues[i] > outerMaxValue) {
			outerMaxValue = outerValues[i];
			maxPointClass = outerPoints[i];
			maxPoint = innerPoints[i];
		}
	}
	std::cout << std::endl;
	std::cout << "Maximum predicted log likelihood (" << outerMaxValue << ") found at:" << std::endl;
	std::cout << "\t" << maxPoint << " (drug variance) and at " << maxPointClass << " (class variance)" << std::endl;

}



} // namespace
