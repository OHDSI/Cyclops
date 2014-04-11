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


void HierarchyGridSearchCrossValidationDriver::drive(CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	cout << "hierarchy drive" << endl;
	std::vector<bsccs::real> weights;
	std::vector<double> outerPoints;
	std::vector<double> innerPoints;
	std::vector<double> outerValues;
	std::vector<double> minValues;

	for (int outerStep = 0; outerStep < gridSize; outerStep++){
		std::vector<double> predLogLikelihoodOuter;
		double outerPoint = computeGridPoint(outerStep);
		ccd.setClassHyperprior(outerPoint);
		cout << "outerPoint = " << outerPoint << endl;
		cout << outerStep << " out of " << gridSize << endl;

		for (int step = 0; step < gridSize; step++) {

			std::vector<double> predLogLikelihood;
			double point = computeGridPoint(step);
			cout << "point = " << point << endl;
			ccd.setHyperprior(point);

			for (int i = 0; i < arguments.foldToCompute; i++) {

				int fold = i % arguments.fold;
				if (fold == 0) {
					selector.permute(); // Permute every full cross-validation rep
				}

				// Get this fold and update
				selector.getWeights(fold, weights);
				ccd.setWeights(&weights[0]);
			//	std::cout << "Running at " << ccd.getPriorInfo() << " ";
				ccd.update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);  //tshaddox temporary comment out
				// Compute predictive loglikelihood for this fold
				selector.getComplement(weights);
				double logLikelihood = ccd.getPredictiveLogLikelihood(&weights[0]);

				std::cout << "Grid-point #" << (step + 1) << " at " << point;
				std::cout << "\tFold #" << (fold + 1)
				          << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
				          << logLikelihood << std::endl;

				// Store value
				predLogLikelihood.push_back(logLikelihood);
			}

			double value = computePointEstimate(predLogLikelihood) /
					(double(arguments.foldToCompute) / double(arguments.fold));
			cout << "\t value = " << value << endl;
			gridPoint.push_back(point);
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



		std::cout << std::endl;
		std::cout << "Maximum predicted log likelihood (" << maxValue << ") found at:" << std::endl;
		std::cout << "\t" << maxPoint << " (variance)" << std::endl;

		if (!arguments.useNormalPrior) {
			double lambda = convertVarianceToHyperparameter(maxPoint);
			std::cout << "\t" << lambda << " (lambda)" << std::endl;
		}

		//std:cout << std::endl;

	}
	double outerMaxPoint = outerPoints[0];
	double innerMaxPoint = innerPoints[0];
	double outerMaxValue = outerValues[0];
	for (int i = 0; i < outerPoints.size(); i++) {
		if (outerValues[i] > outerMaxValue) {
			outerMaxValue = outerValues[i];
			outerMaxPoint = outerPoints[i];
			innerMaxPoint = innerPoints[i];
		}
	}
	std::cout << std::endl;
	std::cout << "Maximum predicted log likelihood (" << outerMaxValue << ") found at:" << std::endl;
	std::cout << "\t" << innerMaxPoint << " (drug variance) and at " << outerMaxPoint << " (class variance)" << std::endl;

}



} // namespace
