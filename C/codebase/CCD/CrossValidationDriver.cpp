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

#include "CrossValidationDriver.h"

CrossValidationDriver::CrossValidationDriver(
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit) : gridSize(iGridSize),
			lowerLimit(iLowerLimit), upperLimit(iUpperLimit) {

	// Do anything???

}

CrossValidationDriver::~CrossValidationDriver() {
	// Do nothing
}

double CrossValidationDriver::computeGridPoint(int step) {
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
double CrossValidationDriver::computePointEstimate(const std::vector<double>& value) {
	// Mean of log values
	return accumulate(value.begin(), value.end(), 0.0);
}


void CrossValidationDriver::logResults(const CCDArguments& arguments) {

	ofstream outLog(arguments.cvFileName.c_str());
	if (!outLog) {
		cerr << "Unable to open log file: " << arguments.cvFileName << endl;
		exit(-1);
	}

	string sep(","); // TODO Make option

	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);

	for (int i = 0; i < gridPoint.size(); i++) {
		outLog << std::setw(5) << std::setprecision(4) << std::fixed << gridPoint[i] << sep;
		if (!arguments.useNormalPrior) {
			outLog << convertVarianceToHyperparameter(gridPoint[i]) << sep;
		}
		outLog << std::scientific << gridValue[i] << sep;
		outLog << (maxValue - gridValue[i]) << std::endl;
	}

	outLog.close();
}

void CrossValidationDriver::resetForOptimal(
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

void CrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		CrossValidationSelector& selector,
		const CCDArguments& arguments) {

	std::vector<real> weights;

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
			std::cout << "Running at " << ccd.getPriorInfo() << " ";
			ccd.update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);

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
		gridPoint.push_back(point);
		gridValue.push_back(value);
	}

	// Report results
	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);

	std::cout << std::endl;
	std::cout << "Maximum predicted log likelihood (" << maxValue << ") found at:" << std::endl;
	std::cout << "\t" << maxPoint << " (variance)" << std::endl;
	if (!arguments.useNormalPrior) {
		double lambda = convertVarianceToHyperparameter(maxPoint);
		std::cout << "\t" << lambda << " (lambda)" << std::endl;
	}
	std:cout << std::endl;
}


void CrossValidationDriver::findMax(double* maxPoint, double* maxValue) {

	*maxPoint = gridPoint[0];
	*maxValue = gridValue[0];
	for (int i = 1; i < gridPoint.size(); i++) {
		if (gridValue[i] > *maxValue) {
			*maxPoint = gridPoint[i];
			*maxValue = gridValue[i];
		}
	}
}

