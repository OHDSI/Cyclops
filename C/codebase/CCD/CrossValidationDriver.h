/*
 * CrossValidationDriver.h
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#ifndef CROSSVALIDATIONDRIVER_H_
#define CROSSVALIDATIONDRIVER_H_

#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "ccd.h"

class CrossValidationDriver {
public:
	CrossValidationDriver(
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit);

	virtual ~CrossValidationDriver();

	void drive(
			CyclicCoordinateDescent& ccd,
			CrossValidationSelector& selector,
			const CCDArguments& arguments);

	void logResults(const CCDArguments& arguments);

private:

	double computeGridPoint(int step);

	double computePointEstimate(const std::vector<double>& value);

	std::vector<double> gridPoint;
	std::vector<double> gridValue;

	int gridSize;
	double lowerLimit;
	double upperLimit;

};

#endif /* CROSSVALIDATIONDRIVER_H_ */
