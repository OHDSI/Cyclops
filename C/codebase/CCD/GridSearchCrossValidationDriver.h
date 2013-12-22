/*
 * CrossValidationDriver.h
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#ifndef CROSSVALIDATIONDRIVER_H_
#define CROSSVALIDATIONDRIVER_H_

#include "AbstractCrossValidationDriver.h"

namespace bsccs {

class GridSearchCrossValidationDriver : public AbstractCrossValidationDriver {
public:
	GridSearchCrossValidationDriver(
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,
			vector<real>* wtsExclude = NULL);

	virtual ~GridSearchCrossValidationDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

	virtual void resetForOptimal(
			CyclicCoordinateDescent& ccd,
			CrossValidationSelector& selector,
			const CCDArguments& arguments);

	virtual void logResults(const CCDArguments& arguments);

private:

	double computeGridPoint(int step);

	double computePointEstimate(const std::vector<double>& value);

	void findMax(double* maxPoint, double* maxValue);

	std::vector<double> gridPoint;
	std::vector<double> gridValue;

	int gridSize;
	double lowerLimit;
	double upperLimit;
	vector<real>* weightsExclude;
};

} // namespace

#endif /* CROSSVALIDATIONDRIVER_H_ */
