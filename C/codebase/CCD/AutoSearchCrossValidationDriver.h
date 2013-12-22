/*
 * AutoSearchCrossValidationDriver.h
 *
 *  Created on: Dec 10, 2013
 *      Author: msuchard
 */

#ifndef AUTOSEARCHCROSSVALIDATIONDRIVER_H_
#define AUTOSEARCHCROSSVALIDATIONDRIVER_H_

#include "AbstractCrossValidationDriver.h"

namespace bsccs {

class AutoSearchCrossValidationDriver : public AbstractCrossValidationDriver {
public:
	AutoSearchCrossValidationDriver(
			const ModelData& modelData,
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,
			vector<real>* wtsExclude = NULL);

	virtual ~AutoSearchCrossValidationDriver();

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

	double normBasedDefaultVar();

	double computeGridPoint(int step);

//	double computePointEstimate(const std::vector<double>& value);

	void findMax(double* maxPoint, double* maxValue);

	std::vector<double> gridPoint;
	std::vector<double> gridValue;

	int gridSize;
	double lowerLimit;
	double upperLimit;
	vector<real>* weightsExclude;

	const ModelData& modelData;
};

} // namespace

#endif /* AUTOSEARCHCROSSVALIDATIONDRIVER_H_ */
