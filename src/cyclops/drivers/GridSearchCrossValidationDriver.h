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
            const CCDArguments& arguments,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
			std::vector<real>* wtsExclude = NULL);

	virtual ~GridSearchCrossValidationDriver();

// 	virtual void drive(
// 			CyclicCoordinateDescent& ccd,
// 			AbstractSelector& selector,
// 			const CCDArguments& arguments);
//
// 	virtual void resetForOptimal(
// 			CyclicCoordinateDescent& ccd,
// 			CrossValidationSelector& selector,
// 			const CCDArguments& arguments);

	virtual void logResults(const CCDArguments& arguments);

protected:

	double computeGridPoint(int step);

	virtual MaxPoint doCrossValidationLoop(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool);

// 	double doCrossValidationStep(
// 			CyclicCoordinateDescent& ccd,
// 			AbstractSelector& selector,
// 			const CCDArguments& arguments,
// 			int step,
// 			bool coldStart,
// 			int nThreads,
// 			std::vector<CyclicCoordinateDescent*>& ccdPool,
// 			std::vector<AbstractSelector*>& selectorPool,
// 			std::vector<double> & predLogLikelihood);

//	double computePointEstimate(const std::vector<double>& value);

	void findMax(double* maxPoint, double* maxValue);

	std::vector<double> gridPoint;
	std::vector<double> gridValue;

	int gridSize;
	double lowerLimit;
	double upperLimit;
// 	std::vector<real>* weightsExclude;
};

} // namespace

#endif /* CROSSVALIDATIONDRIVER_H_ */
