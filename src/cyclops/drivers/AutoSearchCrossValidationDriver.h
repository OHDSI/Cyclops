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
			const AbstractModelData& modelData,
			const CCDArguments& arguments,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
			std::vector<double>* wtsExclude = NULL);

	virtual ~AutoSearchCrossValidationDriver();

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

private:

	double normBasedDefaultVar();

// 	double computeGridPoint(int step);

// 	void findMax(double* maxPoint, double* maxValue);

protected:

//	double computePointEstimate(const std::vector<double>& value);

// 	std::vector<double> gridPoint;
// 	std::vector<double> gridValue;

	const AbstractModelData& modelData;
// 	double maxPoint;
// 	int gridSize;
// 	double lowerLimit;
// 	double upperLimit;
// 	std::vector<double>* weightsExclude;
	double maxSteps;

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
// 			int nThreads,
// 			std::vector<CyclicCoordinateDescent*>& ccdPool,
// 			std::vector<AbstractSelector*>& selectorPool,
// 			std::vector<double> & predLogLikelihood);

};

} // namespace

#endif /* AUTOSEARCHCROSSVALIDATIONDRIVER_H_ */
