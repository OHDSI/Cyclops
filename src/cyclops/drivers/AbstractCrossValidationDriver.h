/*
 * CrossValidationDriver.h
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#ifndef ABSTRACTCROSSVALIDATIONDRIVER_H_
#define ABSTRACTCROSSVALIDATIONDRIVER_H_

#include "AbstractDriver.h"

namespace bsccs {

 // forward references
class CyclicCoordinateDescent;
class AbstractSelector;
struct CCDArguments;

class AbstractCrossValidationDriver : public AbstractDriver {
public:
	AbstractCrossValidationDriver(
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
			std::vector<real>* wtsExclude = nullptr
	);

	virtual ~AbstractCrossValidationDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

	virtual void resetForOptimal(
			CyclicCoordinateDescent& ccd,
			CrossValidationSelector& selector,
			const CCDArguments& arguments);

	virtual void logResults(const CCDArguments& arguments) = 0; // pure virtual

protected:

    // Derived classes use different optimization loops
	virtual std::vector<double> doCrossValidationLoop(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool) = 0; // pure virtual

	double doCrossValidationStep(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments,
			int step,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool,
			std::vector<double> & predLogLikelihood);

	double computePointEstimate(const std::vector<double>& value);

	double computeStDev(const std::vector<double>& value, double mean);

	std::vector<double> maxPoint;
	std::vector<real>* weightsExclude;
};

} // namespace

#endif /* ABSTRACTCROSSVALIDATIONDRIVER_H_ */
