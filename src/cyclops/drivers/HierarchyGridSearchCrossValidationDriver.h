/*
 * HierarchyGridSearchCrossValidationDriver.h
 *
 *  Created on: April 10, 2014
 *      Author: Trevor Shaddox
 */

#ifndef HIERARCHYGRIDSEARCHCROSSVALIDATIONDRIVER_H_
#define HIERARCHYGRIDSEARCHCROSSVALIDATIONDRIVER_H_

#include "GridSearchCrossValidationDriver.h"

namespace bsccs {

class HierarchyGridSearchCrossValidationDriver : public GridSearchCrossValidationDriver {
public:
	HierarchyGridSearchCrossValidationDriver(
			const CCDArguments& arguments,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,			
			std::vector<real>* wtsExclude = NULL);

	virtual ~HierarchyGridSearchCrossValidationDriver();

	virtual void resetForOptimal(
			CyclicCoordinateDescent& ccd,
			CrossValidationSelector& selector,
			const CCDArguments& arguments);

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

private:

	void changeParameter(CyclicCoordinateDescent &ccd, int varianceIndex, double varianceValue);

	double maxPoint;
	double maxPointClass;

};

} // namespace

#endif /* HIERARCHYGRIDCROSSVALIDATIONDRIVER_H_ */
