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
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,
			vector<real>* wtsExclude = NULL);

	virtual ~HierarchyGridSearchCrossValidationDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

};

} // namespace

#endif /* HIERARCHYGRIDCROSSVALIDATIONDRIVER_H_ */
