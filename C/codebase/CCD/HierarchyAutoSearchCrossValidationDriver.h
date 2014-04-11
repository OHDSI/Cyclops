/*
 * HierarchyAutoSearchCrossValidationDriver.h
 *
 *  Created on: April 10, 2014
 *      Author: Trevor Shaddox
 */

#ifndef HIERARCHYAUTOSEARCHCROSSVALIDATIONDRIVER_H_
#define HIERARCHYAUTOSEARCHCROSSVALIDATIONDRIVER_H_

#include "AutoSearchCrossValidationDriver.h"

namespace bsccs {

class HierarchyAutoSearchCrossValidationDriver : public AutoSearchCrossValidationDriver {
public:
	HierarchyAutoSearchCrossValidationDriver(const ModelData& _modelData,
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,
			vector<real>* wtsExclude = NULL);

	virtual ~HierarchyAutoSearchCrossValidationDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

};

} // namespace

#endif /* HIERARCHYAUTOSEARCHCROSSVALIDATIONDRIVER_H_ */
