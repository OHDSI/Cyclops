/*
 * MCMCDriver.h
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */

#ifndef MCMCDRIVER_H_
#define MCMCDRIVER_H_

#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "ccd.h"

namespace bsccs {
class MCMCDriver {
public:
	MCMCDriver();

	virtual ~MCMCDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd);



private:
	int maxIterations;

	int sampleSize;
};
}


#endif /* MCMCDRIVER_H_ */
