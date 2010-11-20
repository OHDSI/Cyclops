/*
 * AbstractDriver.h
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#ifndef ABSTRACTDRIVER_H_
#define ABSTRACTDRIVER_H_

#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "ccd.h"

class AbstractDriver {
public:
	AbstractDriver();

	virtual ~AbstractDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments) = 0; // pure virtual

	virtual void logResults(const CCDArguments& arguments) = 0; // pure virtual
};

#endif /* ABSTRACTDRIVER_H_ */
