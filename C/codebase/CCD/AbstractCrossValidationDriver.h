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
class CCDArguments;

class AbstractCrossValidationDriver : public AbstractDriver {
public:
	AbstractCrossValidationDriver();

	virtual ~AbstractCrossValidationDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments) = 0; // pure virtual

	virtual void resetForOptimal(
			CyclicCoordinateDescent& ccd,
			CrossValidationSelector& selector,
			const CCDArguments& arguments) = 0; // pure virtual

	virtual void logResults(const CCDArguments& arguments) = 0; // pure virtual
};

} // namespace

#endif /* ABSTRACTCROSSVALIDATIONDRIVER_H_ */
