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
	MCMCDriver(InputReader * inReader);

	virtual ~MCMCDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd);

	void logResults(const CCDArguments& arguments, std::string conditionId);

private:
	InputReader* reader;

	vector<vector<bsccs::real> > credibleIntervals;

	vector<bsccs::real> BetaValues;

	int J;

	int maxIterations;

	int nBetaSamples;

	int nSigmaSquaredSamples;
};
}


#endif /* MCMCDRIVER_H_ */
