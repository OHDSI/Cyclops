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

	void generateCholesky();

	void logResults(const CCDArguments& arguments, std::string conditionId);

	void initializeHessian();

private:
	InputReader* reader;

	vector<vector<bsccs::real> > credibleIntervals;

	vector<bsccs::real> BetaValues;

	vector<vector<bsccs::real> > hessian_notGSL;

	double precisionDeterminant;

	vector<vector<bsccs::real> > Cholesky_notGSL;

	int J;

	int maxIterations;

	int nBetaSamples;

	int nSigmaSquaredSamples;
};
}


#endif /* MCMCDRIVER_H_ */
