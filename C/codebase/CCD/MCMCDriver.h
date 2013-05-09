/*
 * MCMCDriver.h
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */

#ifndef MCMCDRIVER_H_
#define MCMCDRIVER_H_


#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "ccd.h"

namespace bsccs {
class MCMCDriver {
public:
	MCMCDriver(InputReader * inReader, std::string MCMCFileName);

	virtual ~MCMCDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd, double betaAmount, long int seed);

	void generateCholesky();

	void initializeHessian();
	void clearHessian();

private:

	std::string MCMCFileNameRoot;

	InputReader* reader;

	double acceptanceTuningParameter;

	Eigen::LLT<Eigen::MatrixXf> CholDecom;

	double acceptanceRatioTarget;

	vector<vector<double> > MCMCResults_BetaVectors;

	void adaptiveKernel(int numberIterations, double alpha);

	vector<double> MCMCResults_SigmaSquared;

	vector<double> BetaValues;

	vector<vector<bsccs::real> > hessian;

	int J;

	int maxIterations;

	int nBetaSamples;

	int nSigmaSquaredSamples;
};
}


#endif /* MCMCDRIVER_H_ */
