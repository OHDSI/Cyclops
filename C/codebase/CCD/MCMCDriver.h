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
#include "Parameter.h"
#include "Model.h"
#include "TransitionKernel.h"

namespace bsccs {
class MCMCDriver {
public:
	MCMCDriver(InputReader * inReader, std::string MCMCFileName);

	virtual ~MCMCDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd, double betaAmount, long int seed);


	void initialize(double betaAmount, Model & model, CyclicCoordinateDescent& ccd);

	void logState(Model & model);

	double targetTransform(double alpha, double target);

	double coolingTransform(int x);

	double getTransformedTuningValue(double tuningParameter);

private:

	std::string MCMCFileNameRoot;

	InputReader* reader;

	double acceptanceTuningParameter;

	double acceptanceRatioTarget;

	vector<vector<double> > MCMCResults_BetaVectors;
	vector<double> MCMCResults_loglikelihoods;


	void adaptiveKernel(int numberIterations, double alpha);

	vector<double> transitionKernelSelectionProb;
	vector<TransitionKernel*> transitionKernels;

	int findTransitionKernelIndex(double uniformRandom, vector<double>& transitionKernelSelectionProb);

	vector<double> MCMCResults_SigmaSquared;

	vector<double> BetaValues;

	int J;

	int maxIterations;

	int nBetaSamples;

	int nSigmaSquaredSamples;

	bool autoAdapt;
};
}


#endif /* MCMCDRIVER_H_ */
