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

namespace bsccs {
class MCMCDriver {
public:
	MCMCDriver(InputReader * inReader, std::string MCMCFileName);

	virtual ~MCMCDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd, double betaAmount, long int seed);

	void generateCholesky();

	void initialize(CyclicCoordinateDescent& ccd, Parameter& Beta_Hat, Parameter& Beta, Parameter& SigmaSquared);

	void logState(Parameter& Beta, Parameter& SigmaSquared);

	void initializeHessian();
	void clearHessian();

	void modifyHessianWithTuning(double tuningParameter);

private:

	std::string MCMCFileNameRoot;

	InputReader* reader;

	double acceptanceTuningParameter;

	Eigen::LLT<Eigen::MatrixXf> CholDecom;

	Eigen::MatrixXf HessianMatrix;

	Eigen::MatrixXf HessianMatrixTuned;


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

	bool autoAdapt;
};
}


#endif /* MCMCDRIVER_H_ */
