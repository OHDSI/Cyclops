/*
 * MCMCDriver.cpp
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */


#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "MCMCDriver.h"
#include "MHRatio.h"
#include "IndependenceSampler.h"
#include "SigmaSampler.h"
#include "CredibleIntervals.h"
#include "Parameter.h"
#include "ModelLikelihood.h"

#include <boost/random.hpp>

//#define Debug_TRS
//#define DEBUG_STATE

namespace bsccs {


MCMCDriver::MCMCDriver(InputReader * inReader, std::string MCMCFileName): reader(inReader) {
	MCMCFileNameRoot = MCMCFileName;
	maxIterations = 1000;
	nBetaSamples = 0;
	nSigmaSquaredSamples = 0;
	acceptanceTuningParameter = 3; // exp(acceptanceTuningParameter) modifies
	acceptanceRatioTarget = 0.30;
	autoAdapt = false;
}

void MCMCDriver::initializeHessian() {

	for (int i = 0; i < J; i ++){
		vector<bsccs::real> columnInHessian(J,0);
		hessian.push_back(columnInHessian);
	}
}

void MCMCDriver::clearHessian() {

	for (int i = 0; i < J; i ++){
		for (int j = 0; j < J; j++) {
			hessian[j][i] = 0;
		}
	}
}

MCMCDriver::~MCMCDriver() {

}

vector<double> storedBetaHat;

void checkValidState(CyclicCoordinateDescent& ccd, MHRatio& MHstep, Parameter& Beta,
		Parameter& Beta_Hat,
		Parameter& SigmaSquared) {
	ccd.setBeta(Beta.returnCurrentValues());
	double logLike = ccd.getLogLikelihood();
	double storedLogLike =  MHstep.getStoredLogLikelihood();
	if (logLike != storedLogLike) {
		cerr << "Error in internal state of beta/log_likelihood." << endl;
		cerr << "\tStored value: " << storedLogLike << endl;
		cerr << "\tRecomp value: " << logLike << endl;
		exit(-1);
	} else {
		cerr << "All fine" << endl;
	}

	if (storedBetaHat.size() == 0) { // first time through
		for (int i = 0; i < Beta_Hat.getSize(); ++i) {
			storedBetaHat.push_back(Beta_Hat.get(i));
		}

	} else {
		for (int i = 0; i < Beta_Hat.getSize(); ++i) {
			if (storedBetaHat[i] != Beta_Hat.get(i)) {
				cerr << "Beta hat has changed!" << endl;
				exit(-1);
			}
		}
	}

	// TODO Check internals with sigma
}

void MCMCDriver::initialize(CyclicCoordinateDescent& ccd, Parameter& Beta_Hat, Parameter& Beta, Parameter& SigmaSquared) {
	// MAS All initialization

	// Beta_Hat = modes from ccd
	J = ccd.getBetaSize();
	Beta_Hat.initialize(ccd.hBeta, J);

	// Set up Beta
	Beta.initialize(ccd.hBeta, J);
	//Beta.setProbabilityUpdate(betaAmount);
	Beta.store();

	// Set up Sigma squared
	bsccs::real sigma2Start;
	sigma2Start = (bsccs::real) ccd.sigma2Beta;
	SigmaSquared.initialize(&sigma2Start, 1);
	SigmaSquared.logParameter();

	ccd.setUpHessianComponents();
	initializeHessian();

	ccd.computeXBeta_GPU_TRS_initialize();

	clearHessian();
	ccd.getHessian(&hessian);
	generateCholesky();

}

void MCMCDriver::logState(Parameter& Beta, Parameter& SigmaSquared){
	MCMCResults_SigmaSquared.push_back(SigmaSquared.returnCurrentValues()[0]);
	MCMCResults_BetaVectors.push_back(Beta.returnCurrentValues());
}

void MCMCDriver::drive(
		CyclicCoordinateDescent& ccd, double betaAmount, long int seed) {


	int getBeta = 0;
	int getSigma = 0;
	int numberAcceptances = 0;
	vector<double> transitionKernelSelectionProb;
	transitionKernelSelectionProb.push_back(betaAmount);
	transitionKernelSelectionProb.push_back(1-betaAmount);

	vector<TransitionKernel> transitionKernels;
	IndependenceSampler independenceSamplerInstance;
	SigmaSampler sigmaSamplerInstance;
	transitionKernels.push_back(independenceSamplerInstance);
	transitionKernels.push_back(sigmaSamplerInstance);

	Parameter Beta_Hat;
	Parameter Beta;
	Parameter SigmaSquared;
	initialize(ccd, Beta_Hat, Beta, SigmaSquared);
	logState(Beta, SigmaSquared);


	//Set Boost rng
	boost::mt19937 rng(seed);


	MHRatio MHstep(ccd);

	double alpha;

	//MCMC Loop
	for (int iterations = 0; iterations < maxIterations; iterations ++) {

		cout << endl << "iteration " << iterations << endl;

#ifdef DEBUG_STATE
		checkValidState(ccd, MHstep, Beta, Beta_Hat, SigmaSquared);
#endif

//		cerr << "Yo!" << endl;
//		exit(-1);

		// Store values
		Beta.store();
		SigmaSquared.store();

		static boost::uniform_01<boost::mt19937> zeroone(rng);

		// Sample from a uniform distribution
		double uniformRandom = zeroone();


		//Select a sample beta vector
		if (betaAmount > uniformRandom) {
			getBeta ++;

//			modifyHessianWithTuning(acceptanceTuningParameter);  // Tuning parameter to one place only
//
//			cout << "Printing Hessian in drive" << endl;
//
//			for (int i = 0; i < J; i ++) {
//				cout << "[";
//					for (int j = 0; j < J; j++) {
//						cout << HessianMatrixTuned(i,j) << ", ";
//					}
//				cout << "]" << endl;
//			}

//			cout << HessianMatrix << endl;
//			cout << (CholDecom.matrixU()) << endl;
//			Beta_Hat.logParameter();
//			exit(-1);

			independenceSamplerInstance.sample(&Beta_Hat, &Beta, rng, CholDecom, acceptanceTuningParameter);

//			Beta.logParameter();
//			exit(-1);


			cout << "acceptanceTuningParameter = " <<  acceptanceTuningParameter << endl;

			//Compute the acceptance ratio
			alpha = MHstep.evaluate(&Beta, &Beta_Hat, &SigmaSquared, ccd, rng,
					HessianMatrix, acceptanceTuningParameter);
			cout << "alpha = " << alpha << endl;

			MCMCResults_BetaVectors.push_back(Beta.returnCurrentValues());
			nBetaSamples ++;

			if (Beta.getChangeStatus()){
				numberAcceptances ++;
			}

			if (autoAdapt) {
				adaptiveKernel(iterations,alpha);
			}

		}

		if (Beta.getProbabilityUpdate() < uniformRandom) {
			getSigma ++;
			SigmaSampler sigmaMaker;
			sigmaMaker.sampleSigma(&SigmaSquared, &Beta, rng);

			MCMCResults_SigmaSquared.push_back(SigmaSquared.returnCurrentValues()[0]);
			nSigmaSquaredSamples ++;

			// TODO Need Wrapper for this....
			ccd.resetBeta();
			ccd.setHyperprior(SigmaSquared.get(0));
			int ZHANG_OLES = 1;
			int ccdIterations = 100;
			double tolerance = 5E-4;

			ccd.update(ccdIterations, ZHANG_OLES, tolerance);
			clearHessian();
			ccd.getHessian(&hessian);
			generateCholesky();
			Beta_Hat.set(ccd.hBeta);
		}


//#ifdef DEBUG_STATE
//		checkValidState(ccd, MHstep, Beta, Beta_Hat, SigmaSquared);
		cerr << "acceptance rate: " << ( static_cast<double>(numberAcceptances)
				/ static_cast<double>(iterations)) << endl;
//#endif

		// End MCMC loop
	}

		cout << "getBeta = " << getBeta << endl;
		cout << "getSigma = " << getSigma << endl;
		cout << "number of acceptances = " << numberAcceptances << endl;
		cout << "Starting Credible Intervals" << endl;

		cout << "at End, nBetaSamples = " << nBetaSamples << endl;
		cout << "at End, nSigmaSquaredSamples = " << nSigmaSquaredSamples << endl;

		CredibleIntervals intervalsToReport;
		intervalsToReport.computeCredibleIntervals(&MCMCResults_BetaVectors, &MCMCResults_SigmaSquared, Beta.getProbabilityUpdate(), SigmaSquared.getProbabilityUpdate(), MCMCFileNameRoot);


}

double coolingTransform(int x) {
//	return std::log(x);
	return std::sqrt(x);
//	return static_cast<double>(x);
}

double targetTransform(double alpha, double target) {
	return (alpha - target);
}

void MCMCDriver::adaptiveKernel(int numberIterations, double alpha) {

	acceptanceTuningParameter = acceptanceTuningParameter +
			(1.0 / (1.0 + coolingTransform(numberIterations))) *
			targetTransform(alpha, acceptanceRatioTarget);
//			(0.4 - std::abs(alpha - acceptanceRatioTarget));

//	double delta;
//	if (alpha < 0.2 || alpha > 0.8) {
//		delta -= 1.0;
//	} else {
//		delta += 1.0;
//	}
//	acceptanceTuningParameter += (1.0 / (1.0 + coolingTransform(numberIterations))) * delta;
}

void MCMCDriver::generateCholesky() {
	HessianMatrix.resize(J, J);
	HessianMatrixTuned.resize(J,J);

	//Convert to Eigen for Cholesky decomposition
	for (int i = 0; i < J; i++) {
		for (int j = 0; j < J; j++) {
			HessianMatrix(i, j) = -hessian[i][j];
			// TODO Debugging here
//			if (i == j) {
//				HessianMatrix(i,j) = 1.0;
//			} else {
//				HessianMatrix(i,j) = 0.0;
//			}

		}
	}

	// Initial tuned precision matrix is the same as the CCD Hessian
	HessianMatrixTuned = HessianMatrix;

	//Perform Cholesky Decomposition
	CholDecom.compute(HessianMatrix);

#ifdef Debug_TRS
		cout << "Printing Hessian in generateCholesky" << endl;

		for (int i = 0; i < J; i ++) {
			cout << "[";
				for (int j = 0; j < J; j++) {
					cout << HessianMatrix(i,j) << ", ";
				}
			cout << "]" << endl;
			}
#endif


}


double getTransformedTuningValue(double tuningParameter) {
	return exp(-tuningParameter);
}


/* Modifies the Hessian with the tuning parameter
 * and calculates the new Cholesky. Cholesky currently
 * recomputed. Dividing starting Cholesky by sqrt(transformed tuning parameter)
 * should be better.
 */
void MCMCDriver::modifyHessianWithTuning(double tuningParameter){
	HessianMatrixTuned = HessianMatrix/getTransformedTuningValue(tuningParameter);  // Divide - working in precision space
	//Perform Cholesky Decomposition
	CholDecom.compute(HessianMatrixTuned);  // Expensive step, will optimize once check accuracy


#ifdef Debug_TRS
	cout << "Printing Hessian in modifyHessianWithTuning" << endl;

	for (int i = 0; i < J; i ++) {
		cout << "[";
			for (int j = 0; j < J; j++) {
				cout << HessianMatrixTuned(i,j) << ", ";
			}
		cout << "]" << endl;
	}

	Eigen::MatrixXf CholeskyDecompL(J, J);
	CholeskyDecompL = CholDecom.matrixL();

	cout << "Printing Cholesky in modifyHessianWithTuning" << endl;

	for (int i = 0; i < J; i ++) {
		cout << "[";
			for (int j = 0; j < J; j++) {
				cout << CholeskyDecompL(i,j) << ", ";
			}
		cout << "]" << endl;
	}
#endif


}

}
