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
#include "MCMCDriver.h"
#include "MHRatio.h"
#include "IndependenceSampler.h"
#include "SigmaSampler.h"
#include "CredibleIntervals.h"
#include "Parameter.h"
#include "ModelLikelihood.h"

#include <boost/random.hpp>

//#define Debug_TRS

namespace bsccs {


MCMCDriver::MCMCDriver(InputReader * inReader, std::string MCMCFileName): reader(inReader) {
	MCMCFileNameRoot = MCMCFileName;
	maxIterations = 50000;
	nBetaSamples = 0;
	nSigmaSquaredSamples = 0;
	acceptanceTuningParameter = 1; // exp(acceptanceTuningParameter) modifies
	acceptanceRatioTarget = 0.30;
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

void MCMCDriver::drive(
		CyclicCoordinateDescent& ccd, double betaAmount, long int seed) {

	// Select First Beta vector = modes from ccd
	J = ccd.getBetaSize();
	Parameter Beta_Hat(ccd.hBeta, J);

	Beta_Hat.logParameter();

	// Set up Beta
	Parameter Beta(ccd.hBeta, J);
	Beta.logParameter();
	Beta.setProbabilityUpdate(betaAmount);
	Beta.store();

	// Set up Sigma squared
	bsccs::real sigma2Start;
	sigma2Start = (bsccs::real) ccd.sigma2Beta;
	Parameter SigmaSquared(&sigma2Start, 1);
	SigmaSquared.logParameter();

	ccd.setUpHessianComponents();
	initializeHessian();

	double loglike = ccd.getLogLikelihood();

	ccd.getHessian(&hessian);

	ccd.computeXBeta_GPU_TRS_initialize();

	generateCholesky();


	////////////////////  GPU Test CODE /////////////////
	/*
	ccd.resetBeta();
	vector<double> betaTest;
	for (int i = 0; i < 3; i ++) {
		betaTest.push_back(2.00);
	}

	ccd.setBeta(betaTest);

	double loglike2 = ccd.getLogLikelihood();

	cout << "loglike = " << loglike << endl;
	cout << "loglike2 = " << loglike2 << endl;
*/
	///////////////////                 /////////////////

	// Generate the tools for the MH loop
	IndependenceSampler sampler;

	//Set Boost rng
	boost::mt19937 rng(seed);

	int getBeta = 0;
	int getSigma = 0;
	int numberAcceptances = 0;
	MCMCResults_SigmaSquared.push_back(SigmaSquared.returnCurrentValues()[0]);
	MCMCResults_BetaVectors.push_back(Beta.returnCurrentValues());

	MHRatio MHstep(ccd);

	double alpha;

	//MCMC Loop
	for (int iterations = 0; iterations < maxIterations; iterations ++) {

		static boost::uniform_01<boost::mt19937> zeroone(rng);
		cout << "iterations = " << iterations << endl;
		// Sample from a uniform distribution
		double uniformRandom = zeroone();


		//Select a sample beta vector
		if (Beta.getProbabilityUpdate() > uniformRandom) {
			getBeta ++;

			modifyHessianWithTuning(acceptanceTuningParameter);  // Tuning parameter to one place only

			cout << "Printing Hessian in drive" << endl;

			for (int i = 0; i < J; i ++) {
				cout << "[";
					for (int j = 0; j < J; j++) {
						cout << HessianMatrixTuned(i,j) << ", ";
					}
				cout << "]" << endl;
			}
			sampler.sample(&Beta_Hat, &Beta, rng, CholDecom);
			cout << "acceptanceTuningParameter = " <<  acceptanceTuningParameter << endl;

			//Compute the acceptance ratio
			alpha = MHstep.evaluate(&Beta, &Beta_Hat, &SigmaSquared, ccd, rng, HessianMatrixTuned);
			cout << "alpha = " << alpha << endl;

			MCMCResults_BetaVectors.push_back(Beta.returnCurrentValues());
			nBetaSamples ++;

			if (Beta.getChangeStatus()){
				numberAcceptances ++;
			}

			//adaptiveKernel(iterations,alpha);

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

void MCMCDriver::adaptiveKernel(int numberIterations, double alpha) {
	acceptanceTuningParameter = acceptanceTuningParameter + (1/(1+sqrt(numberIterations)))*(alpha -
			acceptanceRatioTarget);
}

void MCMCDriver::generateCholesky() {
	HessianMatrix.resize(J, J);
	HessianMatrixTuned.resize(J,J);

	//Convert to Eigen for Cholesky decomposition
	for (int i = 0; i < J; i++) {
		for (int j = 0; j < J; j++) {
			HessianMatrix(i, j) = -hessian[i][j];
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
	return exp(tuningParameter);
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
