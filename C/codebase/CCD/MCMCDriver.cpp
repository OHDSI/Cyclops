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

#define Debug_TRS

namespace bsccs {


MCMCDriver::MCMCDriver(InputReader * inReader): reader(inReader) {
	maxIterations = 10000;
	nBetaSamples = 0;
	nSigmaSquaredSamples = 0;
	acceptanceTuningParameter = 0; // exp(acceptanceTuningParameter) modifies the hessian
	acceptanceRatioTarget = 0.25;
}

void MCMCDriver::initializeHessian() {

	for (int i = 0; i < J; i ++){
		vector<bsccs::real> columnInHessian(J,0);
		hessian.push_back(columnInHessian);
	}
}

MCMCDriver::~MCMCDriver() {

}

void MCMCDriver::drive(
		CyclicCoordinateDescent& ccd) {

	// Select First Beta vector = modes from ccd
	J = ccd.getBetaSize();

	Parameter Beta_Hat(ccd.hBeta, J);
	Beta_Hat.logParameter();

	Parameter Beta(ccd.hBeta, J);
	Beta.logParameter();

	Beta.store();
	bsccs::real sigma2Start;
	sigma2Start = (bsccs::real) ccd.sigma2Beta;
	Parameter SigmaSquared(&sigma2Start, 1);
	SigmaSquared.logParameter();

	initializeHessian();
	ccd.getHessian(&hessian);

	//ccd.computeXBeta_GPU_TRS_initialize();

	generateCholesky();  //Inverts the cholesky too

	// Generate the tools for the MH loop
	IndependenceSampler sampler;

	//Set Boost rng
	boost::mt19937 rng;

	int numberAcceptances = 0;

	//MCMC Loop
	for (int iterations = 0; iterations < maxIterations; iterations ++) {

		cout << endl;
		cout << "New Loop" << endl;
		//Select a sample beta vector
		sampler.sample(&Beta_Hat, &Beta, cholesky, rng, acceptanceTuningParameter);

#ifdef Debug_TRS
		cout << "Sample Draw: ";
		Beta.logParameter();
#endif

		//Compute the acceptance ratio, and decide if Beta and sigma should be changed
		MHRatio MHstep;
		MHstep.evaluate(&Beta, &SigmaSquared, ccd, &hessian);


		MCMCResults_BetaVectors.push_back(Beta.returnCurrentValues());
		nBetaSamples ++;

		if (Beta.getChangeStatus()){
			numberAcceptances ++;
		}

#ifdef Debug_TRS
			cout << "nBetaSamples = " << nBetaSamples << endl;
#endif

		if (SigmaSquared.getNeedToChangeStatus()) {
			SigmaSampler sigmaMaker;
			sigmaMaker.sampleSigma(&SigmaSquared, &Beta, rng);

			MCMCResults_SigmaSquared.push_back(SigmaSquared.returnCurrentValues()[0]);
			nSigmaSquaredSamples ++;
#ifdef Debug_TRS
			cout << "nSigmaSquaredSamples = " << nSigmaSquaredSamples << endl;
			cout << "new SigmaSquared: ";
			SigmaSquared.logParameter();
#endif

			// TODO Need Wrapper for this....
			ccd.resetBeta();
			ccd.setHyperprior(SigmaSquared.get(0));
			int ZHANG_OLES = 1;
			int ccdIterations = 100;
			double tolerance = 5E-4;

			ccd.update(ccdIterations, ZHANG_OLES, tolerance);
			initializeHessian();
			ccd.getHessian(&hessian);
			generateCholesky();
			Beta_Hat.set(ccd.hBeta);

#ifdef Debug_TRS
			Beta_Hat.logParameter();
#endif
		}

	//	Beta_Hat.logParameter();
	//	cout << "This is Beta_Hat" << endl;
		adaptiveKernel(iterations+1, numberAcceptances);
//		cout << endl;
		cout << "numberAcceptances = " << numberAcceptances << endl;
		cout << "iterations = " << iterations << endl;
		cout << "Acceptance Ratio is " << (double) numberAcceptances/(iterations+1) << endl;
		cout << "AcceptanceTuningParameter is " << acceptanceTuningParameter << endl;
	//	Beta_Hat.logParameter();
		cout << "That was Beta_Hat" << endl;

		// End MCMC loop
	}

	cout << "Starting Credible Intervals" << endl;
	if (nBetaSamples > 0 && nSigmaSquaredSamples > 0) {
		cout << "at End, nBetaSamples = " << nBetaSamples << endl;
		cout << "at End, nSigmaSquaredSamples = " << nSigmaSquaredSamples << endl;
		cout << "number of acceptances = " << numberAcceptances << endl;
		CredibleIntervals intervalsToReport;
		intervalsToReport.computeCredibleIntervals(&MCMCResults_BetaVectors, &MCMCResults_SigmaSquared);
	} else {
		cout << "No MCMC data" << endl;
	}

}

void MCMCDriver::adaptiveKernel(int numberIterations, int numberAcceptances) {
	acceptanceTuningParameter = acceptanceTuningParameter + (1/(1+sqrt(numberIterations)))*((double) numberAcceptances/numberIterations - acceptanceRatioTarget);
}

void MCMCDriver::generateCholesky() {
	Eigen::MatrixXf HessianMatrix(J, J);
	Eigen::MatrixXf CholeskyDecompL(J, J);

	// initialize Cholesky matrix
	for (int i = 0; i < J; i ++){
		vector<bsccs::real> columnInCholesky(J,0);
		cholesky.push_back(columnInCholesky);
	}

	//Convert to Eigen for Cholesky decomposition
	for (int i = 0; i < J; i++) {
		for (int j = 0; j < J; j++) {
			HessianMatrix(i, j) = hessian[i][j];
		}
	}

	//Perform Cholesky Decomposition
	Eigen::LLT<Eigen::MatrixXf> CholDecom(HessianMatrix);

	Eigen::VectorXf b = Eigen::VectorXf::Random(J);
	CholeskyDecompL = CholDecom.matrixL();
	Eigen::MatrixXf CholeskyInverted = CholeskyDecompL.inverse();


	for (int i = 0; i < J; i ++) {
		for (int j = 0; j < J; j++) {
			cholesky[i][j] = CholeskyInverted(i,j);
		}
	}
}

}
