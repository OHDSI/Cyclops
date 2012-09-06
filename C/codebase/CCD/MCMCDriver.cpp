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


namespace bsccs {


MCMCDriver::MCMCDriver(InputReader * inReader): reader(inReader) {
	maxIterations = 10000;
	nBetaSamples = 0;
	nSigmaSquaredSamples = 0;


}

void MCMCDriver::initializeHessian() {

	for (int i = 0; i < J; i ++){
		vector<bsccs::real> columnInHessian(J,0);
		hessian_notGSL.push_back(columnInHessian);
	}
}

MCMCDriver::~MCMCDriver() {

}

void MCMCDriver::drive(
		CyclicCoordinateDescent& ccd) {

	// Select First Beta vector = modes from ccd
	J = ccd.getBetaSize();

	cout << "J = " << J << endl;

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
	ccd.getHessian(&hessian_notGSL);
	generateCholesky();  //Inverts the cholesky too

	// Generate the tools for the MH loop
	IndependenceSampler sampler;
	srand(time(NULL));

	//Framework for storing MCMC data
	vector<vector<bsccs::real> > MCMCBetaValues;
	vector<double> MCMCSigmaSquaredValues;

	//MCMC Loop
	for (int iterations = 0; iterations < maxIterations; iterations ++) {

		//Select a sample beta vector
		sampler.sample(&Beta_Hat, &Beta, Cholesky_notGSL);
		cout << "Sample Draw: ";
		Beta.logParameter();

		//Compute the acceptance ratio, and decide if Beta should be changed
		MHRatio MHstep;
		MHstep.evaluate(&Beta, &SigmaSquared, ccd, &hessian_notGSL, precisionDeterminant);

		if (Beta.getChangeStatus()) {
			MCMCResults_BetaVectors.push_back(Beta.returnCurrentValues());
			nBetaSamples ++;
			cout << "nBetaSamples = " << nBetaSamples << endl;
		}

		if (SigmaSquared.getNeedToChangeStatus()) {
			SigmaSampler sigmaMaker;
			sigmaMaker.sampleSigma(&SigmaSquared);
			MCMCResults_SigmaSquared.push_back(SigmaSquared.returnCurrentValues()[0]);
			nSigmaSquaredSamples ++;
			cout << "nSigmaSquaredSamples = " << nSigmaSquaredSamples << endl;
			cout << "new SigmaSquared: ";
			SigmaSquared.logParameter();

			// Need Wrapper for this....
			ccd.resetBeta();
			ccd.setHyperprior(SigmaSquared.get(0));
			int ZHANG_OLES = 1;
			int ccdIterations = 100;
			double tolerance = 5E-4;
			ccd.update(ccdIterations, ZHANG_OLES, tolerance);
		}
	}

	CredibleIntervals intervalsToReport;
	intervalsToReport.computeCredibleIntervals(&MCMCResults_BetaVectors, &MCMCResults_SigmaSquared);
}

void MCMCDriver::generateCholesky() {
	Eigen::MatrixXf HessianMatrix(J, J);
	Eigen::MatrixXf CholeskyDecompL(J, J);

	// initialize Cholesky matrix
	for (int i = 0; i < J; i ++){
		vector<bsccs::real> columnInCholesky(J,0);
		Cholesky_notGSL.push_back(columnInCholesky);
	}

	//Convert to Eigen for Cholesky decomposition
	for (int i = 0; i < J; i++) {
		for (int j = 0; j < J; j++) {
			HessianMatrix(i, j) = hessian_notGSL[i][j];
		}
	}

	precisionDeterminant = HessianMatrix.determinant();

	//Perform Cholesky Decomposition
	Eigen::LLT<Eigen::MatrixXf> CholDecom(HessianMatrix);



	Eigen::VectorXf b = Eigen::VectorXf::Random(J);
	CholeskyDecompL = CholDecom.matrixL();
	Eigen::MatrixXf CholeskyInverted = CholeskyDecompL.inverse();

	for (int i = 0; i < J; i ++) {
		for (int j = 0; j < J; j++) {
			Cholesky_notGSL[i][j] = CholeskyInverted(i,j);
		}
	}
}

}
