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
	maxIterations = 1;
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

	Parameter Beta_Hat(ccd.hBeta, J);
	Parameter Beta(ccd.hBeta, J);
	Beta.store();
	bsccs::real sigma2Start;
	sigma2Start = (bsccs::real) ccd.sigma2Beta;
	Parameter SigmaSquared(&sigma2Start, 1);

	initializeHessian();
	ccd.getHessian(&hessian_notGSL);
	generateCholesky();  //Inverts the cholesky too

	// Generate the tools for the MH loop
	IndependenceSampler sampler;
	// need randomizer

	//Framework for storing MCMC data
	vector<vector<bsccs::real> > MCMCBetaValues;
	vector<double> MCMCSigmaSquaredValues;

	for (int iterations = 0; iterations < maxIterations; iterations ++) {

		//Select a sample beta vector
		sampler.sample(&Beta_Hat, &Beta, Cholesky_notGSL);

		//Compute the acceptance ratio, and decide if accept or reject beta sample

		MHRatio MHstep;
		MHstep.evaluate(&Beta, &SigmaSquared, ccd, &hessian_notGSL, precisionDeterminant);

		if (Beta.getChangeStatus()) {
			// TODO Make storage functionality
			nBetaSamples ++;
		}

		if (SigmaSquared.getNeedToChangeStatus()) {
			SigmaSampler sigmaMaker;
			sigmaMaker.sampleSigma(&SigmaSquared);
			nSigmaSquaredSamples ++;

			// Need Wrapper for this....
			ccd.resetBeta();
			ccd.setHyperprior(SigmaSquared.get(0));
			int ZHANG_OLES = 1;
			int ccdIterations = 100;
			double tolerance = 5E-4;
			ccd.update(ccdIterations, ZHANG_OLES, tolerance);
		}

		/*
		if (acceptanceRatio.acceptBetaBool(ccd, &sampler, uniformRandom, betaSize, gslBetaStart, sample, precisionInverse)){
			// Accept the sampled beta
			nBetaSamples++;  //record sample size
			betaValuesSampled_gsl.push_back(sample);

			if (acceptanceRatio.getSigmaSquaredBool(uniformRandom)){
				nSigmaSquaredSamples ++;
				SigmaSampler getSigma;
				double alpha = 0.5;
				double beta = 0.5;
				double sigmaSquared = getSigma.sampleSigma(sample, betaSize, alpha, beta, r);
				cout << "Sigma squared = " << sigmaSquared << endl;

				ccd.resetBeta();
				ccd.setHyperprior(sigmaSquared);

				int ZHANG_OLES = 1;
				int ccdIterations = 100;
				double tolerance = 5E-4;
				ccd.update(ccdIterations, ZHANG_OLES, tolerance);
				for (int i = 0; i < betaSize; i++) {
					//betaValuesStart.push_back(ccd.getBeta(i));
					gsl_vector_set(gslBetaStart, i, ccd.getBeta(i));
				}
			}

		} else {
			//do nothing
		}
*/
	}

	//CredibleIntervals intervalsToReport;
	//credibleIntervals = intervalsToReport.computeCredibleIntervals(betaValuesSampled_gsl, betaSize, nBetaSamples);
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

void MCMCDriver::logResults(const CCDArguments& arguments, std::string conditionId) {

	ofstream outLog(arguments.outFileName.c_str());
	if (!outLog) {
		cerr << "Unable to open log file: " << arguments.bsFileName << endl;
		exit(-1);
	}

	map<int, DrugIdType> drugMap = reader->getDrugNameMap();

	string sep(","); // TODO Make option

	if (!arguments.reportRawEstimates) {
		outLog << "Drug_concept_id" << sep << "Condition_concept_id" << sep <<
				"score" << sep << "lower" << sep <<
				"upper" << endl;
	}

	for (int j = 0; j < J; ++j) {
		outLog << drugMap[j] << sep << conditionId << sep;

		outLog << BetaValues[j] << sep << credibleIntervals[0][j] << sep << credibleIntervals[1][j] << endl;

	}


	outLog.close();
}


}
