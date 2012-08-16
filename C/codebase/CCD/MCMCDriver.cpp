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
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>

#include "MCMCDriver.h"
#include "MHRatio.h"
#include "IndependenceSampler.h"
#include "SigmaSampler.h"
#include "CredibleIntervals.h"
#include "Parameter.h"
#include "ModelLikelihood.h"


namespace bsccs {


MCMCDriver::MCMCDriver(InputReader * inReader): reader(inReader) {
	maxIterations = 1000;
	nBetaSamples = 0;
	nSigmaSquaredSamples = 0;
}

MCMCDriver::~MCMCDriver() {

}

void MCMCDriver::drive(
		CyclicCoordinateDescent& ccd) {

	// Select First Beta vector = modes from ccd
	int betaSize = ccd.getBetaSize();

	J = ccd.getBetaSize();

	gsl_vector * gslBetaStart = gsl_vector_alloc(betaSize);

	for (int i = 0; i < betaSize; i++) {
		gsl_vector_set(gslBetaStart, i, ccd.getBeta(i));
		BetaValues.push_back(ccd.getBeta(i));
	}

	// Generate the tools for the MH loop
	IndependenceSampler sampler;
	gsl_rng * randomizer = gsl_rng_alloc(gsl_rng_taus);

	vector<gsl_vector*> betaValuesSampled_gsl;
	betaValuesSampled_gsl.push_back(gslBetaStart);
	nBetaSamples ++;

	gsl_matrix * precisionMatrix = gsl_matrix_alloc(betaSize, betaSize);
	ccd.getHessianForCholesky_GSL(precisionMatrix);
	cout << gsl_matrix_get(precisionMatrix, 0,0) << endl;
	gsl_linalg_cholesky_decomp(precisionMatrix); // get Cholesky decomposition (done in place)

	gsl_linalg_cholesky_invert(precisionMatrix); // Inversion of the matrix from which Cholesky was taken (i.e. the Hessian Matrix)

	// Rename the matrix to avoid confusion
	gsl_matrix * precisionInverse = gsl_matrix_alloc(betaSize, betaSize);
	gsl_matrix_set_zero(precisionInverse);
	gsl_matrix_add(precisionInverse, precisionMatrix);


	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

	for (int iterations = 0; iterations < maxIterations; iterations ++) {

		double uniformRandom = gsl_rng_uniform(r);

		//Select a sample beta vector
		gsl_vector * sample = gsl_vector_alloc(betaSize);
		sampler.rmvnorm_stl(randomizer, betaSize, gslBetaStart, precisionInverse, sample);

		//Compute the acceptance ratio, and decide if accept or reject beta sample
		MHRatio acceptanceRatio;

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

	}
	CredibleIntervals intervalsToReport;

	credibleIntervals = intervalsToReport.computeCredibleIntervals(betaValuesSampled_gsl, betaSize, nBetaSamples);

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
