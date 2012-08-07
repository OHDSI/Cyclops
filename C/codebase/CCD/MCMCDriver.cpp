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

namespace bsccs {


MCMCDriver::MCMCDriver() {
	maxIterations = 10;
	sampleSize = 0;
}

MCMCDriver::~MCMCDriver() {

}

void MCMCDriver::drive(
		CyclicCoordinateDescent& ccd) {

	// Select First Beta vector = modes from ccd
	int betaSize = ccd.getBetaSize();
	//vector<bsccs::real> betaValuesStart;

	gsl_vector * gslBetaStart = gsl_vector_alloc(betaSize);

	for (int i = 0; i < betaSize; i++) {
		//betaValuesStart.push_back(ccd.getBeta(i));
		gsl_vector_set(gslBetaStart, i, ccd.getBeta(i));
	}

	// Generate an independent sample
	IndependenceSampler sampler;

	gsl_rng * randomizer = gsl_rng_alloc(gsl_rng_taus);

	//vector< vector<bsccs::real> > betaValuesSampled;
	//betaValuesSampled.push_back(betaValuesStart);

	vector<gsl_vector*> betaValuesSampled_gsl;
	betaValuesSampled_gsl.push_back(gslBetaStart);
	sampleSize ++;

	gsl_matrix * precisionMatrix = gsl_matrix_alloc(betaSize, betaSize);
	ccd.getHessianForCholesky_GSL(precisionMatrix);
	cout << gsl_matrix_get(precisionMatrix, 0,0) << endl;
	gsl_linalg_cholesky_decomp(precisionMatrix); // get Cholesky decomposition

	cout << "Precision Cholesky" << endl;
	for (int i = 0; i < betaSize; i++) {
			cout << "[";
			for (int j = 0; j< betaSize; j++) {
				cout << gsl_matrix_get(precisionMatrix, i, j) << ",";
			}
			cout << "]" << endl;
		}

	gsl_linalg_cholesky_invert(precisionMatrix);

	cout << "Precision Cholesky inverse" << endl;
		for (int i = 0; i < betaSize; i++) {
				cout << "[";
				for (int j = 0; j< betaSize; j++) {
					cout << gsl_matrix_get(precisionMatrix, i, j) << ",";
				}
				cout << "]" << endl;
			}

	gsl_matrix * precisionInverse = gsl_matrix_alloc(betaSize, betaSize);
	gsl_matrix_set_zero(precisionInverse);

	gsl_matrix_add(precisionInverse, precisionMatrix);
	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

	for (int iterations = 0; iterations < maxIterations; iterations ++) {

		double uniformRandom = gsl_rng_uniform(r);
		gsl_vector * sample = gsl_vector_alloc(betaSize);
		sampler.rmvnorm_stl(randomizer, betaSize, gslBetaStart, precisionInverse, sample);

		cout << "<";
		for (int j = 0; j < betaSize; j++) {
			cout << gsl_vector_get(sample, j) << ",";
		}
		cout << ">" << endl;

		MHRatio acceptanceRatio;

		if (acceptanceRatio.acceptBool(ccd, &sampler, uniformRandom, betaSize, gslBetaStart, sample, precisionInverse)){
			cout << "ADDING sample" << endl;
			betaValuesSampled_gsl.push_back(sample);
		} else {
			cout << "NOT ADDING" << endl;
		}
	}


}
}
