/*
 * MHRatio.cpp
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>

#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "MHRatio.h"
#include "Eigen/core"

namespace bsccs{

MHRatio::MHRatio(){
	fudgeFactor = 0.1;
}

MHRatio::~MHRatio(){

}

bool MHRatio::acceptBetaBool(CyclicCoordinateDescent & ccd, IndependenceSampler * sampler, double uniformRandom, int betaSize, gsl_vector * betaOld, gsl_vector * betaToEvaluate, gsl_matrix * covarianceMatrix){


	double ratio = 0;
	double fBetaPossible = 1;
	double fBetaCurrent = 1;
	double pBetaPossible = 1;
	double pBetaCurrent = 1;
	double mvtBetaPossible = 1;
	double mvtBetaCurrent = 1;

	vector<double> betaPossible;
	vector<double> betaOldValues;

	for (int i = 0; i < betaSize; i++) {
		betaPossible.push_back(gsl_vector_get(betaToEvaluate,i));
		betaOldValues.push_back(gsl_vector_get(betaOld,i));
	}

	ccd.setBeta(betaPossible);
	fBetaPossible = exp(ccd.getLogLikelihood());
	pBetaPossible = exp(ccd.getLogPrior());
	mvtBetaPossible = sampler->dmvnorm(betaSize, betaToEvaluate, betaOld, covarianceMatrix);

	//These will be passed in eventually

	ccd.setBeta(betaOldValues);
	fBetaCurrent = exp(ccd.getLogLikelihood());
	pBetaCurrent = exp(ccd.getLogPrior());
	mvtBetaCurrent = sampler->dmvnorm(betaSize, betaOld, betaOld, covarianceMatrix);


	ratio = ((fBetaPossible*pBetaPossible) / mvtBetaPossible) / ((fBetaCurrent*pBetaCurrent) / mvtBetaCurrent);



	alpha = min(ratio, 1.0);

	cout << "ratio = " << ratio << " and uniformRandom = " << uniformRandom << endl;

	if (alpha > uniformRandom) {
		return true;
	} else{
		return true;
	}
}

bool MHRatio::getSigmaSquaredBool(double uniformRandom) {

	cout << "in getSigmaSquaredBool fudgeFactor*alpha = " << fudgeFactor*alpha << endl;
	cout << "in getSigmaSquaredBool uniformRandom = " << uniformRandom<< endl;
	if (fudgeFactor*alpha > uniformRandom) {
		cout << "CHANGING SIGMA SQUARED" << endl;
		return false;
	} else {
		return false;
	}
}


double MHRatio::min(double value1, double value2) {
	if (value1 > value2) {
		return value2;
	} else {
		return value1;
	}

}

}

