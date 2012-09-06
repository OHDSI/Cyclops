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

#include "MHRatio.h"


namespace bsccs{

MHRatio::MHRatio(){
	fudgeFactor = 0.1;
}

MHRatio::~MHRatio(){

}

bool MHRatio::evaluate(Parameter * Beta, Parameter * SigmaSquared, CyclicCoordinateDescent & ccd, vector<vector<bsccs::real> > * precisionMatrix, bsccs::real precisionDeterminant) {//CyclicCoordinateDescent & ccd, IndependenceSampler * sampler, double uniformRandom, int betaSize, gsl_vector * betaOld, gsl_vector * betaToEvaluate, gsl_matrix * covarianceMatrix){


	double ratio = 0;
	double fBetaPossible = 1;
	double fBetaCurrent = 1;
	double pBetaPossible = 1;
	double pBetaCurrent = 1;
	double mvtBetaPossible = 1;
	double mvtBetaCurrent = 1;

	int betaSize = Beta->getSize();

	vector<double> betaPossible = Beta->returnCurrentValues();
	vector<double> betaOldValues = Beta->returnStoredValues();

	ccd.setBeta(betaPossible);
	fBetaPossible = exp(ccd.getLogLikelihood());
	pBetaPossible = exp(ccd.getLogPrior());
	mvtBetaPossible = 1; // TODO Implement

	ccd.setBeta(betaOldValues);
	fBetaCurrent = exp(ccd.getLogLikelihood());
	pBetaCurrent = exp(ccd.getLogPrior());
	mvtBetaCurrent = 1; //TODO Implement


	ratio = ((fBetaPossible*pBetaPossible) / mvtBetaPossible) / ((fBetaCurrent*pBetaCurrent) / mvtBetaCurrent);

	alpha = min(ratio, 1.0);

	double uniformRandom = (rand() / (RAND_MAX + 1.0));
	cout << "ratio = " << ratio << " and uniformRandom = " << uniformRandom << endl;

	if (alpha > uniformRandom) {
		Beta->setChangeStatus(true);
		cout << "Change Beta" << endl;
	} else{
		Beta->setChangeStatus(false);
		Beta->restore();
	}

	if (fudgeFactor*alpha > uniformRandom) {
		SigmaSquared->setChangeStatus(false);
		SigmaSquared->setNeedToChangeStatus(true);
		cout << "Need to change some sigmaSquared" << endl;
	} else {
		SigmaSquared->setChangeStatus(false);
		SigmaSquared->setNeedToChangeStatus(false);
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

