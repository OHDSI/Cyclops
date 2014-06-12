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

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#define Debug_TRS

namespace bsccs{

MHRatio::MHRatio(){}

void MHRatio::initialize(CyclicCoordinateDescent & ccd){
	storedFBetaCurrent = ccd.getLogLikelihood();
	storedPBetaCurrent = ccd.getLogPrior();

}

MHRatio::~MHRatio(){

}


bool MHRatio::evaluate(MCMCModel & model) {

	cout << "MHRatio::evaluate" << endl;
	double logMetropolisRatio = getLogMetropolisRatio(model);
	double logHastingsRatio;

	cout << "M part done" << endl;

	if (model.getUseHastingsRatio()){
		logHastingsRatio = getLogHastingsRatio(model);
	} else {
		//cout << "+++++++++++++++++++++++++++++++   Log Hastings 0   ----------------------" << endl;
		logHastingsRatio = 0;
	}

// Compute the ratio for the MH step
	double logRatio = logMetropolisRatio + logHastingsRatio;

	cout << "H part done" << endl;
//Check for numerical issues
	if (std::isfinite(logMetropolisRatio) && std::isfinite(logHastingsRatio)){// && std::isfinite(ratio)){
	} else {
		cout << "########--------------#########  Warning: Numerical Issues   ########-------#######" << endl;
	}
// Set our alpha

	alpha = min(logRatio, 0.0);

// Sample from a uniform distribution
	double uniformRandom = rand() / ((double) RAND_MAX);
	double logUniformRandom = log(uniformRandom);

	bool returnValue;
	if (alpha > logUniformRandom) {
		//model.setLogLikelihood(fBetaPossible);
		returnValue = true;
	} else{
		returnValue = false;
	}

#ifdef Debug_TRS
                cout << "alpha = " << alpha << endl;
                cout << "logRatio = " << logRatio << endl;
                cout << "uniformRandom = " << uniformRandom << endl;
                cout << "logUniformRandom = " << logUniformRandom << endl;
                cout << "logMetropolisRatio = " << logMetropolisRatio << endl;
                cout << "logHastingsRatio = " << logHastingsRatio << endl;

#endif

	return(returnValue);


}

double MHRatio::getTransformedTuningValue(double tuningParameter){
	// TODO Don't forward reference like this.
	return exp(-tuningParameter);
}




double MHRatio::getLogMetropolisRatio(MCMCModel & model){
	cout << "MHRatio::getLogMetropolisRatio" << endl;
	// Get the proposed Beta values

	// Compute log Likelihood and log prior

	CyclicCoordinateDescent & ccd = model.getCCD();

	ccd.resetBeta();

	for (int i = 0; i < model.getBeta().getSize(); i++){
		ccd.setBeta(i, model.getBeta().get(i));
	}

	model.setLogLikelihood(ccd.getLogLikelihood());
	model.setLogPrior(ccd.getLogPrior());

#ifdef Debug_TRS
  cout << "proposed Likelihood = " << model.getLogLikelihood() << endl;
  cout << "stored Likelihood = " << model.getStoredLogLikelihood() << endl;
  cout << "proposed prior = " << model.getLogPrior() << endl;
  cout << "stored prior = " << model.getStoredLogPrior() << endl;
  #endif


	double ratio = (model.getLogLikelihood() + model.getLogPrior()) - (model.getStoredLogLikelihood() + model.getStoredLogPrior());

	return(ratio);
}

double MHRatio::getLogHastingsRatio(MCMCModel & model){


	int betaLength = (model.getBeta()).getSize();
	Eigen::VectorXf betaCurrent(betaLength);
	Eigen::VectorXf betaProposal(betaLength);

	Eigen::VectorXf beta_hat(betaLength);

	Eigen::VectorXf betaHat_minus_current(betaLength);
	Eigen::VectorXf betaHat_minus_proposal(betaLength);

	Eigen::VectorXf precisionDifferenceProduct_current(betaLength);
	Eigen::VectorXf precisionDifferenceProduct_proposal(betaLength);


	for (int i = 0; i< betaLength; i++){
		betaProposal(i) = (model.getBeta()).get(i);
		betaCurrent(i) = (model.getBeta()).getStored(i);
		beta_hat(i) = (model.getBeta_Hat()).get(i);
	}

	betaHat_minus_current = beta_hat - betaCurrent;
	betaHat_minus_proposal = beta_hat - betaProposal;

	precisionDifferenceProduct_current =  model.getHessian() * betaHat_minus_current;
	precisionDifferenceProduct_proposal = model.getHessian() * betaHat_minus_proposal;

	double numerator = betaHat_minus_current.dot(precisionDifferenceProduct_current);
	double denominator = betaHat_minus_proposal.dot(precisionDifferenceProduct_proposal);

	return(-0.5*(numerator - denominator) / getTransformedTuningValue(model.getTuningParameter())); // log scale
	// NB: tuningParameter scales the variance
	//return(0);

	// TODO Check these numbers!
}


double MHRatio::min(double value1, double value2) {
	if (value1 > value2) {
		return value2;
	} else {
		return value1;
	}

}

}

