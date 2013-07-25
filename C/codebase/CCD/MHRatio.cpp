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

#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#define Debug_TRS

namespace bsccs{

MHRatio::MHRatio(CyclicCoordinateDescent & ccd){
	storedFBetaCurrent = ccd.getLogLikelihood();
	storedPBetaCurrent = ccd.getLogPrior();

}

MHRatio::~MHRatio(){

}

double MHRatio::evaluate(Parameter & Beta, Parameter & Beta_Hat,
		Parameter & SigmaSquared, CyclicCoordinateDescent & ccd,
		boost::mt19937& rng, Eigen::MatrixXf& PrecisionMatrix,
		double tuningParameter) {

// Get the proposed Beta values
	vector<double> * betaPossible = Beta.returnCurrentValuesPointer();

	double logHastingsRatio = getHastingsRatio(Beta,Beta_Hat, PrecisionMatrix, tuningParameter);

// Compute log Likelihood and log prior

	ccd.resetBeta();
	ccd.setBeta(*betaPossible);



	double fBetaPossible = ccd.getLogLikelihood();
	double pBetaPossible = ccd.getLogPrior();

// Have we changed the Beta Values?  If so, get new Log Likelihood and prior values...
	//if (!storedValuesUpToDate) {

//		vector<double> * betaOldValues = Beta->returnStoredValuesPointer();
//		ccd.setBeta(*betaOldValues); // TODO MAS doubts this is necessary  // Just keep track of current logLikelihood... its a number, cache, do not recompute
	//}


//	(ccd.hXI_Transpose).setUseThisStatus(true); // Testing code

// Compute the ratio for the MH step
	double ratio = exp((fBetaPossible + pBetaPossible + logHastingsRatio) - (storedFBetaCurrent + storedPBetaCurrent));

// Set our alpha
	alpha = min(ratio, 1.0);
	static boost::uniform_01<boost::mt19937> zeroone(rng);

	if (ratio != ratio) {
		cerr << "pLL: " << fBetaPossible << endl;
		cerr << "cLL: " << storedFBetaCurrent << endl;
		cerr << "hR : " << logHastingsRatio << endl;
		exit(-1);
	}

// Sample from a uniform distribution
	double uniformRandom = zeroone();

	//cout << "SERIOUS WARNING: UNIFORM RANDOM = 0 >>>>>>>>>  change this " << endl;
	//uniformRandom = 0;

#ifdef Debug_TRS
	cout << "hastingsRatio = " << logHastingsRatio << endl;
	cout << "fBetaPossible = " << fBetaPossible << endl;
	cout << "fBetaCurrent = " << storedFBetaCurrent << endl;
	cout << "pBetaPossible = " << pBetaPossible << endl;
	cout << "pBetaCurrent = " << storedPBetaCurrent << endl;
	cout << "ratio = " << ratio << " and uniformRandom = " << uniformRandom << endl;
#endif


// This is the Metropolis step

	if (alpha > uniformRandom) {

#ifdef Debug_TRS
		cout << "\n \n \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  Change Beta @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n \n" << endl;
#endif

		Beta.setChangeStatus(true);
//		ccd.resetBeta();
//		ccd.setBeta(*betaPossible); // TODO MAS doubts this is necessary
		storedFBetaCurrent = fBetaPossible; // ccd.getLogLikelihood(); // TODO No need to recompute if cached correctly
		storedPBetaCurrent = pBetaPossible; // ccd.getLogPrior();



	} else{

#ifdef Debug_TRS
		cout << "##############  Reject Beta ##################" << endl;
#endif
		Beta.setChangeStatus(false);

		Beta.restore();
	}

	return alpha;


}

double MHRatio::getTransformedTuningValue(double tuningParameter){
	// TODO Don't forward reference like this.
	return exp(-tuningParameter);
}

double MHRatio::getHastingsRatio(Parameter & Beta,
		Parameter & Beta_Hat, Eigen::MatrixXf& PrecisionMatrix,
		double tuningParameter
		){


	int betaLength = Beta.getSize();
	Eigen::VectorXf betaCurrent(betaLength);
	Eigen::VectorXf betaProposal(betaLength);

	Eigen::VectorXf beta_hat(betaLength);

	Eigen::VectorXf betaHat_minus_current(betaLength);
	Eigen::VectorXf betaHat_minus_proposal(betaLength);

	Eigen::VectorXf precisionDifferenceProduct_current(betaLength);
	Eigen::VectorXf precisionDifferenceProduct_proposal(betaLength);


	for (int i = 0; i< betaLength; i++){
		betaProposal(i) = Beta.get(i);
		betaCurrent(i) = Beta.getStored(i);
		beta_hat(i) = Beta_Hat.get(i);
	}

	betaHat_minus_current = beta_hat - betaCurrent;
	betaHat_minus_proposal = beta_hat - betaProposal;

	precisionDifferenceProduct_current =  PrecisionMatrix * betaHat_minus_current;
	precisionDifferenceProduct_proposal = PrecisionMatrix * betaHat_minus_proposal;

	double numerator = betaHat_minus_current.dot(precisionDifferenceProduct_current);
	double denominator = betaHat_minus_proposal.dot(precisionDifferenceProduct_proposal);

	return(-0.5*(numerator - denominator) / getTransformedTuningValue(tuningParameter)); // log scale
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

