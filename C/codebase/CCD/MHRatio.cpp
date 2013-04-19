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

//#define Debug_TRS

namespace bsccs{

MHRatio::MHRatio(){
	storedValuesUpToDate = false;
}

MHRatio::~MHRatio(){

}

void MHRatio::evaluate(Parameter * Beta, Parameter * SigmaSquared, CyclicCoordinateDescent & ccd, boost::mt19937& rng) {

// Get the current Beta values
	vector<double> * betaPossible = Beta->returnCurrentValuesPointer();


// Compute log Likelihood and log prior
	ccd.resetBeta();
	ccd.setBeta(*betaPossible);
	//cout << "logging Beta in MHRatio:evalulate" << endl;
	//Beta->logParameter();
	//cout << "done logging" << endl;

	//cout << "\n Calling Log Likelihood " << endl;
	double fBetaPossible = ccd.getLogLikelihood();
	//cout << "\n Called Log Likelihood " << endl;
	double pBetaPossible = ccd.getLogPrior();

// Have we changed the Beta Values?  If so, get new Log Likelihood and prior values...
	//if (!storedValuesUpToDate) {

		vector<double> * betaOldValues = Beta->returnStoredValuesPointer();
		ccd.setBeta(*betaOldValues);   // Just keep track of current logLikelihood... its a number, cache, do not recompute
		storedFBetaCurrent = ccd.getLogLikelihood();
		storedPBetaCurrent = ccd.getLogPrior();
	//}


	(ccd.hXI_Transpose).setUseThisStatus(true); // Testing code

// Compute the ratio for the MH step
	double ratio = exp((fBetaPossible + pBetaPossible) - (storedFBetaCurrent + storedPBetaCurrent));

// Set our alpha
	alpha = min(ratio, 1.0);
	static boost::uniform_01<boost::mt19937> zeroone(rng);

// Sample from a uniform distribution
	double uniformRandom = zeroone();

#ifdef Debug_TRS
	cout << "fBetaPossible = " << fBetaPossible << endl;
	cout << "fBetaCurrent = " << storedFBetaCurrent << endl;
	cout << "pBetaPossible = " << pBetaPossible << endl;
	cout << "pBetaCurrent = " << storedPBetaCurrent << endl;
	cout << "ratio = " << ratio << " and uniformRandom = " << uniformRandom << endl;
#endif


// This is the Metropolis step
	if (alpha > uniformRandom) {
		Beta->setChangeStatus(true);
		storedValuesUpToDate = false;
#ifdef Debug_TRS
		cout << "--------------  Change Beta ------------------" << endl;
#endif
	} else{
#ifdef Debug_TRS
		cout << "##############  Reject Beta ##################" << endl;
#endif
		Beta->setChangeStatus(false);
		storedValuesUpToDate = true;
		Beta->restore();
	}

// This determines if we will perform a Gibbs step to update sigma squared on the next iteration
	if (sigmaSampleTuningParameter*alpha > uniformRandom) {
		SigmaSquared->setChangeStatus(false);
		SigmaSquared->setNeedToChangeStatus(true);
#ifdef Debug_TRS
		cout << "*****************  Change Sigma Squared ********************" << endl;
#endif
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

