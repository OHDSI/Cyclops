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
	fudgeFactor = 1;
}

MHRatio::~MHRatio(){

}

void MHRatio::evaluate(Parameter * Beta, Parameter * SigmaSquared, CyclicCoordinateDescent & ccd, vector<vector<bsccs::real> > * precisionMatrix) {

	int betaSize = Beta->getSize();

	vector<double> betaPossible = Beta->returnCurrentValues();
	vector<double> betaOldValues = Beta->returnStoredValues();

	(ccd.hXI_Transpose).setUseThisStatus(true);

	ccd.setBeta(betaPossible);
	double fBetaPossible = ccd.getLogLikelihood();
	double pBetaPossible = ccd.getLogPrior();

	ccd.setBeta(betaOldValues);
	double fBetaCurrent = ccd.getLogLikelihood();
	double pBetaCurrent = ccd.getLogPrior();

	(ccd.hXI_Transpose).setUseThisStatus(false);

	double ratio = exp(fBetaPossible + pBetaPossible - (fBetaCurrent + pBetaCurrent));

	alpha = min(ratio, 1.0);
	boost::mt19937 rng(43);
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	double uniformRandom = zeroone();

#ifdef Debug_TRS
	cout << "ratio = " << ratio << " and uniformRandom = " << uniformRandom << endl;
#endif

	if (alpha > uniformRandom) {
		Beta->setChangeStatus(true);
#ifdef Debug_TRS
		cout << "--------------  Change Beta ------------------" << endl;
#endif
	} else{
#ifdef Debug_TRS
		cout << "##############  Reject Beta ##################" << endl;
#endif
		Beta->setChangeStatus(false);
		Beta->restore();
	}

	if (fudgeFactor*alpha > uniformRandom) {
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

