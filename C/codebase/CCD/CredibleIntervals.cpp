/*
 * CredibleIntervals.cpp
 *
 *  Created on: Aug 10, 2012
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



#include "CyclicCoordinateDescent.h"
#include "IndependenceSampler.h"

using std::cout;
using std::cerr;
using std::endl;
//using std::in;
using std::ifstream;

#include <CredibleIntervals.h>

namespace bsccs {
CredibleIntervals::CredibleIntervals() {

}

CredibleIntervals::~CredibleIntervals(){
	//outLog.close();
}

void CredibleIntervals::initialize(std::string MCMCFileNameRootIn) {
	MCMCFileNameRoot = MCMCFileNameRootIn;
	std::stringstream ss;
	ss << MCMCFileNameRoot << ".csv";
	string fileName = ss.str();
	outLog.open(fileName.c_str());
}

void CredibleIntervals::fileLogCredibleIntervals(double loglikelihood, vector<double> * BetaValues, double SigmaSquaredValue, int iteration){

	//
	int betaSize = BetaValues->size();

	string sep(","); // TODO Make option

	outLog << iteration << sep;
	outLog << std::setprecision(15) << loglikelihood << sep;
	for (int k = 0; k < betaSize; k++) {
		outLog << (*BetaValues)[k] << sep;
	}
	outLog << SigmaSquaredValue << sep << endl;

}



}
