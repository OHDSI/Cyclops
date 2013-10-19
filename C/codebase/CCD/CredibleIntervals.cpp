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

}

void CredibleIntervals::computeCredibleIntervals(vector<double> * loglikelihoods, vector<vector<double> > * BetaValues, vector<double> * SigmaSquaredValues,
		double betaProbability, double sigmaProbability, std::string MCMCFileNameRoot){

	int nSamples = BetaValues->size();
	int betaSize = (*BetaValues)[0].size();
	vector<vector<bsccs::real> > returnValues;

	vector<bsccs::real> lowerBoundtoReturn;
	vector<bsccs::real> upperBoundtoReturn;

	cout << "nSamples = " << nSamples << endl;
	cout << "betaSize = " << betaSize << endl;

	//Write Beta Data to a file
	std::stringstream ss;
	ss << MCMCFileNameRoot << ".csv";
	string fileName = ss.str();
	ofstream outLog(fileName.c_str());

	string sep(","); // TODO Make option

	//Thinning...
	int thinningAmount = 1;



	for (int j = 0; j < nSamples;) {
		outLog << j << sep;
		outLog << std::setprecision(15) << (*loglikelihoods)[j] << sep;
		for (int k = 0; k < betaSize; k++) {
			outLog << (*BetaValues)[j][k] << sep;
		}
		outLog << (*SigmaSquaredValues)[j] << sep << endl;
		j = j + thinningAmount;
	}
	outLog.close();


	//TODO construct quantile intervals
}

void CredibleIntervals::logResults(const CCDArguments& arguments, std::string conditionId) {

	//TODO write as separate way to print all data
}

}
