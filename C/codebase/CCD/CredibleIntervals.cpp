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

void CredibleIntervals::computeCredibleIntervals(vector<vector<double> > * BetaValues, vector<double> * SigmaSquaredValues,
		double betaProbability, double sigmaProbability, std::string MCMCFileNameRoot){

	int nSamples = BetaValues->size();
	int betaSize = (*BetaValues)[0].size();
	vector<vector<bsccs::real> > returnValues;

	vector<bsccs::real> lowerBoundtoReturn;
	vector<bsccs::real> upperBoundtoReturn;

	cout << "nSamples = " << nSamples << endl;
	cout << "betaSize = " << betaSize << endl;

	for (int i = 0; i < betaSize; i ++) {
	//	cout << "Avg Beta_" << i << " = ";
		double sum = 0;
		for (int j = 0; j < nSamples; j ++) {
			sum += (*BetaValues)[j][i];
		}
	//	cout << sum/nSamples << endl;
	}


	//Write Beta Data to a file
	std::stringstream ss;
	ss << MCMCFileNameRoot << "_beta.csv";
	string fileName = ss.str();
	ofstream outLog(fileName.c_str());

	string sep(","); // TODO Make option

	//Thinning...
	int thinningAmount = 1;

	for (int j = 0; j < nSamples;) {
		for (int k = 0; k < betaSize; k++) {
			outLog << (*BetaValues)[j][k] << sep;
		}
		outLog << endl;
		j = j + thinningAmount;
	}
	outLog.close();
	nSamples = SigmaSquaredValues->size();

	//Write Sigma Data to a file
	std::stringstream ss2;
	ss2 << MCMCFileNameRoot << "_sigma.csv";
	string fileName2 = ss2.str();
	ofstream outLog2(fileName2.c_str());

	string sep2(","); // TODO Make option

	for (int j = 0; j < nSamples; ++j) {
		outLog2 << (*SigmaSquaredValues)[j] << sep2 << endl;
	}
	outLog2.close();


	//TODO construct quantile intervals
}

void CredibleIntervals::logResults(const CCDArguments& arguments, std::string conditionId) {

	//TODO write as separate way to print all data
}

}
