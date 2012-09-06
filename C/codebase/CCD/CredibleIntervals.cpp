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

#include <Eigen/core>

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

void CredibleIntervals::computeCredibleIntervals(vector<vector<double> > * BetaValues, vector<double> * SigmaSquaredValues){

	int nSamples = BetaValues->size();
	int betaSize = (*BetaValues)[0].size();
	vector<vector<bsccs::real> > returnValues;

	vector<bsccs::real> lowerBoundtoReturn;
	vector<bsccs::real> upperBoundtoReturn;

	cout << "nSamples = " << nSamples << endl;
	cout << "betaSize = " << betaSize << endl;

	for (int i = 0; i < betaSize; i ++) {
		cout << "Avg Beta_" << i << " = ";
		double sum = 0;
		for (int j = 0; j < nSamples; j ++) {
			sum += (*BetaValues)[j][i];
		}
		cout << sum/nSamples << endl;
	}

	/*//Write Beta Data to a file
	string fileName = "/Users/trevorshaddox/Desktop/CredibleIntervals_out.csv";
	ofstream outLog(fileName.c_str());

	string sep(","); // TODO Make option

	for (int j = 0; j < nSamples; ++j) {
		outLog << (*BetaValues)[j][0] << sep << (*BetaValues)[j][1] << endl;
	}
	outLog.close();
	*/

	//TODO construct quantile intervals
}

void CredibleIntervals::logResults(const CCDArguments& arguments, std::string conditionId) {

	/*
	string fileName = "/Users/trevorshaddox/Desktop/CredibleIntervals_out.csv"
	ofstream outLog(fileName.c_str());
	if (!outLog) {
		cerr << "Unable to open log file: " << arguments.bsFileName << endl;
		exit(-1);
	}
	//map<int, DrugIdType> drugMap = reader->getDrugNameMap();

	string sep(","); // TODO Make option


	if (!arguments.reportRawEstimates) {
		outLog << "Drug_concept_id" << sep << "Condition_concept_id" << sep <<
				"score" << sep << "lower" << sep <<
				"upper" << endl;
	}


	for (int j = 0; j < J; ++j) {
		outLog << drugMap[j] << sep << conditionId << sep;

		outLog << BetaValues[j] << sep << credibleIntervals[0][j] << sep << credibleIntervals[1][j] << endl;

	}
	outLog.close();
	*/
}

}
