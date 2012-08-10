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
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_matrix.h>
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

vector<vector<bsccs::real> > CredibleIntervals::computeCredibleIntervals(vector<gsl_vector*> sampledBetaValues, int betaSize, int nSamples){

	vector<vector<bsccs::real> > returnValues;

	vector<bsccs::real> lowerBoundtoReturn;
	vector<bsccs::real> upperBoundtoReturn;

	cout << "nSamples = " << nSamples << endl;

	for (int i = 0 ; i < betaSize; i++) {
		double MCMCSamples[nSamples];
		for (int j = 0; j < nSamples; j++) {
			MCMCSamples[j] = gsl_vector_get(sampledBetaValues[j],i);
		}
		gsl_sort(MCMCSamples,1,nSamples);
		lowerBoundtoReturn.push_back(gsl_stats_quantile_from_sorted_data(MCMCSamples,1,nSamples,0.05));

		upperBoundtoReturn.push_back(gsl_stats_quantile_from_sorted_data(MCMCSamples,1,nSamples,0.95));

	}


	returnValues.push_back(lowerBoundtoReturn);
	returnValues.push_back(upperBoundtoReturn);
	return returnValues;
}

}
