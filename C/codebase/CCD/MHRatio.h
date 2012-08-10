/*
 * MHRatio.h
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */

#ifndef MHRATIO_H_
#define MHRATIO_H_

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
#include <gsl/gsl_matrix.h>
#include "IndependenceSampler.h"


using std::cout;
using std::cerr;
using std::endl;
//using std::in;
using std::ifstream;


namespace bsccs {

	class MHRatio {

	public:

		MHRatio();

		~MHRatio();

		bool acceptBetaBool(CyclicCoordinateDescent &ccd, IndependenceSampler * sampler, double uniformRandom, int betaSize, gsl_vector * betaOld, gsl_vector * betaToEvaluate, gsl_matrix * covarianceMatrix);

		bool getSigmaSquaredBool(double uniformRandom);

	private:
		double min(double value1, double value2);

		double acceptanceRatioNumerator;
		double acceptanceRatioDenominator;

		double alpha;

		double fudgeFactor;

	};
}

#endif /* MHRATIO_H_ */
