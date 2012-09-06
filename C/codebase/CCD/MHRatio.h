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

		bool evaluate(Parameter * Beta, Parameter * SigmaSquared, CyclicCoordinateDescent & ccd, vector<vector<bsccs::real> > * precisionMatrix, bsccs::real precisionDeterminant);

	private:
		double min(double value1, double value2);

		double acceptanceRatioNumerator;
		double acceptanceRatioDenominator;

		double alpha;

		double fudgeFactor;

	};
}

#endif /* MHRATIO_H_ */
