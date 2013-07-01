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

#include "CyclicCoordinateDescent.h"
#include "IndependenceSampler.h"

#include <Eigen/Dense>
#include <Eigen/Cholesky>


using std::cout;
using std::cerr;
using std::endl;
//using std::in;
using std::ifstream;


namespace bsccs {

	class MHRatio {

	public:

		MHRatio(CyclicCoordinateDescent & ccd);

		~MHRatio();

		double evaluate(Parameter * Beta, Parameter * Beta_Hat,
				Parameter * SigmaSquared, CyclicCoordinateDescent & ccd,
				boost::mt19937& rng, Eigen::MatrixXf PrecisionMatrix);

		double getHastingsRatio(Parameter * Beta,
				Parameter * Beta_Hat,
				Eigen::MatrixXf PrecisionMatrix);

	private:
		double min(double value1, double value2);

		double storedFBetaCurrent;
		double storedPBetaCurrent;


		double acceptanceRatioNumerator;
		double acceptanceRatioDenominator;

		double alpha;
	};
}

#endif /* MHRATIO_H_ */
