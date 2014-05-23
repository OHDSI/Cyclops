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
#include "Parameter.h"
#include "MCMCModel.h"

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

		MHRatio();

		~MHRatio();

		void initialize(CyclicCoordinateDescent & ccd);

		bool evaluate(MCMCModel & model);

		double getLogMetropolisRatio(MCMCModel & model);

		double getLogHastingsRatio(MCMCModel & model);

		double getStoredLogLikelihood() { return storedFBetaCurrent; }
		double getStoredLogPrior() { return storedPBetaCurrent; }
		double getTransformedTuningValue(double tuningParameter);


	private:
		double min(double value1, double value2);

		double fBetaPossible;
		double pBetaPossible;


		double storedFBetaCurrent;
		double storedPBetaCurrent;


		double acceptanceRatioNumerator;
		double acceptanceRatioDenominator;

		double alpha;
	};
}

#endif /* MHRATIO_H_ */
