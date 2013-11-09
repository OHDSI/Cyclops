/*
 * CredibleIntervals.h
 *
 *  Created on: Aug 10, 2012
 *      Author: trevorshaddox
 */

#ifndef CREDIBLEINTERVALS_H_
#define CREDIBLEINTERVALS_H_


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


namespace bsccs {

	class CredibleIntervals {

	public:

		CredibleIntervals();

		~CredibleIntervals();

		void initialize(std::string MCMCFileNameRootIn);

		void fileLogCredibleIntervals(double loglikelihood, vector<double> * BetaValues, double SigmaSquaredValue, int iteration);

	private:

		std::string MCMCFileNameRoot;

		ofstream outLog;

	};
}



#endif /* CREDIBLEINTERVALS_H_ */
