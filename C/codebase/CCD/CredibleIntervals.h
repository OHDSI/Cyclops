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

		void computeCredibleIntervals(vector<vector<double> > * BetaValues, vector<double> * SigmaSquaredValues, double betaProbability, double sigmaProbability);

		void logResults(const CCDArguments& arguments, std::string conditionId);

	private:

		vector<vector<bsccs::real> > BetaUpperAndlower;

		vector<bsccs::real> SigmaSquaredUpperAndLower;

	};
}



#endif /* CREDIBLEINTERVALS_H_ */
