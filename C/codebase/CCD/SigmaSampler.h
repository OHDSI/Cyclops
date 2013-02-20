/*
 * SigmaCalculate.h
 *
 *  Created on: Aug 9, 2012
 *      Author: trevorshaddox
 */

#ifndef SIGMACALCULATE_H_
#define SIGMACALCULATE_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>


#include <boost/random.hpp>

#include "Parameter.h"

using std::cout;
using std::cerr;
using std::endl;
//using std::in;
using std::ifstream;


namespace bsccs {

class SigmaSampler {

public:

	SigmaSampler();

	virtual ~SigmaSampler();

	void sampleSigma(Parameter * SigmaSquared, Parameter * BetaValues, boost::mt19937& rng);

protected:


};

}

#endif /* SIGMACALCULATE_H_ */
