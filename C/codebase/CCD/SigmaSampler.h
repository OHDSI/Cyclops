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
#include "TransitionKernel.h"
#include "Model.h"

using std::cout;
using std::cerr;
using std::endl;
//using std::in;
using std::ifstream;


namespace bsccs {

class SigmaSampler: public TransitionKernel {

public:

	SigmaSampler();

	virtual ~SigmaSampler();

	void sample(Model& model, double tuningParameter, boost::mt19937& rng);

	bool evaluateSample(Model& model, double tuningParameter, boost::mt19937& rng, CyclicCoordinateDescent & ccd);

protected:


};

}

#endif /* SIGMACALCULATE_H_ */
