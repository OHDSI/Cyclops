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

#include "Parameter.h"
#include "HyperpriorParameter.h"
#include "BetaParameter.h"
#include "TransitionKernel.h"
#include "MCMCModel.h"

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

	void sample(MCMCModel& model, double tuningParameter);

	bool evaluateSample(MCMCModel& model, double tuningParameter, CyclicCoordinateDescent & ccd);

protected:


};

}

#endif /* SIGMACALCULATE_H_ */
