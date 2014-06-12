/*
 * RandomWalk.h
 *
 *  Created on: Jul 31, 2013
 *      Author: tshaddox
 */

#ifndef RANDOMWALK_H_
#define RANDOMWALK_H_



#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>

#include "BetaParameter.h"
#include "TransitionKernel.h"
#include "MHRatio.h"


#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "ccd.h"


namespace bsccs {

class RandomWalk : public TransitionKernel {

public:

	RandomWalk(CyclicCoordinateDescent & ccd);

	virtual ~RandomWalk();

	void sample(MCMCModel& model, double tuningParameter,  std::default_random_engine& generator);

	bool evaluateSample(MCMCModel& model, double tuningParameter, CyclicCoordinateDescent& ccd);

	double getTransformedTuningValue(double tuningParameter);

protected:

	MHRatio MHstep;

};

}



#endif /* RANDOMWALK_H_ */
