/*
 * IndependenceSampler.h
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */

#ifndef INDEPENDENCESAMPLER_H_
#define INDEPENDENCESAMPLER_H_


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

class IndependenceSampler : public TransitionKernel {

public:

	IndependenceSampler(CyclicCoordinateDescent & ccd);

	virtual ~IndependenceSampler();

	void sample(MCMCModel& model, double tuningParameter,  std::default_random_engine& generator);

	bool evaluateSample(MCMCModel& model, double tuningParameter, CyclicCoordinateDescent& ccd);

	double getTransformedTuningValue(double tuningParameter);

	double evaluateLogMHRatio(MCMCModel& model);


protected:

	MHRatio MHstep;

};

}

#endif /* INDEPENDENCESAMPLER_H_ */
