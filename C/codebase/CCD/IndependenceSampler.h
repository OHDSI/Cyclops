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

#include "Parameter.h"
#include "TransitionKernel.h"
#include "MHRatio.h"

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "ccd.h"

#include <boost/random.hpp>

namespace bsccs {

class IndependenceSampler : public TransitionKernel {

public:

	IndependenceSampler(CyclicCoordinateDescent & ccd);

	virtual ~IndependenceSampler();

	void sample(Model& model, double tuningParameter, boost::mt19937& rng);

	bool evaluateSample(Model& model, double tuningParameter, boost::mt19937& rng, CyclicCoordinateDescent& ccd);

	double getTransformedTuningValue(double tuningParameter);

protected:

	MHRatio MHstep;

};

}

#endif /* INDEPENDENCESAMPLER_H_ */
