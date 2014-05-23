/*
 * OneDimensionRandomWalk.h
 *
 *  Created on: Sep 13, 2013
 *      Author: trevorshaddox
 */

#ifndef ONEDIMENSIONRANDOMWALK_H_
#define ONEDIMENSIONRANDOMWALK_H_


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

class OneDimensionRandomWalk : public TransitionKernel {

public:

	OneDimensionRandomWalk(CyclicCoordinateDescent & ccd, double seed);

	virtual ~OneDimensionRandomWalk();

	void sample(MCMCModel& model, double tuningParameter);

	bool evaluateSample(MCMCModel& model, double tuningParameter, CyclicCoordinateDescent& ccd);

	double getTransformedTuningValue(double tuningParameter);

protected:

	MHRatio MHstep;
	int coordinate;

};

}

#endif /* ONEDIMENSIONRANDOMWALK_H_ */
