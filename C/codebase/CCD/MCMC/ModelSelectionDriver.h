/*
 * ModelSelectionDriver.h
 *
 *  Created on: Jun 26, 2014
 *      Author: trevorshaddox2
 */

#ifndef MODELSELECTIONDRIVER_H_
#define MODELSELECTIONDRIVER_H_


#include <unordered_map>
#include <iostream>
#include <numeric>
#include <vector>
#include <set>
#include <math.h>
#include <cstdlib>
#include <time.h>

#include "MCMCModel.h"
#include "ModelSampler.h"
#include "MCMCDriver.h"
#include "ModelPrior.h"


namespace bsccs {

class ModelSelectionDriver {
public:
	ModelSelectionDriver();

	virtual ~ModelSelectionDriver();

	void drive(CyclicCoordinateDescent& ccd, long int seed, string MCMCFilename, double betaAmount);

};
}


#endif /* MODELSELECTIONDRIVER_H_ */
