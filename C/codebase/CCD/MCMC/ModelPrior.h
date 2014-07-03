/*
 * ModelPrior.h
 *
 *  Created on: Jun 21, 2014
 *      Author: trevorshaddox2
 */

#ifndef MODELPRIOR_H_
#define MODELPRIOR_H_

#include <iostream>
#include <numeric>
#include <vector>
#include <set>
#include <math.h>
#include <cstdlib>
#include <time.h>

#include "MCMCModel.h"



namespace bsccs {

class ModelPrior {
public:

	ModelPrior();

	virtual ~ModelPrior();

	bsccs::real getLogPrior(MCMCModel& model);
};
}






#endif /* MODELPRIOR_H_ */
