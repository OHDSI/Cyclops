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

#include <Eigen/core>

#include "Parameter.h"


#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "ccd.h"

#include <boost/random.hpp>

namespace bsccs {

class IndependenceSampler {

public:

	IndependenceSampler();

	virtual ~IndependenceSampler();

	void sample(Parameter * Beta_Hat, Parameter * Beta, std::vector<std::vector<bsccs::real> > cholesky, boost::mt19937& rng, double tuningParameter);

protected:


};

}

#endif /* INDEPENDENCESAMPLER_H_ */
