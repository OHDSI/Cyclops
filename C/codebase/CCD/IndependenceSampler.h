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

#include <Eigen/Dense>
#include <Eigen/Cholesky>

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

	void sample(Parameter * Beta_Hat, Parameter * Beta, boost::mt19937& rng, Eigen::LLT<Eigen::MatrixXf> & choleskyEigen);

protected:



};

}

#endif /* INDEPENDENCESAMPLER_H_ */
