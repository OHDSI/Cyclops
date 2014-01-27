/*
 * GeneralizedDirectSampler.cpp
 *
 *  Created on: Jan 27, 2014
 *      Author: tshaddox
 */




#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "GeneralizedDirectSampler.h"
#include "MHRatio.h"
#include "IndependenceSampler.h"
#include "RandomWalk.h"
#include "SigmaSampler.h"

#include "Parameter.h"

#include <boost/random.hpp>

//#define Debug_TRS
//#define DEBUG_STATE

namespace bsccs {


GeneralizedDirectSampler::GeneralizedDirectSampler(InputReader * inReader, std::string GDSFileName): reader(inReader){
	GDSFileNameRoot = GDSFileName;
	nDraws = 20;
	M = 30;
	dsScale = 0.1;

}



GeneralizedDirectSampler::~GeneralizedDirectSampler() {

}

}
