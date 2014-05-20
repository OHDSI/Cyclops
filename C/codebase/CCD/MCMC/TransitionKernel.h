/*
 * TransitionKernel.h
 *
 *  Created on: Jul 23, 2013
 *      Author: tshaddox
 */

#ifndef TRANSITIONKERNEL_H_
#define TRANSITIONKERNEL_H_

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "Parameter.h"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "CyclicCoordinateDescent.h"
#include "Model.h"


namespace bsccs {
class TransitionKernel {
public:
	TransitionKernel();
	virtual ~TransitionKernel();
	virtual void sample(Model & model, double tuningParameter, boost::mt19937& rng);

	virtual bool evaluateSample(Model& model, double tuningParameter, boost::mt19937& rng, CyclicCoordinateDescent & ccd);
};
}


#endif /* TRANSITIONKERNEL_H_ */


