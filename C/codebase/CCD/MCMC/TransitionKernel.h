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
#include <random>

#include "CyclicCoordinateDescent.h"
#include "MCMCModel.h"


namespace bsccs {
class TransitionKernel {
public:
	TransitionKernel();
	virtual ~TransitionKernel();
	virtual void sample(MCMCModel & model, double tuningParameter,  std::default_random_engine& generator);

	virtual bool evaluateSample(MCMCModel& model, double tuningParameter, CyclicCoordinateDescent & ccd);

	double generateGaussian(std::default_random_engine& generator);
};
}


#endif /* TRANSITIONKERNEL_H_ */


