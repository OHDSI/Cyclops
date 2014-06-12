/*
 * TransitionKernel.cpp
 *
 *  Created on: Jul 23, 2013
 *      Author: tshaddox
 */

#include "TransitionKernel.h"

namespace bsccs {

 TransitionKernel::TransitionKernel(){}

 TransitionKernel::~TransitionKernel(){}

 void TransitionKernel::sample(MCMCModel & model, double tuningParameter,  std::default_random_engine& generator){
	 cout << "TransitionKernel::sample" << endl;
 }

 bool TransitionKernel::evaluateSample(MCMCModel& model, double tuningParameter,  CyclicCoordinateDescent & ccd){
	 cout << "TransitionKernel::evaluateSample" << endl;
	 return(false);
 }



 double TransitionKernel::generateGaussian(std::default_random_engine& generator)
 {
	 std::normal_distribution<double> distribution(0,1);
	 return(distribution(generator));
 }

}


