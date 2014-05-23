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

 void TransitionKernel::sample(MCMCModel & model, double tuningParameter){
	 cout << "TransitionKernel::sample" << endl;
 }

 bool TransitionKernel::evaluateSample(MCMCModel& model, double tuningParameter,  CyclicCoordinateDescent & ccd){
	 cout << "TransitionKernel::evaluateSample" << endl;
	 return(false);
 }

 //  Replacing the Boost library normal random variable, based on the Wikipedia code

 #define TWO_PI 6.2831853071795864769252866
 double TransitionKernel::generateGaussian()
 {
	 std::default_random_engine generator;
	 std::normal_distribution<double> distribution(0,1);
	 return(distribution(generator));
 }

}


