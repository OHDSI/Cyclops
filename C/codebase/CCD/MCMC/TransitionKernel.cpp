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

 void TransitionKernel::sample(Model & model, double tuningParameter, boost::mt19937& rng){
	 cout << "TransitionKernel::sample" << endl;
 }

 bool TransitionKernel::evaluateSample(Model& model, double tuningParameter, boost::mt19937& rng, CyclicCoordinateDescent & ccd){
	 cout << "TransitionKernel::evaluateSample" << endl;
	 return(false);
 }

}


