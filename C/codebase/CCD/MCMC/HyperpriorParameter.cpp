/*
 * HyperpriorParameter.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: trevorshaddox
 */

#include "HyperpriorParameter.h"



using namespace std;
namespace bsccs{


	HyperpriorParameter::HyperpriorParameter(){}
	HyperpriorParameter::~HyperpriorParameter(){}

	void HyperpriorParameter::initialize(CyclicCoordinateDescent& ccd, int sizeIn){
		restorable = true;
		size = sizeIn;
		parameterValues = (bsccs::real*) calloc(sizeIn, sizeof(bsccs::real));
		storedValues = (bsccs::real*) calloc(size, sizeof(bsccs::real));

		for (int i = 0; i < size; i++){
			parameterValues[i] = ccd.getHyperprior();
			storedValues[i] = ccd.getHyperprior();
		}
	}


}



