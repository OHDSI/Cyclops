/*
 * BetaParameter.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: trevorshaddox
 */

#include "BetaParameter.h"



using namespace std;
namespace bsccs{


	BetaParameter::BetaParameter(){}
	BetaParameter::~BetaParameter(){}

	void BetaParameter::initialize(CyclicCoordinateDescent& ccd, int sizeIn){
		restorable = true;
		size = sizeIn;
		cout << "size = " << size << endl;
		parameterValues = (bsccs::real*) calloc(sizeIn, sizeof(bsccs::real));
		storedValues = (bsccs::real*) calloc(size, sizeof(bsccs::real));

		for (int i = 0; i < size; i++){
			parameterValues[i] = ccd.getBeta(i);
			storedValues[i] = ccd.getBeta(i);
		}

		cout << "getSize = " << getSize() << endl;
	}

}



