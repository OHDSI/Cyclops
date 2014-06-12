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
		cout << "in initialize hyperprior size = " << size << endl;
		size = sizeIn;
		cout << "getSize = " << getSize() << endl;
		cout << "in initialize hyperprior size = " << size << endl;
		parameterValues = (bsccs::real*) calloc(sizeIn, sizeof(bsccs::real));
		storedValues = (bsccs::real*) calloc(size, sizeof(bsccs::real));

		for (int i = 0; i < size; i++){
			parameterValues[i] = ccd.getHyperprior();
			storedValues[i] = ccd.getHyperprior();
			cout << "in initialize hyperprior hyperprior = " << parameterValues[i] << endl;
		}

		cout << "getSize = " << getSize() << endl;
	}


}



