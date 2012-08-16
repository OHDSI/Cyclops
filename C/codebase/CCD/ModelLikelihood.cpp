/*
 * ModelLikelihood.cpp
 *
 *  Created on: Aug 15, 2012
 *      Author: trevorshaddox
 */

#include "ModelLikelihood.h"
#include "CyclicCoordinateDescent.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

namespace bsccs {

	ModelLikelihood::ModelLikelihood(CyclicCoordinateDescent * ccdIn){
		ccd = ccdIn;
		likelihoodKnown = false;
	}

	ModelLikelihood::~ModelLikelihood(){

	}

	double ModelLikelihood::getLL(){
		if (likelihoodKnown) {
			cout << "HERE!" << endl;
			return logLikelihoodValue;
		} else {
			logLikelihoodValue = ccd->getLogLikelihood();
			likelihoodKnown = true;
			return logLikelihoodValue;
		}
	}


}


