/*
 * BetaParameter.h
 *
 *  Created on: Aug 10, 2012
 *      Author: trevorshaddox
 */

#ifndef BETAPARAMETER_H_
#define BETAPARAMETER_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "CyclicCoordinateDescent.h"
#include "Parameter.h"

namespace bsccs{


#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif


	class BetaParameter : public Parameter {
	public:
		BetaParameter();

		~BetaParameter();

		void initialize(CyclicCoordinateDescent& ccd, int sizeIn);



	private:

		int size;

		bsccs::real * parameterValues;

		bsccs::real * storedValues;

		std::vector<double> parameterDoubleValues;

		std::vector<double> storedDoubleValues;

		std::vector<bool> vectorOfChanges;

		bool restorable;
	};
}


#endif /* BETAPARAMETER_H_ */
