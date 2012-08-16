/*
 * ModelLikelihood.h
 *
 *  Created on: Aug 15, 2012
 *      Author: trevorshaddox
 */

#ifndef MODELLIKELIHOOD_H_
#define MODELLIKELIHOOD_H_

#include "CyclicCoordinateDescent.h"

namespace bsccs {

	class ModelLikelihood {

	public:

		ModelLikelihood(CyclicCoordinateDescent* ccdIn);

		~ModelLikelihood();

		double getLL();

		bool likelihoodKnown;

	private:

		CyclicCoordinateDescent * ccd;

		double logLikelihoodValue;

	};

}


#endif /* MODELLIKELIHOOD_H_ */
