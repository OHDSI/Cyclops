/*
 * GeneralizedDirectSampler.h
 *
 *  Created on: Jan 27, 2014
 *      Author: tshaddox
 */



#ifndef GENERALIZEDDIREDSAMMPLER_H_
#define GENERALIZEDDIREDSAMMPLER_H_


#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "ccd.h"
#include "Parameter.h"
#include "Model.h"
#include "TransitionKernel.h"
#include "CredibleIntervals.h"

namespace bsccs {
class GeneralizedDirectSampler {
public:
	GeneralizedDirectSampler(InputReader * inReader, std::string MCMCFileName);

	virtual ~GeneralizedDirectSampler();

private:

	std::string GDSFileNameRoot;

	InputReader* reader;

	int nDraws;
	int M;
	double dsScale;


};
}


#endif /* GENERALIZEDDIREDSAMMPLER_H_ */

