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

	void initialize(Model & model, CyclicCoordinateDescent& ccd, long int seed);

	void drive(CyclicCoordinateDescent& ccd,long int seed);

	int getMultinomialSample(vector<double>& probabilites, long int seed);

	double getTransformedTuningValue(double tuningParameter);

private:

	std::string GDSFileNameRoot;

	InputReader* reader;

	int nDraws; // "R" in the algorithm of Braun et al.
	int M; // number of proposal draws
	double dsScale;

	IndependenceSampler* GDSSampler;

	Parameter mode;

	CredibleIntervals intervalsToReport;


};
}


#endif /* GENERALIZEDDIREDSAMMPLER_H_ */

