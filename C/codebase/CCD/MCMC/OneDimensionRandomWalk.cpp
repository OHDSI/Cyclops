/*
 * OneDimensionRandomWalk.cpp
 *
 *  Created on: Sep 13, 2013
 *      Author: trevorshaddox
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <ctime>
#include <set>

#include <math.h>

#include "OneDimensionRandomWalk.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>

#define PI	3.14159265358979323851280895940618620443274267017841339111328125

//#define Debug_TRS

namespace bsccs {

OneDimensionRandomWalk::OneDimensionRandomWalk(CyclicCoordinateDescent & ccd, double seed) {
	srand(seed);
	coordinate = 0;
	MHstep.initialize(ccd);
}


OneDimensionRandomWalk::~OneDimensionRandomWalk() {

}

double OneDimensionRandomWalk::getTransformedTuningValue(double tuningParameter){
	return exp(-tuningParameter);
}
void OneDimensionRandomWalk::sample(MCMCModel& model, double tuningParameter,  std::default_random_engine& generator) {

	BetaParameter & Beta = model.getBeta();
	BetaParameter & Beta_Hat = model.getBeta_Hat();
	Eigen::LLT<Eigen::MatrixXf> choleskyEigen = model.getCholeskyLLT();

	int sizeOfSample = Beta.getSize();

	// Select random coordinate
	coordinate = rand() % sizeOfSample; // TODO not a good way to do this.




	Eigen::VectorXf independentNormal = Eigen::VectorXf::Random(sizeOfSample);
	for (int i = 0; i < sizeOfSample; i++) {
		bsccs::real normalValue = generateGaussian(generator);
		// NB: tuningParameter scales the VARIANCE
		independentNormal[i] = normalValue * std::sqrt(getTransformedTuningValue(tuningParameter)); // multiply by stdev
	}



	bsccs::real normalValue = generateGaussian(generator);
	// NB: tuningParameter scales the VARIANCE
	bsccs::real scaledNormalValue = normalValue * std::sqrt(getTransformedTuningValue(tuningParameter)); // multiply by stdev

	scaledNormalValue = scaledNormalValue / (choleskyEigen.matrixU())(coordinate, coordinate); //TODO is this right??

	Beta.set(coordinate, scaledNormalValue + Beta_Hat.get(coordinate));
}

bool OneDimensionRandomWalk::evaluateSample(MCMCModel& model, double tuningParameter, CyclicCoordinateDescent & ccd){

	BetaParameter & Beta = model.getBeta();
	BetaParameter & Beta_Hat = model.getBeta_Hat();

	bool accept = MHstep.evaluate(model);

	if(accept) {
		Beta_Hat.set(coordinate, Beta.get(coordinate));
	}

	return(accept);
}



}





