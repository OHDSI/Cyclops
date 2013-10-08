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

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

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
void OneDimensionRandomWalk::sample(Model& model, double tuningParameter, boost::mt19937& rng) {

	Parameter & Beta = model.getBeta();
	Parameter & Beta_Hat = model.getBeta_Hat();
	Eigen::LLT<Eigen::MatrixXf> choleskyEigen = model.getCholeskyLLT();

	int sizeOfSample = Beta.getSize();

	// Select random coordinate
	coordinate = rand() % sizeOfSample; // TODO not a good way to do this.




	vector<bsccs::real> independentNormal;  //Sampled independent normal values

	boost::normal_distribution<> nd(0.0, 1.0); // TODO Construct once

	boost::variate_generator<boost::mt19937&,
	                           boost::normal_distribution<> > var_nor(rng, nd); // TODO Construct once


	bsccs::real normalValue = var_nor();
	// NB: tuningParameter scales the VARIANCE
	bsccs::real scaledNormalValue = normalValue * std::sqrt(getTransformedTuningValue(tuningParameter)); // multiply by stdev

	scaledNormalValue = scaledNormalValue / (choleskyEigen.matrixU())(coordinate, coordinate); //TODO is this right??

	Beta.set(coordinate, scaledNormalValue + Beta_Hat.get(coordinate));
}

bool OneDimensionRandomWalk::evaluateSample(Model& model, double tuningParameter, boost::mt19937& rng, CyclicCoordinateDescent & ccd){

	Parameter & Beta = model.getBeta();
	Parameter & Beta_Hat = model.getBeta_Hat();

	bool accept = MHstep.evaluate(model, model.getBeta(), model.getBeta_Hat(), model.getSigmaSquared(), ccd, rng, model.getHessian(), tuningParameter);

	if(accept) {
		Beta_Hat.set(coordinate, Beta.get(coordinate));
	}

	return(accept);
}



}





