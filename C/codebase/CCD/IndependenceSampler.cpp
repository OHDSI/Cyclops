/*
 * IndependenceSampler.cpp
 *
 *  Created on: Aug 6, 2012
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

#include "IndependenceSampler.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>



#define PI	3.14159265358979323851280895940618620443274267017841339111328125

//#define Debug_TRS

namespace bsccs {

IndependenceSampler::IndependenceSampler(CyclicCoordinateDescent & ccd) {
	MHstep.initialize(ccd);
}


IndependenceSampler::~IndependenceSampler() {

}

double IndependenceSampler::getTransformedTuningValue(double tuningParameter){
	return exp(-tuningParameter);
}


void IndependenceSampler::sample(Model& model, double tuningParameter, boost::mt19937& rng) {
	//cout << "IndependenceSampler::sample" << endl;

	model.BetaStore();
	Parameter & Beta = model.getBeta();
	Parameter & Beta_Hat = model.getBeta_Hat();
	//Eigen::LLT<Eigen::MatrixXf> choleskyEigen = model.getCholeskyLLT();


	int sizeOfSample = Beta.getSize();


	vector<bsccs::real> independentNormal;  //Sampled independent normal values

	boost::normal_distribution<> nd(0.0, 1.0); // TODO Construct once

	boost::variate_generator<boost::mt19937&,
	                           boost::normal_distribution<> > var_nor(rng, nd); // TODO Construct once

	Eigen::VectorXf b = Eigen::VectorXf::Random(sizeOfSample);
	for (int i = 0; i < sizeOfSample; i++) {
		bsccs::real normalValue = var_nor();
		// NB: tuningParameter scales the VARIANCE
		b[i] = normalValue * std::sqrt(getTransformedTuningValue(tuningParameter)); // multiply by stdev
	}

#ifdef Debug_TRS
	cout << "Cholesky in Sampler " << endl;
	Eigen::MatrixXf CholeskyDecompL(sizeOfSample, sizeOfSample);
	CholeskyDecompL = choleskyEigen.matrixU();
	cout << CholeskyDecompL << endl;
#endif

	((model.getCholeskyLLT()).matrixU()).solveInPlace(b);

	// TODO Check marginal variance on b[i]


	for (int i = 0; i < sizeOfSample; i++) {
		Beta.set(i, b[i] + Beta_Hat.get(i));
	}

	/*
	cout << "Beta current" << endl;
	Beta.logParameter();
	cout << "Beta storred" << endl;
	Beta.logStored();
	cout << "BetaHat current" << endl;
	Beta_Hat.logParameter();
	cout << "BetaHat storred" << endl;
	Beta_Hat.logStored();
	*/

}

bool IndependenceSampler::evaluateSample(Model& model, double tuningParameter, boost::mt19937& rng, CyclicCoordinateDescent & ccd){
	//cout << "IndependenceSampler::evaluateSample" << endl;

	bool accept = MHstep.evaluate(model);


	return(accept);
}


}
