/*
 * RandomWalk.cpp
 *
 *  Created on: Jul 31, 2013
 *      Author: tshaddox
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

#include "RandomWalk.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>


#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#define PI	3.14159265358979323851280895940618620443274267017841339111328125

//#define Debug_TRS

namespace bsccs {

RandomWalk::RandomWalk(CyclicCoordinateDescent & ccd) {
	MHstep.initialize(ccd);
}


RandomWalk::~RandomWalk() {

}

double RandomWalk::getTransformedTuningValue(double tuningParameter){
	return exp(-tuningParameter);
}
void RandomWalk::sample(Model& model, double tuningParameter, boost::mt19937& rng) {
	cout << "RandomWalk::sample" << endl;

	Parameter & Beta = model.getBeta();
	cout << "Beta is" << endl;
	Beta.logParameter();
	Parameter & Beta_Hat = model.getBeta_Hat();
	cout << "BetaHat is " << endl;
	Beta_Hat.logParameter();
	Eigen::LLT<Eigen::MatrixXf> choleskyEigen = model.getCholeskyLLT();
//	Beta->store();
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

	(choleskyEigen.matrixU()).solveInPlace(b);

	// TODO Check marginal variance on b[i]


	for (int i = 0; i < sizeOfSample; i++) {
		Beta.set(i, b[i] + Beta_Hat.get(i));
	}
	cout << "End of Sample Beta is" << endl;
	Beta.logParameter();

}

bool RandomWalk::evaluateSample(Model& model, double tuningParameter, boost::mt19937& rng, CyclicCoordinateDescent & ccd){
	cout << "RandomWalk::evaluateSample" << endl;

	Parameter & Beta = model.getBeta();
	Parameter & Beta_Hat = model.getBeta_Hat();

	model.setUseHastingsRatio(false);

	bool accept = MHstep.evaluate(model, model.getBeta(), model.getBeta_Hat(), model.getSigmaSquared(), ccd, rng, model.getHessian(), tuningParameter);

	if(accept) {
		cout << "Accepted in evaluate Sample" << endl;
		for (int i = 0; i < Beta.getSize(); i++) {
			Beta_Hat.set(i, Beta.get(i));
		}
	}

	return(accept);
}




}

