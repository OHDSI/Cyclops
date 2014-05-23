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
void RandomWalk::sample(MCMCModel& model, double tuningParameter) {
	cout << "RandomWalk::sample" << endl;

	BetaParameter & Beta = model.getBeta();
	cout << "Beta is" << endl;
	Beta.logParameter();
	BetaParameter & Beta_Hat = model.getBeta_Hat();
	cout << "BetaHat is " << endl;
	Beta_Hat.logParameter();
	Eigen::LLT<Eigen::MatrixXf> choleskyEigen = model.getCholeskyLLT();
//	Beta->store();
	int sizeOfSample = Beta.getSize();


	Eigen::VectorXf independentNormal = Eigen::VectorXf::Random(sizeOfSample);
	for (int i = 0; i < sizeOfSample; i++) {
		bsccs::real normalValue = generateGaussian();
		// NB: tuningParameter scales the VARIANCE
		independentNormal[i] = normalValue * std::sqrt(getTransformedTuningValue(tuningParameter)); // multiply by stdev
	}


#ifdef Debug_TRS
	cout << "Cholesky in Sampler " << endl;
	Eigen::MatrixXf CholeskyDecompL(sizeOfSample, sizeOfSample);
	CholeskyDecompL = choleskyEigen.matrixU();
	cout << CholeskyDecompL << endl;
#endif

	(choleskyEigen.matrixU()).solveInPlace(independentNormal);

	// TODO Check marginal variance on b[i]


	for (int i = 0; i < sizeOfSample; i++) {
		Beta.set(i, independentNormal[i] + Beta_Hat.get(i));
	}
	cout << "End of Sample Beta is" << endl;
	Beta.logParameter();

}

bool RandomWalk::evaluateSample(MCMCModel& model, double tuningParameter, CyclicCoordinateDescent & ccd){
	cout << "RandomWalk::evaluateSample" << endl;

	BetaParameter & Beta = model.getBeta();
	BetaParameter & Beta_Hat = model.getBeta_Hat();

	model.setUseHastingsRatio(false);

	bool accept = MHstep.evaluate(model);

	if(accept) {
		cout << "Accepted in evaluate Sample" << endl;
		for (int i = 0; i < Beta.getSize(); i++) {
			Beta_Hat.set(i, Beta.get(i));
		}
	}

	return(accept);
}




}

