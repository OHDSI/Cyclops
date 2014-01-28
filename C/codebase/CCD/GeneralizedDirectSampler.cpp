/*
 * GeneralizedDirectSampler.cpp
 *
 *  Created on: Jan 27, 2014
 *      Author: tshaddox
 */




#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>
#include <time.h>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <boost/random.hpp>
#include <boost/format.hpp>

#include "GeneralizedDirectSampler.h"
#include "MHRatio.h"
#include "IndependenceSampler.h"
#include "RandomWalk.h"
#include "SigmaSampler.h"

#include "Parameter.h"

#include <boost/random.hpp>

//#define Debug_TRS
//#define DEBUG_STATE

namespace bsccs {


GeneralizedDirectSampler::GeneralizedDirectSampler(InputReader * inReader, std::string GDSFileName): reader(inReader){
	GDSFileNameRoot = GDSFileName;
	nDraws = 20;
	M = 3;
	dsScale = 0.1;

}


GeneralizedDirectSampler::~GeneralizedDirectSampler() {

}

void GeneralizedDirectSampler::initialize(Model & model, CyclicCoordinateDescent& ccd, long int seed) {

	//cout << "MCMCDriver initialize" << endl;
	model.initialize(ccd, seed);

	GDSSampler = new IndependenceSampler(ccd);

	intervalsToReport.initialize(GDSFileNameRoot);
}



void GeneralizedDirectSampler::drive(CyclicCoordinateDescent& ccd, long int seed) {

	cout << "GDSDrive" << endl;
	Model model;
	initialize(model, ccd, seed);

	vector<double> vmValues(M,0);
	vector<double> omegaValues(M, -1);

	mode.set(ccd.hBeta);

	double c_1 = ccd.getLogLikelihood()*ccd.getLogPrior(); // using notation from algorithm...

	double flag = true;

	int counter = 0;
	while(flag){
		cout << "counter = " << counter << endl;
		//Choose proposal distribution
		flag = false;
		for (int i = 0; i < M;i++) {
			GDSSampler->sample(model, dsScale, model.getRng());
			double vm = -GDSSampler->evaluateLogMHRatio(model);
			vmValues[i] = vm;
			cout << "vm = " << vm << endl;
			if (vm < 0){
				flag = true;
				break;
			}
			counter ++;
		}
	}

	cout << "vmVector = <";
	for (int j = 0; j < M; j++){
		cout << vmValues[j] << ", " ;
	}
	cout << ">" << endl;

	std::sort (vmValues.begin(), vmValues.end());

	cout << "vmVector = <";
	for (int j = 0; j < M; j++){
		cout << vmValues[j] << ", " ;
	}
	cout << ">" << endl;

	for (int k = 0; k<(M-1); k++){
		omegaValues[k] = k*(exp(-vmValues[k]) - exp(-vmValues[k+1]));
	}
	omegaValues[M-1] = (M-1)*exp(-vmValues[M-1]);

	cout << "omegaValues = <";
	for (int j = 0; j < M; j++){
		cout << omegaValues[j] << ", " ;
	}
	cout << ">" << endl;

	for (int r = 0; r < nDraws; r ++){
		int j = getMultinomialSample(omegaValues, seed);

		///  HERE
	}

}

int GeneralizedDirectSampler::getMultinomialSample(vector<double>& probabilities, long int seed){
	double sum = 0;
	boost::mt19937 rng(seed);
	// but zeroone makes a copy which means the rng above *never advances*
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	double target = zeroone();
	for (int i = 0; i < M; i++){
		sum += probabilities[i];
		if (sum >= target){
			return(i);
		}
	}

	return(M);

}



double GeneralizedDirectSampler::getTransformedTuningValue(double tuningParameter){
	// TODO Don't forward reference like this.
	return exp(-tuningParameter);
}


}
