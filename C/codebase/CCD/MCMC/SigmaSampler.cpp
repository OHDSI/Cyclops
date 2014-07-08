/*
 * SigmaCalculate.cpp
 *
 *  Created on: Aug 9, 2012
 *      Author: trevorshaddox
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>

#include "SigmaSampler.h"
#include <random>

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;


namespace bsccs {


	SigmaSampler::SigmaSampler(){


	}

	SigmaSampler::~SigmaSampler(){

	}

	void SigmaSampler::sample(MCMCModel& model, double tuningParameter,  std::default_random_engine& generator){
		//cout << "SigmaSampler::sample" << endl;

		// tau | BetaVector ~ gamma(alpha + N/2, Beta + (1/2)(SUM(beta_i - mu)^2)
		// prior: tau ~ gamma(alpha, beta)

		BetaParameter& BetaValues = model.getBeta();
		HyperpriorParameter& SigmaSquared = model.getSigmaSquared();

		//SigmaSquared.store();

		//SigmaSquared.setRestorable(true);

		double SigmaParameter_alpha = 2;
		double SigmaParameter_beta = 8;

		double BetaMinusMu = 0;
		double Mu = 0;

		for (int i = 0; i < BetaValues.getSize(); i++) {
			Mu += BetaValues.get(i) / BetaValues.getSize();
		}

		for (int j = 0; j < BetaValues.getSize(); j++ ) {
			BetaMinusMu += (BetaValues.get(j) - Mu)*(BetaValues.get(j) - Mu); //
		}

		const double shape = SigmaParameter_alpha + BetaValues.getSize() / 2;
		double scale = SigmaParameter_beta + BetaMinusMu / 2;

		///  Check scale
		//std::default_random_engine generator2;
		std::gamma_distribution<double> distribution(shape,scale);

		double newValue = distribution(generator);


		//SigmaSquared.logParameter();
		SigmaSquared.set(0, newValue);
	}

	bool SigmaSampler::evaluateSample(MCMCModel& model, double tuningParameter, CyclicCoordinateDescent & ccd){
		//cout << "SigmaSampler::evaluateSample" << endl;
		model.resetWithNewSigma();
		return(true); // Gibbs step always accepts
	}



}


