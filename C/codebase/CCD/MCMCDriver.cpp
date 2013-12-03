/*
 * MCMCDriver.cpp
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */


#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "MCMCDriver.h"
#include "MHRatio.h"
#include "IndependenceSampler.h"
#include "RandomWalk.h"
#include "SigmaSampler.h"

#include "Parameter.h"

#include <boost/random.hpp>

//#define Debug_TRS
//#define DEBUG_STATE

namespace bsccs {


MCMCDriver::MCMCDriver(InputReader * inReader, std::string MCMCFileName): reader(inReader) {
	MCMCFileNameRoot = MCMCFileName;
	thinningValueForWritingToFile = 1000;
	maxIterations = 3000000;
	nBetaSamples = 0;
	nSigmaSquaredSamples = 0;
	acceptanceTuningParameter = 0; // exp(acceptanceTuningParameter) modifies
	acceptanceRatioTarget = 0.30;
	autoAdapt = false;
}



MCMCDriver::~MCMCDriver() {

}

vector<double> storedBetaHat;

void checkValidState(CyclicCoordinateDescent& ccd, Model& model, Parameter& Beta,
		Parameter& Beta_Hat,
		Parameter& SigmaSquared) {

	cout << "Check Valid State" << endl;
	ccd.setBeta(Beta.returnCurrentValues());
	double logLike = ccd.getLogLikelihood();
	double logPrior = ccd.getLogPrior();
	double storedLogLike =  model.getLogLikelihood();
	double storedLogPrior = model.getLogPrior();
	if (abs(logLike - storedLogLike) > 0.000001) {
		cerr << "\n\n\n \t\tError in internal state of beta/log_likelihood." << endl;
		cerr << std::setprecision(15) << "\tStored value: " << storedLogLike << endl;
		cerr << std::setprecision(15) << "\tRecomp value: " << logLike << endl;
		exit(-1);
	} else {
		cerr << "All fine - likelihood" << endl;
	}

	if (abs(logPrior - storedLogPrior) > 0.000001) {
		cerr << "\n\n\n \t\tError in internal state of beta/log_prior." << endl;
		cerr << std::setprecision(15) << "\tStored value: " << storedLogLike << endl;
		cerr << std::setprecision(15) << "\tRecomp value: " << logLike << endl;
		exit(-1);
	} else {
		cerr << "All fine - prior" << endl;
	}


	if (storedBetaHat.size() == 0) { // first time through
		for (int i = 0; i < Beta_Hat.getSize(); ++i) {
			storedBetaHat.push_back(Beta_Hat.get(i));
		}

	} else {
		for (int i = 0; i < Beta_Hat.getSize(); ++i) {
			if (abs(storedBetaHat[i] - Beta_Hat.get(i)) > 0.0001) {
	//			cerr << "Beta hat has changed!" << endl;
	//			exit(-1);
			}
		}
	}

	// TODO Check internals with sigma
}

void MCMCDriver::initialize(double betaAmount, Model & model, CyclicCoordinateDescent& ccd, long int seed) {

	//cout << "MCMCDriver initialize" << endl;
	model.initialize(ccd, seed);


	transitionKernelSelectionProb.push_back(betaAmount);
	transitionKernelSelectionProb.push_back(1.0 - betaAmount);

	transitionKernels.push_back(new IndependenceSampler(ccd));
	transitionKernels.push_back(new SigmaSampler);
	intervalsToReport.initialize(MCMCFileNameRoot);


}

void MCMCDriver::logState(Model & model, int iteration){
	//cout << "\n MCMCDriver::logState" << endl;
	//MCMCResults_SigmaSquared.push_back(model.getSigmaSquared().returnCurrentValues()[0]);
	//model.getSigmaSquared().logParameter();
	//MCMCResults_BetaVectors.push_back(model.getBeta().returnCurrentValues());
	//model.getBeta().logParameter();
	double loglikelihoodHere = model.getLogLikelihood();
	//cerr << "loglikelihood = " << loglikelihoodHere << endl;
	double logPriorHere = model.getLogPrior();
	//cerr << "logPrior = " << logPriorHere << endl;
	MCMCResults_loglikelihoods.push_back(model.getLogLikelihood());
	//cout << "MCMCDriver::logState end" << endl;
	if (iteration % thinningValueForWritingToFile == 0){
		//cout << "logging" << endl;
		intervalsToReport.fileLogCredibleIntervals(model.getLogLikelihood(), &(model.getBeta().returnCurrentValues()), model.getSigmaSquared().returnCurrentValues()[0], iteration);
	}
}

int MCMCDriver::findTransitionKernelIndex(double uniformRandom, vector<double>& transitionKernelSelectionProb){
	//cout << "\t MCMCDriver::findTransitionKernalIndex" << endl;
	int length = transitionKernelSelectionProb.size();
	double currentTotal = 0;
	for (int i = 0; i < length; i++){
		currentTotal += transitionKernelSelectionProb[i];
		if (uniformRandom <= currentTotal){
			cout << "\t\t Picking Kernel " << i << endl;
			return(i);
		}
	}
}

void MCMCDriver::drive(
		CyclicCoordinateDescent& ccd, double betaAmount, long int seed) {

	Model model;
	initialize(betaAmount, model, ccd, seed);
	logState(model,0);
	//Set Boost rng

	model.writeVariances();
	logState(model,0);
	model.restore();
	logState(model,0);

	double acceptNumber = 0.0;
	double countIndependenceSampler = 0.0;

	//MCMC Loop
	for (int iterations = 0; iterations < maxIterations; iterations ++) {
		cout << endl << "MCMC iteration " << iterations << endl;

#ifdef DEBUG_STATE
		checkValidState(ccd, MHstep, Beta, Beta_Hat, SigmaSquared);
#endif

		// Sample from a uniform distribution
		static boost::uniform_01<boost::mt19937> zeroone(model.getRng());
		double uniformRandom = zeroone();

		int transitionKernelIndex = findTransitionKernelIndex(uniformRandom, transitionKernelSelectionProb);
		transitionKernels[transitionKernelIndex]->sample(model, acceptanceTuningParameter, model.getRng());

		bool accept = transitionKernels[transitionKernelIndex]->evaluateSample(model, acceptanceTuningParameter, model.getRng(), ccd);
		if (transitionKernelIndex == 0){
			acceptNumber = acceptNumber + accept;
			countIndependenceSampler = countIndependenceSampler + 1.0;
		}

		//cout << "**********  WARNING  ************" << endl;
		//accept = true;
		cout << "accept = " << accept << endl;

		if (accept) {
			cout << "\t\t\t\t #######Accept#################" << endl;
			model.acceptChanges();
		} else {
			cout << "\t\t\t\t #######Reject#################" << endl;
			model.restore();
		}
		logState(model, iterations);
		//checkValidState(ccd, model, model.getBeta(), model.getBeta_Hat(), model.getSigmaSquared());
	}
	cout << "acceptances% = " << acceptNumber/countIndependenceSampler + 0.0 << endl;
}

double MCMCDriver::coolingTransform(int x) {
//	return std::log(x);
	return std::sqrt(x);
//	return static_cast<double>(x);
}

double MCMCDriver::targetTransform(double alpha, double target) {
	return (alpha - target);
}

void MCMCDriver::adaptiveKernel(int numberIterations, double alpha) {

	acceptanceTuningParameter = acceptanceTuningParameter +
			(1.0 / (1.0 + coolingTransform(numberIterations))) *
			targetTransform(alpha, acceptanceRatioTarget);
//			(0.4 - std::abs(alpha - acceptanceRatioTarget));

//	double delta;
//	if (alpha < 0.2 || alpha > 0.8) {
//		delta -= 1.0;
//	} else {
//		delta += 1.0;
//	}
//	acceptanceTuningParameter += (1.0 / (1.0 + coolingTransform(numberIterations))) * delta;
}




double MCMCDriver::getTransformedTuningValue(double tuningParameter) {
	return exp(-tuningParameter);
}

}
