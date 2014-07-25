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
#include "ModelSampler.h"
#include "RandomWalk.h"
#include "SigmaSampler.h"

#include "Parameter.h"

#define Debug_TRS
//#define DEBUG_STATE

namespace bsccs {

//blah
MCMCDriver::MCMCDriver(std::string MCMCFileName) {
	MCMCFileNameRoot = MCMCFileName;
	thinningValueForWritingToFile = 1;
	maxIterations = 1000;
	nBetaSamples = 0;
	nSigmaSquaredSamples = 0;
	acceptanceTuningParameter = 0; // exp(acceptanceTuningParameter) modifies
	acceptanceRatioTarget = 0.30;
	autoAdapt = false;
}



MCMCDriver::~MCMCDriver() {

}

vector<double> storedBetaHat;

void checkValidState(CyclicCoordinateDescent& ccd, MCMCModel& model, Parameter& Beta,
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

void MCMCDriver::initialize(double betaAmount, CyclicCoordinateDescent& ccd) {


	transitionKernelSelectionProb.push_back(betaAmount);

	transitionKernelSelectionProb.push_back(1.0 - betaAmount);

	transitionKernels.push_back(new IndependenceSampler(ccd));
	transitionKernels.push_back(new SigmaSampler);
	intervalsToReport.initialize(MCMCFileNameRoot);

}

void MCMCDriver::logState(MCMCModel & model, int iteration, double logProbability){

	//MCMCResults_SigmaSquared.push_back(model.getSigmaSquared().returnCurrentValues()[0]);
	//model.getSigmaSquared().logParameter();
	//MCMCResults_BetaVectors.push_back(model.getBeta().returnCurrentValues());
	//model.getBeta().logParameter();
	double loglikelihoodHere = model.getLogLikelihood();
	//cerr << "loglikelihood = " << loglikelihoodHere << endl;
	double logPriorHere = model.getLogPrior();
	//cerr << "logPrior = " << logPriorHere << endl;
	MCMCResults_loglikelihoods.push_back(model.getLogLikelihood());



	if (iteration % thinningValueForWritingToFile == 0){
		double uniformRandom = rand() / ((double) RAND_MAX);
		if (logProbability > uniformRandom) {
			//intervalsToReport.fileLogCredibleIntervals(model.getLogLikelihood(), &(model.getBeta().returnCurrentValues()), model.getSigmaSquared().returnCurrentValues()[0], iteration);
			intervalsToReport.fileLogCredibleIntervals(model.getLogLikelihood(), model.getBeta(), model.getSigmaSquared(), iteration);
		}
	}
}

int MCMCDriver::findTransitionKernelIndex(double uniformRandom, vector<double>& transitionKernelSelectionProb){
	int length = transitionKernelSelectionProb.size();
	double currentTotal = 0;
	for (int i = 0; i < length; i++){
		currentTotal += transitionKernelSelectionProb[i];
		if (uniformRandom <= currentTotal){
			return(i);
		}
	}
}

void MCMCDriver::drive(
		MCMCModel& model, CyclicCoordinateDescent& ccd, double betaAmount, long int seed, double logProbability) {

	//MCMCModel model;

	//initialize(betaAmount, ccd);

	logState(model,0,1.0);

	logState(model,0,1.0);

	model.restore();
	logState(model,0,1.0);

	srand(seed);

	std::default_random_engine generator(seed);


	double acceptNumber = 0.0;
	double countIndependenceSampler = 0.0;

	//MCMC Loop
	for (int iterations = 0; iterations < maxIterations; iterations ++) {
		//cout << endl << "MCMC iteration " << iterations << endl;

#ifdef DEBUG_STATE
		checkValidState(ccd, MHstep, Beta, Beta_Hat, SigmaSquared);
#endif

		// Sample from a uniform distribution
		double uniformRandom = rand() / ((double) RAND_MAX);


		model.store();
		int transitionKernelIndex = findTransitionKernelIndex(uniformRandom, transitionKernelSelectionProb);
		transitionKernels[transitionKernelIndex]->sample(model, acceptanceTuningParameter,generator);
		bool accept = transitionKernels[transitionKernelIndex]->evaluateSample(model, acceptanceTuningParameter, ccd);
		//cout << "**********  WARNING  ************" << endl;
		//accept = true;

		if (transitionKernelIndex == 0){
			acceptNumber = acceptNumber + accept;
			countIndependenceSampler = countIndependenceSampler + 1.0;
		}
		/////  need a model set to store or something like that...
		if (accept) {
			model.acceptChanges();
		} else {
			model.restore();
		}
		logState(model, iterations,logProbability);
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
