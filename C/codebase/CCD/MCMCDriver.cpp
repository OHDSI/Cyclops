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
#include "SigmaSampler.h"
#include "CredibleIntervals.h"
#include "Parameter.h"

#include <boost/random.hpp>

//#define Debug_TRS
//#define DEBUG_STATE

namespace bsccs {


MCMCDriver::MCMCDriver(InputReader * inReader, std::string MCMCFileName): reader(inReader) {
	MCMCFileNameRoot = MCMCFileName;
	maxIterations = 10;
	nBetaSamples = 0;
	nSigmaSquaredSamples = 0;
	acceptanceTuningParameter = 3; // exp(acceptanceTuningParameter) modifies
	acceptanceRatioTarget = 0.30;
	autoAdapt = false;
}



MCMCDriver::~MCMCDriver() {

}

vector<double> storedBetaHat;

void checkValidState(CyclicCoordinateDescent& ccd, MHRatio& MHstep, Parameter& Beta,
		Parameter& Beta_Hat,
		Parameter& SigmaSquared) {
	ccd.setBeta(Beta.returnCurrentValues());
	double logLike = ccd.getLogLikelihood();
	double storedLogLike =  MHstep.getStoredLogLikelihood();
	if (logLike != storedLogLike) {
		cerr << "Error in internal state of beta/log_likelihood." << endl;
		cerr << "\tStored value: " << storedLogLike << endl;
		cerr << "\tRecomp value: " << logLike << endl;
		exit(-1);
	} else {
		cerr << "All fine" << endl;
	}

	if (storedBetaHat.size() == 0) { // first time through
		for (int i = 0; i < Beta_Hat.getSize(); ++i) {
			storedBetaHat.push_back(Beta_Hat.get(i));
		}

	} else {
		for (int i = 0; i < Beta_Hat.getSize(); ++i) {
			if (storedBetaHat[i] != Beta_Hat.get(i)) {
				cerr << "Beta hat has changed!" << endl;
				exit(-1);
			}
		}
	}

	// TODO Check internals with sigma
}

void MCMCDriver::initialize(double betaAmount, Model & model, CyclicCoordinateDescent& ccd) {
	// MAS All initialization


	cout << "MCMCDriver initialize" << endl;
	model.initialize(ccd);

	transitionKernelSelectionProb.push_back(betaAmount);
	transitionKernelSelectionProb.push_back(1.0);

	transitionKernels.push_back(new IndependenceSampler);
	transitionKernels.push_back(new SigmaSampler);



}

void MCMCDriver::logState(Model & model){
	cout << "\n MCMCDriver::logState" << endl;
	MCMCResults_SigmaSquared.push_back(model.getSigmaSquared().returnCurrentValues()[0]);
	model.getSigmaSquared().logParameter();
	MCMCResults_BetaVectors.push_back(model.getBeta().returnCurrentValues());
	model.getBeta().logParameter();

}

int MCMCDriver::findTransitionKernelIndex(double uniformRandom, vector<double>& transitionKernelSelectionProb){
	cout << "\t MCMCDriver::findTransitionKernalIndex" << endl;
	int length = transitionKernelSelectionProb.size();
	for (int i = 0; i < length; i++){
		if (uniformRandom <= transitionKernelSelectionProb[i]){
			cout << "\t\t Picking Kernel " << i << endl;
			return(i);
		}
	}

}

void MCMCDriver::drive(
		CyclicCoordinateDescent& ccd, double betaAmount, long int seed) {

	Model model;
	initialize(betaAmount, model, ccd);
	logState(model);

	//Set Boost rng
	boost::mt19937 rng(seed);


	//MCMC Loop
	for (int iterations = 0; iterations < maxIterations; iterations ++) {

		cout << endl << "MCMC iteration " << iterations << endl;

#ifdef DEBUG_STATE
		checkValidState(ccd, MHstep, Beta, Beta_Hat, SigmaSquared);
#endif

		// Store values
		//Beta.store();
		//SigmaSquared.store();

		static boost::uniform_01<boost::mt19937> zeroone(rng);

		// Sample from a uniform distribution
		double uniformRandom = zeroone();

		int transitionKernelIndex = findTransitionKernelIndex(uniformRandom, transitionKernelSelectionProb);
		TransitionKernel* currentTransitionKernel = transitionKernels[transitionKernelIndex];

		transitionKernels[transitionKernelIndex]->sample(model, acceptanceTuningParameter, rng);

		model.logState();

		bool accept = transitionKernels[transitionKernelIndex]->evaluateSample(model, acceptanceTuningParameter, rng, ccd);

		cout << "accept = " << accept << endl;

		if (accept) {
			//model.keepCurrentState
		} else {
			model.restore();
		}
		//Select a sample beta vector

		/*
		if (betaAmount > uniformRandom) {
			getBeta ++;


			independenceSamplerInstance.sample(&Beta_Hat, &Beta, rng, CholDecom, acceptanceTuningParameter);


			cout << "acceptanceTuningParameter = " <<  acceptanceTuningParameter << endl;

			//Compute the acceptance ratio
			alpha = MHstep.evaluate(&Beta, &Beta_Hat, &SigmaSquared, ccd, rng,
					HessianMatrix, acceptanceTuningParameter);
			cout << "alpha = " << alpha << endl;

			MCMCResults_BetaVectors.push_back(Beta.returnCurrentValues());
			nBetaSamples ++;

			if (Beta.getChangeStatus()){
				numberAcceptances ++;
			}

			if (autoAdapt) {
				adaptiveKernel(iterations,alpha);
			}

		}

		if (Beta.getProbabilityUpdate() < uniformRandom) {
			getSigma ++;
			SigmaSampler sigmaMaker;
			sigmaMaker.sampleSigma(&SigmaSquared, &Beta, rng);

			MCMCResults_SigmaSquared.push_back(SigmaSquared.returnCurrentValues()[0]);
			nSigmaSquaredSamples ++;

			// TODO Need Wrapper for this....
			ccd.resetBeta();
			ccd.setHyperprior(SigmaSquared.get(0));
			int ZHANG_OLES = 1;
			int ccdIterations = 100;
			double tolerance = 5E-4;

			ccd.update(ccdIterations, ZHANG_OLES, tolerance);
			clearHessian();
			ccd.getHessian(&hessian);
			generateCholesky();
			Beta_Hat.set(ccd.hBeta);
		}


//#ifdef DEBUG_STATE
//		checkValidState(ccd, MHstep, Beta, Beta_Hat, SigmaSquared);
		cerr << "acceptance rate: " << ( static_cast<double>(numberAcceptances)
				/ static_cast<double>(iterations)) << endl;
//#endif

		// End MCMC loop
	}

		cout << "getBeta = " << getBeta << endl;
		cout << "getSigma = " << getSigma << endl;
		cout << "number of acceptances = " << numberAcceptances << endl;
		cout << "Starting Credible Intervals" << endl;

		cout << "at End, nBetaSamples = " << nBetaSamples << endl;
		cout << "at End, nSigmaSquaredSamples = " << nSigmaSquaredSamples << endl;

		CredibleIntervals intervalsToReport;
		intervalsToReport.computeCredibleIntervals(&MCMCResults_BetaVectors, &MCMCResults_SigmaSquared, Beta.getProbabilityUpdate(), SigmaSquared.getProbabilityUpdate(), MCMCFileNameRoot);
*/

		logState(model); }

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
