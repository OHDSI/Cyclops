/*
 * ccd.h
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#ifndef CCD_H_
#define CCD_H_

#include <time.h>
#include <sys/time.h>

#include "CyclicCoordinateDescent.h"

using namespace bsccs;

struct CCDArguments {

	// Needed for fitting
	std::string inFileName;
	std::string outFileName;
	std::string fileFormat;
	bool useGPU;
	bool useBetterGPU;
	int deviceNumber;
	double tolerance;
	double hyperprior;
	bool useNormalPrior;
	bool hyperPriorSet;
	int maxIterations;
	int convergenceType;
	long seed;

	// Needed to test balance of beta and sigma MCMC tshaddox

	double betaAmount;
	double sigmaAmount;

	// Needed for cross-validation
	bool doCrossValidation;
	double lowerLimit;
	double upperLimit;
	int fold;
	int foldToCompute;
	int gridSteps;
	std::string cvFileName;
	bool doFitAtOptimal;

	// Needed for boot-strapping
	bool doBootstrap;
	bool reportRawEstimates;
	int replicates;
	std::string bsFileName;

	// Needed for model specification
	bool doLogisticRegression;
};


void parseCommandLine(
		int argc,
		char* argv[],
		CCDArguments &arguments);

void parseCommandLine(
		std::vector<std::string>& argcpp,
		CCDArguments& arguments);

double initializeModel(
		InputReader** reader,
		CyclicCoordinateDescent** ccd,
		CCDArguments &arguments);

double fitModel(
		CyclicCoordinateDescent *ccd,
		CCDArguments &arguments);

double runCrossValidation(
		CyclicCoordinateDescent *ccd,
		InputReader *reader,
		CCDArguments &arguments);

double runBoostrap(
		CyclicCoordinateDescent *ccd,
		InputReader *reader,
		CCDArguments &arguments);

double calculateSeconds(
		const struct timeval &time1,
		const struct timeval &time2);

void setDefaultArguments(
		CCDArguments &arguments);

#endif /* CCD_H_ */
