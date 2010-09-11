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


struct CCDArguments {

	// Needed for fitting
	std::string inFileName;
	std::string outFileName;
	bool useGPU;
	int deviceNumber;
	double tolerance;
	double hyperprior;
	bool useNormalPrior;
	bool hyperPriorSet;
	int maxIterations;
	int convergenceType;
	long seed;

	// Needed for cross-validation
	bool doCrossValidation;
	double lowerLimit;
	double upperLimit;
	int fold;
	int foldToCompute;
	int gridSteps;

	// Needed for boot-strapping
};

void parseCommandLine(
		int argc,
		char* argv[],
		CCDArguments &arguments);

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

double calculateSeconds(
		const struct timeval &time1,
		const struct timeval &time2);

#endif /* CCD_H_ */
