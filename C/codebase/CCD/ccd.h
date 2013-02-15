/*
 * ccd.h
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#ifndef CCD_H_
#define CCD_H_

#include <time.h>

#ifdef _WIN32
#include <stddef.h>
#include <io.h>
#include <stdlib.h>
#include <winsock.h>
#include <stdio.h>
#else
#include <sys/time.h>
#endif

#include "ModelSpecifics.h"
#include "CyclicCoordinateDescent.h"

struct CCDArguments {

	// Needed for fitting
	std::string inFileName;
	std::string outFileName;
	std::string fileFormat;
	std::string outputFormat;
	bool useGPU;
	bool useBetterGPU;
	int deviceNumber;
	double tolerance;
	double hyperprior;
	bool useNormalPrior;
	bool hyperPriorSet;
	int maxIterations;
	std::string convergenceTypeString;
	int convergenceType;
	long seed;

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
	bool doPartial;

	// Needed for model specification
//	bool doLogisticRegression;
	int modelType;
	std::string modelName;
};


void parseCommandLine(
		int argc,
		char* argv[],
		CCDArguments &arguments);

void parseCommandLine(
		std::vector<std::string>& argcpp,
		CCDArguments& arguments);

double initializeModel(
		ModelData** modelData,
		CyclicCoordinateDescent** ccd,
//		ModelSpecifics<DefaultModel>** model,
		AbstractModelSpecifics** model,
		CCDArguments &arguments);

double fitModel(
		CyclicCoordinateDescent *ccd,
		CCDArguments &arguments);

double predictModel(
		CyclicCoordinateDescent *ccd,
		ModelData *modelData,
		CCDArguments &arguments);

double runCrossValidation(
		CyclicCoordinateDescent *ccd,
		ModelData *modelData,
		CCDArguments &arguments);

double runBoostrap(
		CyclicCoordinateDescent *ccd,
		ModelData *modelData,
		CCDArguments &arguments);

double calculateSeconds(
		const struct timeval &time1,
		const struct timeval &time2);

void setDefaultArguments(
		CCDArguments &arguments);

#endif /* CCD_H_ */
