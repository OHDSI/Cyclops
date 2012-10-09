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

struct ImputeArguments{
	// Needed for doing multiple imputation
	bool doImputation;
	int numberOfImputations;
	bool includeY;
};

void parseCommandLine(
		int argc,
		char* argv[],
		CCDArguments &ccdArgs,
		ImputeArguments &imputeArgs);

void parseCommandLine(
		std::vector<std::string>& argcpp,
		CCDArguments& ccdArgs,
		ImputeArguments& imputeArgs);

void setDefaultArguments(
		CCDArguments &ccdArgs,
		ImputeArguments &imputeArgs);

#endif /* CCD_H_ */
