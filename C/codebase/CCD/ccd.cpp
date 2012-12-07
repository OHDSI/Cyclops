/*
 * ccd.cpp
 *
 *  Created on: July, 2010
 *      Author: msuchard
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>

#include <iostream>
#include <time.h>
#ifndef _WIN32
	#include <sys/time.h>
#endif

#include <math.h>

#include "ccd.h"
#include "CyclicCoordinateDescent.h"
#include "ModelData.h"
#include "io/InputReader.h"
#include "io/CLRInputReader.h"
#include "io/RTestInputReader.h"
#include "io/CCTestInputReader.h"
#include "io/CoxInputReader.h"
#include "io/NewCLRInputReader.h"
#include "io/NewSCCSInputReader.h"
#include "io/NewCoxInputReader.h"
#include "io/BBRInputReader.h"
#include "CrossValidationSelector.h"
#include "CrossValidationDriver.h"
#include "BootstrapSelector.h"
#include "ProportionSelector.h"
#include "BootstrapDriver.h"
#include "ModelSpecifics.h"

#include "tclap/CmdLine.h"

//#include <R.h>

#ifdef CUDA
	#include "GPUCyclicCoordinateDescent.h"
//	#include "BetterGPU.h"
#endif


#define NEW

using namespace TCLAP;
using namespace std;

//Sushil:Implementing gettimeofday functionality for windows.
#ifdef _WIN32
	#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
		#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
	#else
		#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
	#endif
	struct timezone
	{
		int  tz_minuteswest; /* minutes W of Greenwich */
		int  tz_dsttime;     /* type of dst correction */
	};

	// Definition of a gettimeofday function

	int gettimeofday(struct timeval *tv, struct timezone *tz)
	{
		// Define a structure to receive the current Windows filetime
		FILETIME ft;

		// Initialize the present time to 0 and the timezone to UTC
		unsigned __int64 tmpres = 0;
		static int tzflag = 0;

		if (NULL != tv)
		{
			GetSystemTimeAsFileTime(&ft);

			// The GetSystemTimeAsFileTime returns the number of 100 nanosecond
			// intervals since Jan 1, 1601 in a structure. Copy the high bits to
			// the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
			tmpres |= ft.dwHighDateTime;
			tmpres <<= 32;
			tmpres |= ft.dwLowDateTime;

			// Convert to microseconds by dividing by 10
			tmpres /= 10;

			// The Unix epoch starts on Jan 1 1970.  Need to subtract the difference
			// in seconds from Jan 1 1601.
			tmpres -= DELTA_EPOCH_IN_MICROSECS;

			// Finally change microseconds to seconds and place in the seconds value.
			// The modulus picks up the microseconds.
			tv->tv_sec = (long)(tmpres / 1000000UL);
			tv->tv_usec = (long)(tmpres % 1000000UL);
		}

		if (NULL != tz)
		{
			if (!tzflag)
			{
				_tzset();
				tzflag++;
			}

			// Adjust for the timezone west of Greenwich
			tz->tz_minuteswest = _timezone / 60;
			tz->tz_dsttime = _daylight;
		}

		return 0;
	}
#endif

double calculateSeconds(const timeval &time1, const timeval &time2) {
	return time2.tv_sec - time1.tv_sec +
			(double)(time2.tv_usec - time1.tv_usec) / 1000000.0;
}

void parseCommandLine(int argc, char* argv[],
		CCDArguments &arguments) {
	std::vector<std::string> args;
	for (int i = 0; i < argc; i++)
		args.push_back(argv[i]);
	parseCommandLine(args, arguments);
}

void setDefaultArguments(CCDArguments &arguments) {
	arguments.useGPU = false;
	arguments.maxIterations = 100;
	arguments.inFileName = "default_in";
	arguments.outFileName = "default_out";
	arguments.hyperPriorSet = false;
	arguments.hyperprior = 1.0;
	arguments.tolerance = 5E-4;
	arguments.seed = 123;
	arguments.doCrossValidation = false;
	arguments.lowerLimit = 0.01;
	arguments.upperLimit = 20.0;
	arguments.fold = 10;
	arguments.gridSteps = 10;
	arguments.cvFileName = "cv.txt";
	arguments.doBootstrap = false;
	arguments.replicates = 100;
	arguments.reportRawEstimates = false;
//	arguments.doLogisticRegression = false;
	arguments.modelName = "sccs";
	arguments.fileFormat = "sccs";
	arguments.useNormalPrior = false;
	arguments.convergenceType = GRADIENT;
	arguments.convergenceTypeString = "gradient";
	arguments.doPartial = false;
}


void parseCommandLine(std::vector<std::string>& args,
		CCDArguments &arguments) {

	setDefaultArguments(arguments);

	try {
		CmdLine cmd("Cyclic coordinate descent algorithm for self-controlled case studies", ' ', "0.1");
		ValueArg<int> gpuArg("g","GPU","Use GPU device", arguments.useGPU, -1, "device #");
//		SwitchArg betterGPUArg("1","better", "Use better GPU implementation", false);
		ValueArg<int> maxIterationsArg("", "maxIterations", "Maximum iterations", false, arguments.maxIterations, "int");
		UnlabeledValueArg<string> inFileArg("inFileName","Input file name", true, arguments.inFileName, "inFileName");
		UnlabeledValueArg<string> outFileArg("outFileName","Output file name", true, arguments.outFileName, "outFileName");

		// Prior arguments
		ValueArg<double> hyperPriorArg("v", "variance", "Hyperprior variance", false, arguments.hyperprior, "real");
		SwitchArg normalPriorArg("n", "normalPrior", "Use normal prior, default is laplace", arguments.useNormalPrior);

		// Convergence criterion arguments
		ValueArg<double> toleranceArg("t", "tolerance", "Convergence criterion tolerance", false, arguments.tolerance, "real");
//		SwitchArg zhangOlesConvergenceArg("z", "zhangOles", "Use Zhange-Oles convergence criterion, default is true", true);
		std::vector<std::string> allowedConvergence;
		allowedConvergence.push_back("gradient");
		allowedConvergence.push_back("ZhangOles");
		allowedConvergence.push_back("Lange");
		allowedConvergence.push_back("Mittal");
		ValuesConstraint<std::string> allowedConvergenceValues(allowedConvergence);
		ValueArg<string> convergenceArg("", "convergence", "Convergence criterion", false, arguments.convergenceTypeString, &allowedConvergenceValues);

		ValueArg<long> seedArg("s", "seed", "Random number generator seed", false, arguments.seed, "long");

		// Cross-validation arguments
		SwitchArg doCVArg("c", "cv", "Perform cross-validation selection of hyperprior variance", arguments.doCrossValidation);
		ValueArg<double> lowerCVArg("l", "lower", "Lower limit for cross-validation search", false, arguments.lowerLimit, "real");
		ValueArg<double> upperCVArg("u", "upper", "Upper limit for cross-validation search", false, arguments.upperLimit, "real");
		ValueArg<int> foldCVArg("f", "fold", "Fold level for cross-validation", false, arguments.fold, "int");
		ValueArg<int> gridCVArg("", "gridSize", "Uniform grid size for cross-validation search", false, arguments.gridSteps, "int");
		ValueArg<int> foldToComputeCVArg("", "computeFold", "Number of fold to iterate, default is 'fold' value", false, 10, "int");
		ValueArg<string> outFile2Arg("", "cvFileName", "Cross-validation output file name", false, arguments.cvFileName, "cvFileName");

		// Bootstrap arguments
		SwitchArg doBootstrapArg("b", "bs", "Perform bootstrap estimation", arguments.doBootstrap);
//		ValueArg<string> bsOutFileArg("", "bsFileName", "Bootstrap output file name", false, "bs.txt", "bsFileName");
		ValueArg<int> replicatesArg("r", "replicates", "Number of bootstrap replicates", false, arguments.replicates, "int");
		SwitchArg reportRawEstimatesArg("","raw", "Report the raw bootstrap estimates", arguments.reportRawEstimates);
		ValueArg<int> partialArg("", "partial", "Number of rows to use in partial estimation", false, -1, "int");

		// Model arguments
//		SwitchArg doLogisticRegressionArg("", "logistic", "Use ordinary logistic regression", arguments.doLogisticRegression);
		std::vector<std::string> allowedModels;
		allowedModels.push_back("sccs");
		allowedModels.push_back("clr");
		allowedModels.push_back("lr");
		allowedModels.push_back("ls");
		allowedModels.push_back("cox");
		ValuesConstraint<std::string> allowedModelValues(allowedModels);
		ValueArg<string> modelArg("", "model", "Model specification", false, arguments.modelName, &allowedModelValues);

		// Format arguments
		std::vector<std::string> allowedFormats;
		allowedFormats.push_back("sccs");
		allowedFormats.push_back("clr");
		allowedFormats.push_back("csv");
		allowedFormats.push_back("cc");
		allowedFormats.push_back("cox-csv");
		allowedFormats.push_back("new-cox");
		allowedFormats.push_back("bbr");
		allowedFormats.push_back("generic");
		ValuesConstraint<std::string> allowedFormatValues(allowedFormats);
		ValueArg<string> formatArg("", "format", "Format of data file", false, arguments.fileFormat, &allowedFormatValues);

		cmd.add(gpuArg);
//		cmd.add(betterGPUArg);
		cmd.add(toleranceArg);
		cmd.add(maxIterationsArg);
		cmd.add(hyperPriorArg);
		cmd.add(normalPriorArg);
//		cmd.add(zhangOlesConvergenceArg);
		cmd.add(convergenceArg);
		cmd.add(seedArg);
		cmd.add(modelArg);
		cmd.add(formatArg);

		cmd.add(doCVArg);
		cmd.add(lowerCVArg);
		cmd.add(upperCVArg);
		cmd.add(foldCVArg);
		cmd.add(gridCVArg);
		cmd.add(foldToComputeCVArg);
		cmd.add(outFile2Arg);

		cmd.add(doBootstrapArg);
//		cmd.add(bsOutFileArg);
		cmd.add(replicatesArg);
		cmd.add(partialArg);
		cmd.add(reportRawEstimatesArg);
//		cmd.add(doLogisticRegressionArg);

		cmd.add(inFileArg);
		cmd.add(outFileArg);
		cmd.parse(args);

		if (gpuArg.getValue() > -1) {
			arguments.useGPU = true;
			arguments.deviceNumber = gpuArg.getValue();
		} else {
			arguments.useGPU = false;
		}
//		arguments.useBetterGPU = betterGPUArg.isSet();

		arguments.inFileName = inFileArg.getValue();
		arguments.outFileName = outFileArg.getValue();
		arguments.tolerance = toleranceArg.getValue();
		arguments.maxIterations = maxIterationsArg.getValue();
		arguments.hyperprior = hyperPriorArg.getValue();
		arguments.useNormalPrior = normalPriorArg.getValue();
		arguments.seed = seedArg.getValue();

		arguments.modelName = modelArg.getValue();
		arguments.fileFormat = formatArg.getValue();
		arguments.convergenceTypeString = convergenceArg.getValue();

		if (hyperPriorArg.isSet()) {
			arguments.hyperPriorSet = true;
		} else {
			arguments.hyperPriorSet = false;
		}

//		if (zhangOlesConvergenceArg.isSet()) {
//			arguments.convergenceType = ZHANG_OLES;
//		} else {
//			arguments.convergenceType = LANGE;
//		}
		if (arguments.convergenceTypeString == "ZhangOles") {
			arguments.convergenceType = ZHANG_OLES;
		} else if (arguments.convergenceTypeString == "Lange") {
			arguments.convergenceType = LANGE;
		} else if (arguments.convergenceTypeString == "Mittal") {
			arguments.convergenceType = MITTAL;
		} else if (arguments.convergenceTypeString == "gradient") {
			arguments.convergenceType = GRADIENT;
		} else {
			cerr << "Unknown convergence type: " << convergenceArg.getValue() << " " << arguments.convergenceTypeString << endl;
			exit(-1);
		}

		// Cross-validation
		arguments.doCrossValidation = doCVArg.isSet();
		if (arguments.doCrossValidation) {
			arguments.lowerLimit = lowerCVArg.getValue();
			arguments.upperLimit = upperCVArg.getValue();
			arguments.fold = foldCVArg.getValue();
			arguments.gridSteps = gridCVArg.getValue();
			if(foldToComputeCVArg.isSet()) {
				arguments.foldToCompute = foldToComputeCVArg.getValue();
			} else {
				arguments.foldToCompute = arguments.fold;
			}
			arguments.cvFileName = outFile2Arg.getValue();
			arguments.doFitAtOptimal = true;
		}

		// Bootstrap
		arguments.doBootstrap = doBootstrapArg.isSet();
		if (arguments.doBootstrap) {
//			arguments.bsFileName = bsOutFileArg.getValue();
			arguments.replicates = replicatesArg.getValue();
			if (reportRawEstimatesArg.isSet()) {
				arguments.reportRawEstimates = true;
			} else {
				arguments.reportRawEstimates = false;
			}
		}

		if (partialArg.getValue() != -1) {
			arguments.doPartial = true;
			arguments.replicates = partialArg.getValue();
		}

//		arguments.doLogisticRegression = doLogisticRegressionArg.isSet();
	} catch (ArgException &e) {
		cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
		exit(-1);
	}
}

double initializeModel(
		ModelData** modelData,
		CyclicCoordinateDescent** ccd,
		AbstractModelSpecifics** model,
//		ModelSpecifics<DefaultModel>** model,
		CCDArguments &arguments) {
	
	cout << "Running CCD (" <<
#ifdef DOUBLE_PRECISION
	"double"
#else
	"single"
#endif
	"-precision) ..." << endl;

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	InputReader* reader;

	if (arguments.fileFormat == "sccs") {
		reader = new SCCSInputReader();
	} else if (arguments.fileFormat == "clr") {
//		reader = new CLRInputReader();
		reader = new NewCLRInputReader();
	} else if (arguments.fileFormat == "csv") {
		reader = new RTestInputReader();
	} else if (arguments.fileFormat == "cc") {
		reader = new CCTestInputReader();
	} else if (arguments.fileFormat == "cox-csv") {
		reader = new CoxInputReader();
	} else if (arguments.fileFormat == "bbr") {
		reader = new BBRInputReader<NoImputation>();
	} else if (arguments.fileFormat == "generic") {
		reader = new NewSCCSInputReader();
	} else if (arguments.fileFormat == "new-cox") {
		reader = new NewCoxInputReader();
	} else {
		cerr << "Invalid file format." << endl;
		exit(-1);
	}

	reader->readFile(arguments.inFileName.c_str()); // TODO Check for error
	// delete reader;
	*modelData = reader->getModelData();

	if (arguments.modelName == "sccs") {
		*model = new ModelSpecifics<SelfControlledCaseSeries<real>,real>(**modelData);
	} else if (arguments.modelName == "clr") {
		*model = new ModelSpecifics<ConditionalLogisticRegression<real>,real>(**modelData);
	} else if (arguments.modelName == "lr") {
		*model = new ModelSpecifics<LogisticRegression<real>,real>(**modelData);
	} else if (arguments.modelName == "ls") {
		*model = new ModelSpecifics<LeastSquares<real>,real>(**modelData);
	} else if (arguments.modelName == "cox") {
		*model = new ModelSpecifics<CoxProportionalHazards<real>,real>(**modelData);
	} else {
		cerr << "Invalid model type." << endl;
		exit(-1);
	}

#ifdef CUDA
	if (arguments.useGPU) {
		*ccd = new GPUCyclicCoordinateDescent(arguments.deviceNumber, *reader, **model);
	} else {
#endif

	*ccd = new CyclicCoordinateDescent(*modelData /* TODO Change to ref */, **model);

#ifdef CUDA
	}
#endif

	// Set prior from the command-line
	if (arguments.useNormalPrior) {
		(*ccd)->setPriorType(NORMAL);
	}
	if (arguments.hyperPriorSet) {
		(*ccd)->setHyperprior(arguments.hyperprior);
	}

	gettimeofday(&time2, NULL);
	double sec1 = calculateSeconds(time1, time2);

	cout << "Everything loaded and ready to run ..." << endl;
	
	return sec1;
}

double fitModel(CyclicCoordinateDescent *ccd, CCDArguments &arguments) {
#ifndef MY_RCPP_FLAG
	cout << "Using prior: " << ccd->getPriorInfo() << endl;
#endif

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	ccd->update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);

	gettimeofday(&time2, NULL);

#ifndef MY_RCPP_FLAG
	ccd->logResults(arguments.outFileName.c_str());
#endif

	return calculateSeconds(time1, time2);
}

double runBoostrap(
		CyclicCoordinateDescent *ccd,
		ModelData *modelData,
		CCDArguments &arguments,
		std::vector<real>& savedBeta) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	BootstrapSelector selector(arguments.replicates, modelData->getPidVectorSTL(),
			SUBJECT, arguments.seed);
	BootstrapDriver driver(arguments.replicates, modelData);

	driver.drive(*ccd, selector, arguments);
	gettimeofday(&time2, NULL);

	driver.logResults(arguments, savedBeta, ccd->getConditionId());
	return calculateSeconds(time1, time2);
}

double runCrossValidation(CyclicCoordinateDescent *ccd, ModelData *modelData,
		CCDArguments &arguments) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	CrossValidationSelector selector(arguments.fold, modelData->getPidVectorSTL(),
			SUBJECT, arguments.seed);
	CrossValidationDriver driver(arguments.gridSteps, arguments.lowerLimit, arguments.upperLimit);

	driver.drive(*ccd, selector, arguments);

	gettimeofday(&time2, NULL);

	driver.logResults(arguments);

	if (arguments.doFitAtOptimal) {
		std::cout << "Fitting model at optimal hyperparameter" << std::endl;
 		// Do full fit for optimal parameter
		driver.resetForOptimal(*ccd, selector, arguments);
		fitModel(ccd, arguments);
	}

	return calculateSeconds(time1, time2);
}

#if 0

int main(int argc, char* argv[]) {

	std::string myFileName = "short.txt";

    CCDArguments* arguments = new CCDArguments;
    setDefaultArguments(*arguments);

    // Change options
    arguments->inFileName = myFileName;
    arguments->outFileName = "out.txt";
    arguments->fileFormat = "sccs";

//    std::vector<std::string> args;
//    args.push_back("R"); // program name
//    args.push_back("short.txt");
//    args.push_back("out.txt");

//	parseCommandLine(args, arguments); // TODO No idea why this doesn't work in Rcpp

    CyclicCoordinateDescent* ccd = NULL;
    InputReader* reader = NULL;
    double timeInitialize = initializeModel(&reader, &ccd, *arguments);

	double timeUpdate = fitModel(ccd, *arguments);

	cout << "Load   duration: " << scientific << timeInitialize << endl;
	cout << "Update duration: " << scientific << timeUpdate << endl;

	if (ccd)
		delete ccd;
	if (reader)
		delete reader;

    return 0;
}

#else

int main(int argc, char* argv[]) {

	CyclicCoordinateDescent* ccd = NULL;
	AbstractModelSpecifics* model = NULL;
	ModelData* modelData = NULL;
	CCDArguments arguments;

	parseCommandLine(argc, argv, arguments);

	double timeInitialize = initializeModel(&modelData, &ccd, &model, arguments);

	double timeUpdate;
	if (arguments.doCrossValidation) {
		timeUpdate = runCrossValidation(ccd, modelData, arguments);
	} else {
		if (arguments.doPartial) {
			ProportionSelector selector(arguments.replicates, modelData->getPidVectorSTL(),
					SUBJECT, arguments.seed);
			std::vector<real> weights;
			selector.getWeights(0, weights);
			ccd->setWeights(&weights[0]);
		}
		timeUpdate = fitModel(ccd, arguments);
	}

	if (arguments.doBootstrap) {
		// Save parameter point-estimates
		std::vector<real> savedBeta;
		for (int j = 0; j < ccd->getBetaSize(); ++j) {
			savedBeta.push_back(ccd->getBeta(j));
		}
		timeUpdate += runBoostrap(ccd, modelData, arguments, savedBeta);
	}
		
	cout << endl;
	cout << "Load   duration: " << scientific << timeInitialize << endl;
	cout << "Update duration: " << scientific << timeUpdate << endl;
	
	if (ccd)
		delete ccd;
	if (model)
		delete model;
	if (modelData)
		delete modelData;

    return 0;
}

#endif
