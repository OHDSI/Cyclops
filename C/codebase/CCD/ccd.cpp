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
#include <sys/time.h>

#include <math.h>

#include "ccd.h"
#include "CyclicCoordinateDescent.h"
#include "InputReader.h"
#include "CLRInputReader.h"
#include "RTestInputReader.h"
#include "CCTestInputReader.h"
#include "CrossValidationSelector.h"
#include "CrossValidationDriver.h"
#include "BootstrapSelector.h"
#include "BootstrapDriver.h"
#include "SparseRowVector.h"

#include "tclap/CmdLine.h"

//#include <R.h>


#ifdef CUDA
	#include "GPUCyclicCoordinateDescent.h"
	#include "BetterGPU.h"
#endif



#define NEW

using namespace TCLAP;
using namespace std;
using namespace BayesianSCCS;

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
	arguments.doLogisticRegression = false;
	arguments.fileFormat = "sccs";
	arguments.useNormalPrior = false;
	arguments.convergenceType = ZHANG_OLES;
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
		ValueArg<double> hyperPriorArg("v", "variance", "Hyperprior variance", false, arguments.hyperprior, "realTRS");
		SwitchArg normalPriorArg("n", "normalPrior", "Use normal prior, default is laplace", arguments.useNormalPrior);

		// Convergence criterion arguments
		ValueArg<double> toleranceArg("t", "tolerance", "Convergence criterion tolerance", false, arguments.tolerance, "realTRS");
		SwitchArg zhangOlesConvergenceArg("z", "zhangOles", "Use Zhange-Oles convergence criterion, default is true", true);
		ValueArg<long> seedArg("s", "seed", "Random number generator seed", false, arguments.seed, "long");

		// Cross-validation arguments
		SwitchArg doCVArg("c", "cv", "Perform cross-validation selection of hyperprior variance", arguments.doCrossValidation);
		ValueArg<double> lowerCVArg("l", "lower", "Lower limit for cross-validation search", false, arguments.lowerLimit, "realTRS");
		ValueArg<double> upperCVArg("u", "upper", "Upper limit for cross-validation search", false, arguments.upperLimit, "realTRS");
		ValueArg<int> foldCVArg("f", "fold", "Fold level for cross-validation", false, arguments.fold, "int");
		ValueArg<int> gridCVArg("", "gridSize", "Uniform grid size for cross-validation search", false, arguments.gridSteps, "int");
		ValueArg<int> foldToComputeCVArg("", "computeFold", "Number of fold to iterate, default is 'fold' value", false, 10, "int");
		ValueArg<string> outFile2Arg("", "cvFileName", "Cross-validation output file name", false, arguments.cvFileName, "cvFileName");

		// Bootstrap arguments
		SwitchArg doBootstrapArg("b", "bs", "Perform bootstrap estimation", arguments.doBootstrap);
//		ValueArg<string> bsOutFileArg("", "bsFileName", "Bootstrap output file name", false, "bs.txt", "bsFileName");
		ValueArg<int> replicatesArg("r", "replicates", "Number of bootstrap replicates", false, arguments.replicates, "int");
		SwitchArg reportRawEstimatesArg("","raw", "Report the raw bootstrap estimates", arguments.reportRawEstimates);

		// Model arguments
		SwitchArg doLogisticRegressionArg("", "logistic", "Use ordinary logistic regression", arguments.doLogisticRegression);

		// Format arguments
		std::vector<std::string> allowed;
		allowed.push_back("sccs");
		allowed.push_back("clr");
		allowed.push_back("csv");
		allowed.push_back("cc");
		ValuesConstraint<std::string> allowedValues(allowed);
		ValueArg<string> formatArg("", "format", "Format of data file", false, arguments.fileFormat, &allowedValues);

		cmd.add(gpuArg);
//		cmd.add(betterGPUArg);
		cmd.add(toleranceArg);
		cmd.add(maxIterationsArg);
		cmd.add(hyperPriorArg);
		cmd.add(normalPriorArg);
		cmd.add(zhangOlesConvergenceArg);
		cmd.add(seedArg);
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
		cmd.add(reportRawEstimatesArg);
		cmd.add(doLogisticRegressionArg);

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

		arguments.fileFormat = formatArg.getValue();

		if (hyperPriorArg.isSet()) {
			arguments.hyperPriorSet = true;
		} else {
			arguments.hyperPriorSet = false;
		}

		if (zhangOlesConvergenceArg.isSet()) {
			arguments.convergenceType = ZHANG_OLES;
		} else {
			arguments.convergenceType = LANGE;
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
		arguments.doLogisticRegression = doLogisticRegressionArg.isSet();
	} catch (ArgException &e) {
		cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
		exit(-1);
	}
}

double initializeModel(
		InputReader** reader,
		CyclicCoordinateDescent** ccd,
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

	if (arguments.fileFormat == "sccs") {
		*reader = new SCCSInputReader();
	} else if (arguments.fileFormat == "clr") {
		*reader = new CLRInputReader();
	} else if (arguments.fileFormat == "csv") {
		*reader = new RTestInputReader();
	} else if (arguments.fileFormat == "cc") {
		*reader = new CCTestInputReader();
	} else {
		cerr << "Invalid file format." << endl;
		exit(-1);
	}
	(*reader)->readFile(arguments.inFileName.c_str()); // TODO Check for error


#ifdef CUDA
	if (arguments.useGPU) {
		*ccd = new GPUCyclicCoordinateDescent(arguments.deviceNumber, *reader);
	} else {
#endif

	*ccd = new CyclicCoordinateDescent(*reader);

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

	// Set model from the command-line
	if (arguments.doLogisticRegression) {
		(*ccd)->setLogisticRegression(true);
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
		InputReader *reader,
		CCDArguments &arguments,
		std::vector<realTRS>& savedBeta) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	BootstrapSelector selector(arguments.replicates, reader->getPidVectorSTL(),
			SUBJECT, arguments.seed);
	BootstrapDriver driver(arguments.replicates, reader);

	driver.drive(*ccd, selector, arguments);
	gettimeofday(&time2, NULL);

	driver.logResults(arguments, savedBeta, ccd->getConditionId());
	return calculateSeconds(time1, time2);
}

double runCrossValidation(CyclicCoordinateDescent *ccd, InputReader *reader,
		CCDArguments &arguments) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	CrossValidationSelector selector(arguments.fold, reader->getPidVectorSTL(),
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
	InputReader* reader = NULL;
	CCDArguments arguments;



	parseCommandLine(argc, argv, arguments);

	double timeInitialize = initializeModel(&reader, &ccd, arguments);

	double timeUpdate;
	if (arguments.doCrossValidation) {
		timeUpdate = runCrossValidation(ccd, reader, arguments);
	} else {
		timeUpdate = fitModel(ccd, arguments);
	}

	if (arguments.doBootstrap) {
		// Save parameter point-estimates
		std::vector<realTRS> savedBeta;
		for (int j = 0; j < ccd->getBetaSize(); ++j) {
			savedBeta.push_back(ccd->getBeta(j));
		}
		timeUpdate += runBoostrap(ccd, reader, arguments, savedBeta);
	}
		
	cout << endl;
	cout << "Load   duration: " << scientific << timeInitialize << endl;
	cout << "Update duration: " << scientific << timeUpdate << endl;
	
	if (ccd)
		delete ccd;
	if (reader)
		delete reader;

    return 0;
}

#endif
