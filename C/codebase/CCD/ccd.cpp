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

#ifndef _MSC_VER
	#include <sys/time.h>
#endif

#include <math.h>

#include "Types.h"
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
#include "io/NewGenericInputReader.h"
#include "io/BBRInputReader.h"
#include "io/OutputWriter.h"
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
#ifdef _MSC_VER
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
	arguments.maxIterations = 1000;
	arguments.inFileName = "default_in";
	arguments.outFileName = "default_out";
	arguments.hyperPriorSet = false;
	arguments.hyperprior = 1.0;
	arguments.tolerance = 1E-6; //5E-4;
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
	arguments.modelName = "sccs";
	arguments.fileFormat = "sccs";
	//arguments.outputFormat = "estimates";
	arguments.computeMLE = false;
	arguments.fitMLEAtMode = false;
	arguments.useNormalPrior = false;
	arguments.convergenceType = GRADIENT;
	arguments.convergenceTypeString = "gradient";
	arguments.doPartial = false;
	arguments.noiseLevel = NOISY;
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
		SwitchArg computeMLEArg("", "MLE", "Compute maximum likelihood estimates only", arguments.computeMLE);
		SwitchArg computeMLEAtModeArg("", "MLEAtMode", "Compute maximum likelihood estimates at posterior mode", arguments.fitMLEAtMode);

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
		allowedModels.push_back("pr");
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

		// Output format arguments
		std::vector<std::string> allowedOutputFormats;
		allowedOutputFormats.push_back("estimates");
		allowedOutputFormats.push_back("prediction");
		allowedOutputFormats.push_back("diagnostics");
		ValuesConstraint<std::string> allowedOutputFormatValues(allowedOutputFormats);
//		ValueArg<string> outputFormatArg("", "outputFormat", "Format of the output file", false, arguments.outputFormat, &allowedOutputFormatValues);
		MultiArg<std::string> outputFormatArg("", "output", "Format of the output file", false, &allowedOutputFormatValues);

		// Control screen output volume
		SwitchArg quietArg("q", "quiet", "Limit writing to standard out", arguments.noiseLevel <= QUIET);


		cmd.add(gpuArg);
//		cmd.add(betterGPUArg);
		cmd.add(toleranceArg);
		cmd.add(maxIterationsArg);
		cmd.add(hyperPriorArg);
		cmd.add(normalPriorArg);
		cmd.add(computeMLEArg);
		cmd.add(computeMLEAtModeArg);
//		cmd.add(zhangOlesConvergenceArg);
		cmd.add(convergenceArg);
		cmd.add(seedArg);
		cmd.add(modelArg);
		cmd.add(formatArg);
		cmd.add(outputFormatArg);

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

		cmd.add(quietArg);

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
		arguments.computeMLE = computeMLEArg.getValue();
		arguments.fitMLEAtMode = computeMLEAtModeArg.getValue();
		arguments.seed = seedArg.getValue();

		arguments.modelName = modelArg.getValue();
		arguments.fileFormat = formatArg.getValue();
		arguments.outputFormat = outputFormatArg.getValue();
		if (arguments.outputFormat.size() == 0) {
			arguments.outputFormat.push_back("estimates");
		}

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

		if (quietArg.getValue()) {
			arguments.noiseLevel = QUIET;
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

	// Parse type of model
	//using namespace bsccs::Models;
	bsccs::Models::ModelType modelType;
	if (arguments.modelName == "sccs") {
		modelType = bsccs::Models::SELF_CONTROLLED_MODEL;
	} else if (arguments.modelName == "clr") {
		modelType = bsccs::Models::CONDITIONAL_LOGISTIC;
	} else if (arguments.modelName == "lr") {
		modelType = bsccs::Models::LOGISTIC;
	} else if (arguments.modelName == "ls") {
		modelType = bsccs::Models::NORMAL;
	} else if (arguments.modelName == "pr") {
		modelType = bsccs::Models::POISSON;
	} else if (arguments.modelName == "cox") {
		modelType = bsccs::Models::COX;
	} else {
		cerr << "Invalid model type." << endl;
		exit(-1);
	}

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
		reader = new NewGenericInputReader(modelType);
	} else if (arguments.fileFormat == "new-cox") {
		reader = new NewCoxInputReader();
	} else {
		cerr << "Invalid file format." << endl;
		exit(-1);
	}

	reader->readFile(arguments.inFileName.c_str()); // TODO Check for error
	// delete reader;
	*modelData = reader->getModelData();

	switch (modelType) {
		case bsccs::Models::SELF_CONTROLLED_MODEL :
			*model = new ModelSpecifics<SelfControlledCaseSeries<real>,real>(**modelData);
			break;
		case bsccs::Models::CONDITIONAL_LOGISTIC :
			*model = new ModelSpecifics<ConditionalLogisticRegression<real>,real>(**modelData);
			break;
		case bsccs::Models::LOGISTIC :
			*model = new ModelSpecifics<LogisticRegression<real>,real>(**modelData);
			break;
		case bsccs::Models::NORMAL :
			*model = new ModelSpecifics<LeastSquares<real>,real>(**modelData);
			break;
		case bsccs::Models::POISSON :
			*model = new ModelSpecifics<PoissonRegression<real>,real>(**modelData);
			break;
		case bsccs::Models::COX :
			*model = new ModelSpecifics<CoxProportionalHazards<real>,real>(**modelData);
			break;
		default:
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

	(*ccd)->setNoiseLevel(arguments.noiseLevel);

	if (arguments.computeMLE) {
		(*ccd)->setPriorType(NONE);
		if (arguments.fitMLEAtMode) {
			cerr << "Unable to compute MLE at posterior mode, if mode is not first explored." << endl;
			exit(-1);
		}
	}

	gettimeofday(&time2, NULL);
	double sec1 = calculateSeconds(time1, time2);

	cout << "Everything loaded and ready to run ..." << endl;
	
	return sec1;
}

double predictModel(CyclicCoordinateDescent *ccd, ModelData *modelData, CCDArguments &arguments) {

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	bsccs::PredictionOutputWriter predictor(*ccd, *modelData);

	string fileName;
	if (arguments.outputFormat.size() == 1) {
		fileName = arguments.outFileName;
	} else {
		fileName = "pred_" + arguments.outFileName;
	}

	predictor.writeFile(fileName.c_str());

	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

double diagnoseModel(CyclicCoordinateDescent *ccd, ModelData *modelData,
		CCDArguments& arguments,
		double loadTime,
		double updateTime) {

	using namespace bsccs;
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	DiagnosticsOutputWriter diagnostics(*ccd, *modelData);

	string fileName;
	if (arguments.outputFormat.size() == 1) {
		fileName = arguments.outFileName;
	} else {
		fileName = "diag_" + arguments.outFileName;
	}

	vector<ExtraInformation> extraInfo;
	extraInfo.push_back(ExtraInformation("load_time",loadTime));
	extraInfo.push_back(ExtraInformation("update_time",updateTime));

	diagnostics.addExtraInformation(extraInfo);
	diagnostics.writeFile(fileName.c_str());


	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);

}

void setZeroBetaAsFixed(CyclicCoordinateDescent *ccd) {
	for (int j = 0; j < ccd->getBetaSize(); ++j) {
		if (ccd->getBeta(j) == 0.0) {
			ccd->setFixedBeta(j, true);
		}
	}
}

double fitModel(CyclicCoordinateDescent *ccd, CCDArguments &arguments) {
#ifndef MY_RCPP_FLAG
	cout << "Using prior: " << ccd->getPriorInfo() << endl;
#endif

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	ccd->update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);

	gettimeofday(&time2, NULL);

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


double runFitMLEAtMode(CyclicCoordinateDescent* ccd, CCDArguments &arguments) {
	std::cout << std::endl << "Estimating MLE at posterior mode" << std::endl;

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	setZeroBetaAsFixed(ccd);
	ccd->setPriorType(NONE);
	fitModel(ccd, arguments);

	gettimeofday(&time2, NULL);
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
		if (arguments.fitMLEAtMode) {
			runFitMLEAtMode(ccd, arguments);
		}
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

bool includesOption(const string& line, const string& option) {
	size_t found = line.find(option);
	return found != string::npos;
}

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
		if (arguments.fitMLEAtMode) {
			timeUpdate += runFitMLEAtMode(ccd, arguments);
		}
	}

	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "estimates")
			!= arguments.outputFormat.end()) {
#ifndef MY_RCPP_FLAG
		// TODO Make into OutputWriter
		bool withASE = false;
		string fileName;
		if (arguments.outputFormat.size() == 1) {
			fileName = arguments.outFileName;
		} else {
			fileName = "est_" + arguments.outFileName;
		}
		ccd->logResults(fileName.c_str(), withASE);
#endif
	}

	double timePredict;
	bool doPrediction = false;
	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "prediction")
			!= arguments.outputFormat.end()) {
		doPrediction = true;
		timePredict = predictModel(ccd, modelData, arguments);
	}

	double timeDiagnose;
	bool doDiagnosis = false;
	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "diagnostics")
			!= arguments.outputFormat.end()) {
		doDiagnosis = true;
		timeDiagnose = diagnoseModel(ccd, modelData, arguments, timeInitialize, timeUpdate);
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
	cout << "Load    duration: " << scientific << timeInitialize << endl;
	cout << "Update  duration: " << scientific << timeUpdate << endl;
	if (doPrediction) {
		cout << "Predict duration: " << scientific << timePredict << endl;
	}
	
	if (doDiagnosis) {
		cout << "Diag    duration: " << scientific << timeDiagnose << endl;
	}

//#define PRINT_LOG_LIKELIHOOD
#ifdef PRINT_LOG_LIKELIHOOD
	cout << endl << setprecision(15) << ccd->getLogLikelihood() << endl;
#endif

	if (ccd)
		delete ccd;
	if (model)
		delete model;
	if (modelData)
		delete modelData;

    return 0;
}

#endif
