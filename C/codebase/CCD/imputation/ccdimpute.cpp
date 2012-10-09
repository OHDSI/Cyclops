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

#include "ccdimpute.h"
#include "CyclicCoordinateDescent.h"
#include "ModelData.h"
#include "io/InputReader.h"
#include "io/CLRInputReader.h"
#include "io/RTestInputReader.h"
#include "io/CCTestInputReader.h"
#include "io/CoxInputReader.h"
#include "io/BBRInputReader.h"
#include "CrossValidationSelector.h"
#include "CrossValidationDriver.h"
#include "BootstrapSelector.h"
#include "ProportionSelector.h"
#include "BootstrapDriver.h"
#include "ModelSpecifics.h"
#include "ImputeVariables.h"

#include "tclap/CmdLine.h"

//#include <R.h>

#ifdef CUDA
	#include "GPUCyclicCoordinateDescent.h"
//	#include "BetterGPU.h"
#endif


#define NEW

using namespace TCLAP;
using namespace std;

void parseCommandLine(int argc, char* argv[],
	CCDArguments &ccdArgs, ImputeArguments& imputeArgs) {
		std::vector<std::string> args;
		for (int i = 0; i < argc; i++)
			args.push_back(argv[i]);
		parseCommandLine(args, ccdArgs, imputeArgs);
}

void setDefaultArguments(CCDArguments &ccdArgs, ImputeArguments& imputeArgs) {
	ccdArgs.useGPU = false;
	ccdArgs.maxIterations = 100;
	ccdArgs.inFileName = "default_in";
	ccdArgs.outFileName = "default_out";
	ccdArgs.hyperPriorSet = false;
	ccdArgs.hyperprior = 1.0;
	ccdArgs.tolerance = 5E-4;
	ccdArgs.seed = 123;
	ccdArgs.doCrossValidation = false;
	ccdArgs.lowerLimit = 0.01;
	ccdArgs.upperLimit = 20.0;
	ccdArgs.fold = 10;
	ccdArgs.gridSteps = 10;
	ccdArgs.cvFileName = "cv.txt";
	ccdArgs.doBootstrap = false;
	ccdArgs.replicates = 100;
	ccdArgs.reportRawEstimates = false;
	//	ccdArgs.doLogisticRegression = false;
	ccdArgs.modelName = "sccs";
	ccdArgs.fileFormat = "sccs";
	ccdArgs.useNormalPrior = false;
	ccdArgs.convergenceType = GRADIENT;
	ccdArgs.convergenceTypeString = "gradient";
	ccdArgs.doPartial = false;
	imputeArgs.doImputation = false;
	imputeArgs.numberOfImputations = 5;
	imputeArgs.includeY = false;
}

void parseCommandLine(std::vector<std::string>& args,
		CCDArguments &ccdArgs, ImputeArguments& imputeArgs) {

	setDefaultArguments(ccdArgs,imputeArgs);

	try {
		CmdLine cmd("Cyclic coordinate descent algorithm for self-controlled case studies", ' ', "0.1");
		ValueArg<int> gpuArg("g","GPU","Use GPU device", ccdArgs.useGPU, -1, "device #");
//		SwitchArg betterGPUArg("1","better", "Use better GPU implementation", false);
		ValueArg<int> maxIterationsArg("", "maxIterations", "Maximum iterations", false, ccdArgs.maxIterations, "int");
		UnlabeledValueArg<string> inFileArg("inFileName","Input file name", true, ccdArgs.inFileName, "inFileName");
		UnlabeledValueArg<string> outFileArg("outFileName","Output file name", true, ccdArgs.outFileName, "outFileName");

		// Prior ccdArgs
		ValueArg<double> hyperPriorArg("v", "variance", "Hyperprior variance", false, ccdArgs.hyperprior, "real");
		SwitchArg normalPriorArg("n", "normalPrior", "Use normal prior, default is laplace", ccdArgs.useNormalPrior);

		// Convergence criterion ccdArgs
		ValueArg<double> toleranceArg("t", "tolerance", "Convergence criterion tolerance", false, ccdArgs.tolerance, "real");
//		SwitchArg zhangOlesConvergenceArg("z", "zhangOles", "Use Zhange-Oles convergence criterion, default is true", true);
		std::vector<std::string> allowedConvergence;
		allowedConvergence.push_back("gradient");
		allowedConvergence.push_back("ZhangOles");
		allowedConvergence.push_back("Lange");
		allowedConvergence.push_back("Mittal");
		ValuesConstraint<std::string> allowedConvergenceValues(allowedConvergence);
		ValueArg<string> convergenceArg("", "convergence", "Convergence criterion", false, ccdArgs.convergenceTypeString, &allowedConvergenceValues);

		ValueArg<long> seedArg("s", "seed", "Random number generator seed", false, ccdArgs.seed, "long");

		// Cross-validation ccdArgs
		SwitchArg doCVArg("c", "cv", "Perform cross-validation selection of hyperprior variance", ccdArgs.doCrossValidation);
		ValueArg<double> lowerCVArg("l", "lower", "Lower limit for cross-validation search", false, ccdArgs.lowerLimit, "real");
		ValueArg<double> upperCVArg("u", "upper", "Upper limit for cross-validation search", false, ccdArgs.upperLimit, "real");
		ValueArg<int> foldCVArg("f", "fold", "Fold level for cross-validation", false, ccdArgs.fold, "int");
		ValueArg<int> gridCVArg("", "gridSize", "Uniform grid size for cross-validation search", false, ccdArgs.gridSteps, "int");
		ValueArg<int> foldToComputeCVArg("", "computeFold", "Number of fold to iterate, default is 'fold' value", false, 10, "int");
		ValueArg<string> outFile2Arg("", "cvFileName", "Cross-validation output file name", false, ccdArgs.cvFileName, "cvFileName");

		// Bootstrap ccdArgs
		SwitchArg doBootstrapArg("b", "bs", "Perform bootstrap estimation", ccdArgs.doBootstrap);
//		ValueArg<string> bsOutFileArg("", "bsFileName", "Bootstrap output file name", false, "bs.txt", "bsFileName");
		ValueArg<int> replicatesArg("r", "replicates", "Number of bootstrap replicates", false, ccdArgs.replicates, "int");
		SwitchArg reportRawEstimatesArg("","raw", "Report the raw bootstrap estimates", ccdArgs.reportRawEstimates);
		ValueArg<int> partialArg("", "partial", "Number of rows to use in partial estimation", false, -1, "int");

		// Model ccdArgs
//		SwitchArg doLogisticRegressionArg("", "logistic", "Use ordinary logistic regression", ccdArgs.doLogisticRegression);
		std::vector<std::string> allowedModels;
		allowedModels.push_back("sccs");
		allowedModels.push_back("clr");
		allowedModels.push_back("lr");
		allowedModels.push_back("ls");
		allowedModels.push_back("cox");
		ValuesConstraint<std::string> allowedModelValues(allowedModels);
		ValueArg<string> modelArg("", "model", "Model specification", false, ccdArgs.modelName, &allowedModelValues);

		// Format ccdArgs
		std::vector<std::string> allowedFormats;
		allowedFormats.push_back("sccs");
		allowedFormats.push_back("clr");
		allowedFormats.push_back("csv");
		allowedFormats.push_back("cc");
		allowedFormats.push_back("cox-csv");
		allowedFormats.push_back("bbr");
		ValuesConstraint<std::string> allowedFormatValues(allowedFormats);
		ValueArg<string> formatArg("", "format", "Format of data file", false, ccdArgs.fileFormat, &allowedFormatValues);

		// Imputation ccdArgs
		SwitchArg doImputationArg("i", "imputation", "Perform multiple imputation", imputeArgs.doImputation);
		ValueArg<int> numberOfImputationsArg("m", "numberOfImputations", "Number of imputed data sets (default is m=5)", false, imputeArgs.numberOfImputations, "int");
		SwitchArg includeYArg("y", "includeY", "Use output vector y for imputation", imputeArgs.includeY);

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

		cmd.add(doImputationArg);
		cmd.add(numberOfImputationsArg);
		cmd.add(includeYArg);
		cmd.add(inFileArg);
		cmd.add(outFileArg);
		cmd.parse(args);

		if (gpuArg.getValue() > -1) {
			ccdArgs.useGPU = true;
			ccdArgs.deviceNumber = gpuArg.getValue();
		} else {
			ccdArgs.useGPU = false;
		}
//		ccdArgs.useBetterGPU = betterGPUArg.isSet();

		ccdArgs.inFileName = inFileArg.getValue();
		ccdArgs.outFileName = outFileArg.getValue();
		ccdArgs.tolerance = toleranceArg.getValue();
		ccdArgs.maxIterations = maxIterationsArg.getValue();
		ccdArgs.hyperprior = hyperPriorArg.getValue();
		ccdArgs.useNormalPrior = normalPriorArg.getValue();
		ccdArgs.seed = seedArg.getValue();

		ccdArgs.modelName = modelArg.getValue();
		ccdArgs.fileFormat = formatArg.getValue();
		ccdArgs.convergenceTypeString = convergenceArg.getValue();

		if (hyperPriorArg.isSet()) {
			ccdArgs.hyperPriorSet = true;
		} else {
			ccdArgs.hyperPriorSet = false;
		}

//		if (zhangOlesConvergenceArg.isSet()) {
//			ccdArgs.convergenceType = ZHANG_OLES;
//		} else {
//			ccdArgs.convergenceType = LANGE;
//		}
		if (ccdArgs.convergenceTypeString == "ZhangOles") {
			ccdArgs.convergenceType = ZHANG_OLES;
		} else if (ccdArgs.convergenceTypeString == "Lange") {
			ccdArgs.convergenceType = LANGE;
		} else if (ccdArgs.convergenceTypeString == "Mittal") {
			ccdArgs.convergenceType = MITTAL;
		} else if (ccdArgs.convergenceTypeString == "gradient") {
			ccdArgs.convergenceType = GRADIENT;
		} else {
			cerr << "Unknown convergence type: " << convergenceArg.getValue() << " " << ccdArgs.convergenceTypeString << endl;
			exit(-1);
		}

		// Cross-validation
		ccdArgs.doCrossValidation = doCVArg.isSet();
		if (ccdArgs.doCrossValidation) {
			ccdArgs.lowerLimit = lowerCVArg.getValue();
			ccdArgs.upperLimit = upperCVArg.getValue();
			ccdArgs.fold = foldCVArg.getValue();
			ccdArgs.gridSteps = gridCVArg.getValue();
			if(foldToComputeCVArg.isSet()) {
				ccdArgs.foldToCompute = foldToComputeCVArg.getValue();
			} else {
				ccdArgs.foldToCompute = ccdArgs.fold;
			}
			ccdArgs.cvFileName = outFile2Arg.getValue();
			ccdArgs.doFitAtOptimal = true;
		}

		// Bootstrap
		ccdArgs.doBootstrap = doBootstrapArg.isSet();
		if (ccdArgs.doBootstrap) {
//			ccdArgs.bsFileName = bsOutFileArg.getValue();
			ccdArgs.replicates = replicatesArg.getValue();
			if (reportRawEstimatesArg.isSet()) {
				ccdArgs.reportRawEstimates = true;
			} else {
				ccdArgs.reportRawEstimates = false;
			}
		}

		// Imputation
		imputeArgs.doImputation = doImputationArg.isSet();
		if(imputeArgs.doImputation){
			imputeArgs.numberOfImputations = numberOfImputationsArg.getValue();
			imputeArgs.includeY = includeYArg.isSet();
		}

		if (partialArg.getValue() != -1) {
			ccdArgs.doPartial = true;
			ccdArgs.replicates = partialArg.getValue();
		}

//		ccdArgs.doLogisticRegression = doLogisticRegressionArg.isSet();
	} catch (ArgException &e) {
		cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
		exit(-1);
	}
}

int main(int argc, char* argv[]) {

	CCDArguments ccdArgs;
	ImputeArguments imputeArgs;

	parseCommandLine(argc, argv, ccdArgs, imputeArgs);

	ImputeVariables imputation;
	imputation.initialize(ccdArgs, imputeArgs.numberOfImputations, imputeArgs.includeY);
	imputation.impute();
	return 0;
}
