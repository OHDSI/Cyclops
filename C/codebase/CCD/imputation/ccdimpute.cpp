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

#include "imputation/ccdimpute.h"
#include "ccd.h"
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

namespace bsccs {

using namespace TCLAP;
using namespace std;

void parseCommandLine(int argc, char* argv[],
	CCDImputeArguments &ccdImputeArgs) {
		std::vector<std::string> args;
		for (int i = 0; i < argc; i++)
			args.push_back(argv[i]);
		parseCommandLine(args, ccdImputeArgs);
}

void setDefaultArguments(CCDImputeArguments &ccdImputeArgs) {
	ccdImputeArgs.useGPU = false;
	ccdImputeArgs.maxIterations = 100;
	ccdImputeArgs.inFileName = "default_in";
	ccdImputeArgs.outFileName = "default_out";
	ccdImputeArgs.hyperPriorSet = false;
	ccdImputeArgs.hyperprior = 1.0;
	ccdImputeArgs.tolerance = 5E-4;
	ccdImputeArgs.seed = 123;
	ccdImputeArgs.doCrossValidation = false;
	ccdImputeArgs.lowerLimit = 0.01;
	ccdImputeArgs.upperLimit = 20.0;
	ccdImputeArgs.fold = 10;
	ccdImputeArgs.gridSteps = 10;
	ccdImputeArgs.cvFileName = "cv.txt";
	ccdImputeArgs.doBootstrap = false;
	ccdImputeArgs.replicates = 100;
	ccdImputeArgs.reportRawEstimates = false;
	//	ccdImputeArgs.doLogisticRegression = false;
	ccdImputeArgs.modelName = "sccs";
	ccdImputeArgs.fileFormat = "sccs";
	ccdImputeArgs.useNormalPrior = false;
	ccdImputeArgs.convergenceType = GRADIENT;
	ccdImputeArgs.convergenceTypeString = "gradient";
	ccdImputeArgs.doPartial = false;
	ccdImputeArgs.doImputation = false;
	ccdImputeArgs.numberOfImputations = 5;
	ccdImputeArgs.includeY = false;
}

void parseCommandLine(std::vector<std::string>& args,
		CCDImputeArguments &ccdImputeArgs) {

	setDefaultArguments(ccdImputeArgs);

	try {
		CmdLine cmd("Cyclic coordinate descent algorithm for self-controlled case studies", ' ', "0.1");
		ValueArg<int> gpuArg("g","GPU","Use GPU device", ccdImputeArgs.useGPU, -1, "device #");
//		SwitchArg betterGPUArg("1","better", "Use better GPU implementation", false);
		ValueArg<int> maxIterationsArg("", "maxIterations", "Maximum iterations", false, ccdImputeArgs.maxIterations, "int");
		UnlabeledValueArg<string> inFileArg("inFileName","Input file name", true, ccdImputeArgs.inFileName, "inFileName");
		UnlabeledValueArg<string> outFileArg("outFileName","Output file name", true, ccdImputeArgs.outFileName, "outFileName");

		// Prior ccdImputeArgs
		ValueArg<double> hyperPriorArg("v", "variance", "Hyperprior variance", false, ccdImputeArgs.hyperprior, "real");
		SwitchArg normalPriorArg("n", "normalPrior", "Use normal prior, default is laplace", ccdImputeArgs.useNormalPrior);

		// Convergence criterion ccdImputeArgs
		ValueArg<double> toleranceArg("t", "tolerance", "Convergence criterion tolerance", false, ccdImputeArgs.tolerance, "real");
//		SwitchArg zhangOlesConvergenceArg("z", "zhangOles", "Use Zhange-Oles convergence criterion, default is true", true);
		std::vector<std::string> allowedConvergence;
		allowedConvergence.push_back("gradient");
		allowedConvergence.push_back("ZhangOles");
		allowedConvergence.push_back("Lange");
		allowedConvergence.push_back("Mittal");
		ValuesConstraint<std::string> allowedConvergenceValues(allowedConvergence);
		ValueArg<string> convergenceArg("", "convergence", "Convergence criterion", false, ccdImputeArgs.convergenceTypeString, &allowedConvergenceValues);

		ValueArg<long> seedArg("s", "seed", "Random number generator seed", false, ccdImputeArgs.seed, "long");

		// Cross-validation ccdImputeArgs
		SwitchArg doCVArg("c", "cv", "Perform cross-validation selection of hyperprior variance", ccdImputeArgs.doCrossValidation);
		ValueArg<double> lowerCVArg("l", "lower", "Lower limit for cross-validation search", false, ccdImputeArgs.lowerLimit, "real");
		ValueArg<double> upperCVArg("u", "upper", "Upper limit for cross-validation search", false, ccdImputeArgs.upperLimit, "real");
		ValueArg<int> foldCVArg("f", "fold", "Fold level for cross-validation", false, ccdImputeArgs.fold, "int");
		ValueArg<int> gridCVArg("", "gridSize", "Uniform grid size for cross-validation search", false, ccdImputeArgs.gridSteps, "int");
		ValueArg<int> foldToComputeCVArg("", "computeFold", "Number of fold to iterate, default is 'fold' value", false, 10, "int");
		ValueArg<string> outFile2Arg("", "cvFileName", "Cross-validation output file name", false, ccdImputeArgs.cvFileName, "cvFileName");

		// Bootstrap ccdImputeArgs
		SwitchArg doBootstrapArg("b", "bs", "Perform bootstrap estimation", ccdImputeArgs.doBootstrap);
//		ValueArg<string> bsOutFileArg("", "bsFileName", "Bootstrap output file name", false, "bs.txt", "bsFileName");
		ValueArg<int> replicatesArg("r", "replicates", "Number of bootstrap replicates", false, ccdImputeArgs.replicates, "int");
		SwitchArg reportRawEstimatesArg("","raw", "Report the raw bootstrap estimates", ccdImputeArgs.reportRawEstimates);
		ValueArg<int> partialArg("", "partial", "Number of rows to use in partial estimation", false, -1, "int");

		// Model ccdImputeArgs
//		SwitchArg doLogisticRegressionArg("", "logistic", "Use ordinary logistic regression", ccdImputeArgs.doLogisticRegression);
		std::vector<std::string> allowedModels;
		allowedModels.push_back("sccs");
		allowedModels.push_back("clr");
		allowedModels.push_back("lr");
		allowedModels.push_back("ls");
		allowedModels.push_back("pr");
		allowedModels.push_back("cox");
		ValuesConstraint<std::string> allowedModelValues(allowedModels);
		ValueArg<string> modelArg("", "model", "Model specification", false, ccdImputeArgs.modelName, &allowedModelValues);

		// Format ccdImputeArgs
		std::vector<std::string> allowedFormats;
		allowedFormats.push_back("sccs");
		allowedFormats.push_back("clr");
		allowedFormats.push_back("csv");
		allowedFormats.push_back("cc");
		allowedFormats.push_back("cox-csv");
		allowedFormats.push_back("bbr");
		ValuesConstraint<std::string> allowedFormatValues(allowedFormats);
		ValueArg<string> formatArg("", "format", "Format of data file", false, ccdImputeArgs.fileFormat, &allowedFormatValues);

		// Imputation ccdImputeArgs
		SwitchArg doImputationArg("i", "imputation", "Perform multiple imputation", ccdImputeArgs.doImputation);
		ValueArg<int> numberOfImputationsArg("m", "numberOfImputations", "Number of imputed data sets (default is m=5)", false, ccdImputeArgs.numberOfImputations, "int");
		SwitchArg includeYArg("y", "includeY", "Use output vector y for imputation", ccdImputeArgs.includeY);

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
			ccdImputeArgs.useGPU = true;
			ccdImputeArgs.deviceNumber = gpuArg.getValue();
		} else {
			ccdImputeArgs.useGPU = false;
		}
//		ccdImputeArgs.useBetterGPU = betterGPUArg.isSet();

		ccdImputeArgs.inFileName = inFileArg.getValue();
		ccdImputeArgs.outFileName = outFileArg.getValue();
		ccdImputeArgs.tolerance = toleranceArg.getValue();
		ccdImputeArgs.maxIterations = maxIterationsArg.getValue();
		ccdImputeArgs.hyperprior = hyperPriorArg.getValue();
		ccdImputeArgs.useNormalPrior = normalPriorArg.getValue();
		ccdImputeArgs.seed = seedArg.getValue();

		ccdImputeArgs.modelName = modelArg.getValue();
		ccdImputeArgs.fileFormat = formatArg.getValue();
		ccdImputeArgs.convergenceTypeString = convergenceArg.getValue();

		if (hyperPriorArg.isSet()) {
			ccdImputeArgs.hyperPriorSet = true;
		} else {
			ccdImputeArgs.hyperPriorSet = false;
		}

//		if (zhangOlesConvergenceArg.isSet()) {
//			ccdImputeArgs.convergenceType = ZHANG_OLES;
//		} else {
//			ccdImputeArgs.convergenceType = LANGE;
//		}
		if (ccdImputeArgs.convergenceTypeString == "ZhangOles") {
			ccdImputeArgs.convergenceType = ZHANG_OLES;
		} else if (ccdImputeArgs.convergenceTypeString == "Lange") {
			ccdImputeArgs.convergenceType = LANGE;
		} else if (ccdImputeArgs.convergenceTypeString == "Mittal") {
			ccdImputeArgs.convergenceType = MITTAL;
		} else if (ccdImputeArgs.convergenceTypeString == "gradient") {
			ccdImputeArgs.convergenceType = GRADIENT;
		} else {
			cerr << "Unknown convergence type: " << convergenceArg.getValue() << " " << ccdImputeArgs.convergenceTypeString << endl;
			exit(-1);
		}

		// Cross-validation
		ccdImputeArgs.doCrossValidation = doCVArg.isSet();
		if (ccdImputeArgs.doCrossValidation) {
			ccdImputeArgs.lowerLimit = lowerCVArg.getValue();
			ccdImputeArgs.upperLimit = upperCVArg.getValue();
			ccdImputeArgs.fold = foldCVArg.getValue();
			ccdImputeArgs.gridSteps = gridCVArg.getValue();
			if(foldToComputeCVArg.isSet()) {
				ccdImputeArgs.foldToCompute = foldToComputeCVArg.getValue();
			} else {
				ccdImputeArgs.foldToCompute = ccdImputeArgs.fold;
			}
			ccdImputeArgs.cvFileName = outFile2Arg.getValue();
			ccdImputeArgs.doFitAtOptimal = true;
		}

		// Bootstrap
		ccdImputeArgs.doBootstrap = doBootstrapArg.isSet();
		if (ccdImputeArgs.doBootstrap) {
//			ccdImputeArgs.bsFileName = bsOutFileArg.getValue();
			ccdImputeArgs.replicates = replicatesArg.getValue();
			if (reportRawEstimatesArg.isSet()) {
				ccdImputeArgs.reportRawEstimates = true;
			} else {
				ccdImputeArgs.reportRawEstimates = false;
			}
		}

		// Imputation
		ccdImputeArgs.doImputation = doImputationArg.isSet();
		if(ccdImputeArgs.doImputation){
			ccdImputeArgs.numberOfImputations = numberOfImputationsArg.getValue();
			ccdImputeArgs.includeY = includeYArg.isSet();
		}

		if (partialArg.getValue() != -1) {
			ccdImputeArgs.doPartial = true;
			ccdImputeArgs.replicates = partialArg.getValue();
		}

//		ccdImputeArgs.doLogisticRegression = doLogisticRegressionArg.isSet();
	} catch (ArgException &e) {
		cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
		exit(-1);
	}
}

CCDArguments convertCCDToCCDImpute(CCDImputeArguments ccdImputeArgs){
	CCDArguments ccdArgs;

	ccdArgs.inFileName = ccdImputeArgs.inFileName;
	ccdArgs.outFileName = ccdImputeArgs.outFileName;
	ccdArgs.fileFormat = ccdImputeArgs.fileFormat;
	ccdArgs.useGPU = ccdImputeArgs.useGPU;
	ccdArgs.useBetterGPU = ccdImputeArgs.useBetterGPU;
	ccdArgs.deviceNumber = ccdImputeArgs.deviceNumber;
	ccdArgs.tolerance = ccdImputeArgs.tolerance;
	ccdArgs.hyperprior = ccdImputeArgs.hyperprior;
	ccdArgs.useNormalPrior = ccdImputeArgs.useNormalPrior;
	ccdArgs.hyperPriorSet = ccdImputeArgs.hyperPriorSet;
	ccdArgs.maxIterations = ccdImputeArgs.maxIterations;
	ccdArgs.convergenceTypeString = ccdImputeArgs.convergenceTypeString;
	ccdArgs.convergenceType = ccdImputeArgs.convergenceType;
	ccdArgs.seed = ccdImputeArgs.seed;
	ccdArgs.doCrossValidation = ccdImputeArgs.doCrossValidation;
	ccdArgs.lowerLimit = ccdImputeArgs.lowerLimit;
	ccdArgs.upperLimit = ccdImputeArgs.upperLimit;
	ccdArgs.fold = ccdImputeArgs.fold;
	ccdArgs.foldToCompute = ccdImputeArgs.foldToCompute;
	ccdArgs.gridSteps = ccdImputeArgs.gridSteps;
	ccdArgs.cvFileName = ccdImputeArgs.cvFileName;
	ccdArgs.doFitAtOptimal = ccdImputeArgs.doFitAtOptimal;
	ccdArgs.doBootstrap = ccdImputeArgs.doBootstrap;
	ccdArgs.reportRawEstimates = ccdImputeArgs.reportRawEstimates;
	ccdArgs.replicates = ccdImputeArgs.replicates;
	ccdArgs.bsFileName = ccdImputeArgs.bsFileName;
	ccdArgs.doPartial = ccdImputeArgs.doPartial;
	ccdArgs.modelType = ccdImputeArgs.modelType;
	ccdArgs.modelName = ccdImputeArgs.modelName;

	return ccdArgs;
}

} // namespace

int main(int argc, char* argv[]) {

	using namespace bsccs;

	CCDImputeArguments ccdImputeArgs;

	parseCommandLine(argc, argv, ccdImputeArgs);

	ImputeVariables imputation;
	CCDArguments ccdArgs = convertCCDToCCDImpute(ccdImputeArgs);
	imputation.initialize(ccdArgs, ccdImputeArgs.numberOfImputations, ccdImputeArgs.includeY);
	imputation.impute();
	return 0;
}
