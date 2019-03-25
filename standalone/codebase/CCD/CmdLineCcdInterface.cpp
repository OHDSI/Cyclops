/*
 * CmdLineCcdInterface.cpp
 *
 *  Created on: April 15, 2014
 *      Author: Marc Suchard
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

#include <cmath>
// #include <math.h>

// #include "Types.h"
#include "CmdLineCcdInterface.h"
// #include "CyclicCoordinateDescent.h"
// #include "ModelData.h"
#include "io/InputReader.h"
#include "io/HierarchyReader.h"
//#include "io/CLRInputReader.h"
//#include "io/RTestInputReader.h"
//#include "io/CCTestInputReader.h"
//#include "io/CoxInputReader.h"
//#include "io/SCCSInputReader.h"
#include "io/NewCLRInputReader.h"
#include "io/NewSCCSInputReader.h"
#include "io/NewCoxInputReader.h"
#include "io/NewGenericInputReader.h"
#include "io/BBRInputReader.h"
#include "io/OutputWriter.h"
// #include "drivers/CrossValidationSelector.h"
// #include "drivers/GridSearchCrossValidationDriver.h"
// #include "drivers/HierarchyGridSearchCrossValidationDriver.h"
// #include "drivers/AutoSearchCrossValidationDriver.h"
// #include "drivers/HierarchyAutoSearchCrossValidationDriver.h"
// #include "drivers/BootstrapSelector.h"
// #include "drivers/ProportionSelector.h"
// #include "drivers/BootstrapDriver.h"
//#include "engine/ModelSpecifics.h"
#include "io/CmdLineProgressLogger.h"
// 
#include "tclap/CmdLine.h"
// #include "utils/RZeroIn.h"

//#include <R.h>

// #ifdef CUDA
// 	#include "GPUCyclicCoordinateDescent.h"
// //	#include "BetterGPU.h"
// #endif


namespace bsccs {

using namespace TCLAP;
using namespace std;

void CmdLineCcdInterface::parseCommandLine(std::vector<std::string>& args) {

	setDefaultArguments();

	try {
		CmdLine cmd("Cyclic coordinate descent algorithm for self-controlled case studies", ' ', "0.1");
		ValueArg<int> gpuArg("g","GPU","Use GPU device", arguments.useGPU, -1, "device #");
//		SwitchArg betterGPUArg("1","better", "Use better GPU implementation", false);
		ValueArg<int> maxIterationsArg("", "maxIterations", "Maximum iterations", false, arguments.modeFinding.maxIterations, "int");
		UnlabeledValueArg<string> inFileArg("inFileName","Input file name", true, arguments.inFileName, "inFileName");
		UnlabeledValueArg<string> outFileArg("outFileName","Output file name", true, arguments.outFileName, "outFileName");
		ValueArg<string> outDirectoryNameArg("", "outDirectoryName", "Output directory name", false, arguments.outDirectoryName, "outDirectoryName");


		// Prior arguments
		ValueArg<double> hyperPriorArg("v", "variance", "Hyperprior variance", false, arguments.hyperprior, "real");
		SwitchArg normalPriorArg("n", "normalPrior", "Use normal prior, default is laplace", arguments.useNormalPrior);
		SwitchArg computeMLEArg("", "MLE", "Compute maximum likelihood estimates only", arguments.computeMLE);
		SwitchArg computeMLEAtModeArg("", "MLEAtMode", "Compute maximum likelihood estimates at posterior mode", arguments.fitMLEAtMode);
		SwitchArg reportASEArg("","ASE", "Compute asymptotic standard errors at posterior mode", arguments.reportASE);

		//Hierarchy arguments
		SwitchArg useHierarchyArg("", "hier", "Use hierarchy in analysis", arguments.useHierarchy);
		ValueArg<string> hierarchyFileArg("a", "hierarchyFile", "Hierarchy file name", false, "noFileName", "hierarchyFile");
		ValueArg<double> classHierarchyVarianceArg("d","classHierarchyVariance","Variance for drug class hierarchy", false, 10, "Variance at the class level of the hierarchy");


		// Convergence criterion arguments
		ValueArg<double> toleranceArg("t", "tolerance", "Convergence criterion tolerance", false, arguments.modeFinding.tolerance, "real");
//		SwitchArg zhangOlesConvergenceArg("z", "zhangOles", "Use Zhange-Oles convergence criterion, default is true", true);
		std::vector<std::string> allowedConvergence;
		allowedConvergence.push_back("gradient");
		allowedConvergence.push_back("ZhangOles");
		allowedConvergence.push_back("Lange");
		allowedConvergence.push_back("Mittal");
		ValuesConstraint<std::string> allowedConvergenceValues(allowedConvergence);
		ValueArg<string> convergenceArg("", "convergence", "Convergence criterion", false, arguments.modeFinding.convergenceTypeString, &allowedConvergenceValues);

		ValueArg<long> seedArg("s", "seed", "Random number generator seed", false, arguments.seed, "long");

		// Cross-validation arguments
		SwitchArg doCVArg("c", "cv", "Perform cross-validation selection of hyperprior variance", arguments.crossValidation.doCrossValidation);
		SwitchArg useAutoSearchCVArg("", "auto", "Use an auto-search when performing cross-validation", arguments.crossValidation.useAutoSearchCV);
		ValueArg<double> lowerCVArg("l", "lower", "Lower limit for cross-validation search", false, arguments.crossValidation.lowerLimit, "real");
		ValueArg<double> upperCVArg("u", "upper", "Upper limit for cross-validation search", false, arguments.crossValidation.upperLimit, "real");
		ValueArg<int> foldCVArg("f", "fold", "Fold level for cross-validation", false, arguments.crossValidation.fold, "int");
		ValueArg<int> gridCVArg("", "gridSize", "Uniform grid size for cross-validation search", false, arguments.crossValidation.gridSteps, "int");
		ValueArg<int> foldToComputeCVArg("", "computeFold", "Number of fold to iterate, default is 'fold' value", false, arguments.crossValidation.foldToCompute, "int");
		ValueArg<string> outFile2Arg("", "cvFileName", "Cross-validation output file name", false, arguments.crossValidation.cvFileName, "cvFileName");

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

		MultiArg<long> profileCIArg("","profileCI", "Report confidence interval for covariate", false, "integer");
		MultiArg<long> flatPriorArg("","flat", "Place no prior on covariate", false, "integer");


		cmd.add(gpuArg);
//		cmd.add(betterGPUArg);
		cmd.add(toleranceArg);
		cmd.add(maxIterationsArg);
		cmd.add(hyperPriorArg);
		cmd.add(normalPriorArg);
		cmd.add(computeMLEArg);
		cmd.add(computeMLEAtModeArg);
		cmd.add(reportASEArg);
//		cmd.add(zhangOlesConvergenceArg);
		cmd.add(convergenceArg);
		cmd.add(seedArg);
		cmd.add(modelArg);
		cmd.add(formatArg);
		cmd.add(outputFormatArg);
		cmd.add(profileCIArg);
		cmd.add(flatPriorArg);

		//Hierarchy arguments
		cmd.add(useHierarchyArg);
		cmd.add(hierarchyFileArg);
		cmd.add(classHierarchyVarianceArg);

		cmd.add(doCVArg);
		cmd.add(useAutoSearchCVArg);
		cmd.add(lowerCVArg);
		cmd.add(upperCVArg);
		cmd.add(foldCVArg);
		cmd.add(gridCVArg);
		cmd.add(foldToComputeCVArg);
		cmd.add(outFile2Arg);
		cmd.add(outDirectoryNameArg);

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
		arguments.outDirectoryName = outDirectoryNameArg.getValue();
		arguments.modeFinding.tolerance = toleranceArg.getValue();
		arguments.modeFinding.maxIterations = maxIterationsArg.getValue();
		arguments.hyperprior = hyperPriorArg.getValue();
		arguments.useNormalPrior = normalPriorArg.getValue();
		arguments.computeMLE = computeMLEArg.getValue();
		arguments.fitMLEAtMode = computeMLEAtModeArg.getValue();
		arguments.reportASE = reportASEArg.getValue();
		arguments.seed = seedArg.getValue();

		//Hierarchy arguments
		arguments.useHierarchy = useHierarchyArg.isSet();
		arguments.hierarchyFileName = hierarchyFileArg.getValue(); // Hierarchy argument
		arguments.classHierarchyVariance = classHierarchyVarianceArg.getValue(); //Hierarchy argument

		arguments.modelName = modelArg.getValue();
		arguments.fileFormat = formatArg.getValue();
		arguments.outputFormat = outputFormatArg.getValue();
		if (arguments.outputFormat.size() == 0) {
			arguments.outputFormat.push_back("estimates");
		}
						
		for (int i = 0; i < profileCIArg.getValue().size(); ++i) {
		    arguments.profileCI.push_back(profileCIArg.getValue()[i]);
		}
		
		for (int i = 0; i < flatPriorArg.getValue().size(); ++i) {
		    arguments.flatPrior.push_back(flatPriorArg.getValue()[i]);
		}		

		arguments.modeFinding.convergenceTypeString = convergenceArg.getValue();

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
		if (arguments.modeFinding.convergenceTypeString == "ZhangOles") {
			arguments.modeFinding.convergenceType = ZHANG_OLES;
		} else if (arguments.modeFinding.convergenceTypeString == "Lange") {
			arguments.modeFinding.convergenceType = LANGE;
		} else if (arguments.modeFinding.convergenceTypeString == "Mittal") {
			arguments.modeFinding.convergenceType = MITTAL;
		} else if (arguments.modeFinding.convergenceTypeString == "gradient") {
			arguments.modeFinding.convergenceType = GRADIENT;
		} else {
			cerr << "Unknown convergence type: " << convergenceArg.getValue() << " " << arguments.modeFinding.convergenceTypeString << endl;
			exit(-1);
		}

		// Cross-validation
		arguments.crossValidation.doCrossValidation = doCVArg.isSet();
		if (arguments.crossValidation.doCrossValidation) {
			arguments.crossValidation.useAutoSearchCV = useAutoSearchCVArg.isSet();
			arguments.crossValidation.lowerLimit = lowerCVArg.getValue();
			arguments.crossValidation.upperLimit = upperCVArg.getValue();
			arguments.crossValidation.fold = foldCVArg.getValue();
			arguments.crossValidation.gridSteps = gridCVArg.getValue();
			if(foldToComputeCVArg.isSet()) {
				arguments.crossValidation.foldToCompute = foldToComputeCVArg.getValue();
			} else {
				arguments.crossValidation.foldToCompute = arguments.crossValidation.fold;
			}
			arguments.crossValidation.cvFileName = outFile2Arg.getValue();
			arguments.crossValidation.doFitAtOptimal = true;
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

void CmdLineCcdInterface::initializeModelImpl(
		AbstractModelData** modelData,
		CyclicCoordinateDescent** ccd,
		AbstractModelSpecifics** model) {
	
	// TODO Break up function; too long

	cout << "Running CCD (" <<
#ifdef DOUBLE_PRECISION
	"double"
#else
	"single"
#endif
	"-precision) ..." << endl;

	// Parse type of model
	//using namespace bsccs::Models;
	ModelType modelType;
	if (arguments.modelName == "sccs") {
		modelType = ModelType::SELF_CONTROLLED_MODEL;
	} else if (arguments.modelName == "clr") {
		modelType = ModelType::CONDITIONAL_LOGISTIC;
	} else if (arguments.modelName == "lr") {
		modelType = ModelType::LOGISTIC;
	} else if (arguments.modelName == "ls") {
		modelType = ModelType::NORMAL;
	} else if (arguments.modelName == "pr") {
		modelType = ModelType::POISSON;
	} else if (arguments.modelName == "cox") {
		modelType = ModelType::COX;
	} else {
		cerr << "Invalid model type." << endl;
		exit(-1);
	}

	InputReader* reader;
	logger = bsccs::make_shared<loggers::CoutLogger>();
	error = bsccs::make_shared<loggers::CerrErrorHandler>();
	if (arguments.fileFormat == "sccs") {
//		reader = new SCCSInputReader();
		cerr << "Invalid file format." << endl;
		exit(-1);
	} else if (arguments.fileFormat == "clr") {
		reader = new NewCLRInputReader(logger, error);
// 	} else if (arguments.fileFormat == "csv") {
// 		reader = new RTestInputReader();
// 	} else if (arguments.fileFormat == "cc") {
// 		reader = new CCTestInputReader();
// 	} else if (arguments.fileFormat == "cox-csv") {
// 		reader = new CoxInputReader();
	} else if (arguments.fileFormat == "bbr") {
		reader = new BBRInputReader<NoImputation>();
	} else if (arguments.fileFormat == "generic") {
		reader = new NewGenericInputReader(modelType, logger, error);
	} else if (arguments.fileFormat == "new-cox") {
		reader = new NewCoxInputReader(logger, error);
	} else {
		cerr << "Invalid file format." << endl;
		exit(-1);
	}

	reader->readFile(arguments.inFileName.c_str()); // TODO Check for error
	// delete reader;
	*modelData = reader->getModelData();

// 	switch (modelType) {
// 		case bsccs::Models::SELF_CONTROLLED_MODEL :
// 			*model = new ModelSpecifics<SelfControlledCaseSeries<real>,real>(**modelData);
// 			break;
// 		case bsccs::Models::CONDITIONAL_LOGISTIC :
// 			*model = new ModelSpecifics<ConditionalLogisticRegression<real>,real>(**modelData);
// 			break;
// 		case bsccs::Models::LOGISTIC :
// 			*model = new ModelSpecifics<LogisticRegression<real>,real>(**modelData);
// 			break;
// 		case bsccs::Models::NORMAL :
// 			*model = new ModelSpecifics<LeastSquares<real>,real>(**modelData);
// 			break;
// 		case bsccs::Models::POISSON :
// 			*model = new ModelSpecifics<PoissonRegression<real>,real>(**modelData);
// 			break;
// 		case bsccs::Models::COX :
// 			*model = new ModelSpecifics<CoxProportionalHazards<real>,real>(**modelData);
// 			break;
// 		default:
// 			cerr << "Invalid model type." << endl;
// 			exit(-1);
// 	}

	*model = AbstractModelSpecifics::factory(modelType, **modelData,
			DeviceType::CPU, "");
	if (*model == nullptr) {
		cerr << "Invalid model type." << endl;
		exit(-1);
	}

#ifdef CUDA
	if (arguments.useGPU) {
		*ccd = new GPUCyclicCoordinateDescent(arguments.deviceNumber, *reader, **model);
	} else {
#endif

	// Hierarchy management
	HierarchyReader* hierarchyData;
	if (arguments.useHierarchy) {
		hierarchyData = new HierarchyReader(arguments.hierarchyFileName.c_str(), *modelData);
	}


	using namespace bsccs::priors;
	PriorPtr singlePrior;
	if (arguments.useNormalPrior) {
		singlePrior = std::make_shared<NormalPrior>(arguments.hyperprior);
	} else if (arguments.computeMLE) {
		if (arguments.fitMLEAtMode) {
			cerr << "Unable to compute MLE at posterior mode, if mode is not first explored." << endl;
			exit(-1);
		}
		singlePrior = std::make_shared<NoPrior>();
	} else {
		singlePrior = std::make_shared<LaplacePrior>(arguments.hyperprior);
	}
	//singlePrior->setVariance(arguments.hyperprior);

	JointPriorPtr prior;
	if (arguments.flatPrior.size() == 0) {
		prior = std::make_shared<FullyExchangeableJointPrior>(singlePrior);
	} else {
		const int length =  (*modelData)->getNumberOfCovariates();
		std::shared_ptr<MixtureJointPrior> mixturePrior = std::make_shared<MixtureJointPrior>(
						singlePrior, length
				);

		PriorPtr noPrior = std::make_shared<NoPrior>();
		for (ProfileVector::const_iterator it = arguments.flatPrior.begin();
				it != arguments.flatPrior.end(); ++it) {
			int index = (*modelData)->getColumnIndexByName(*it);
			if (index == -1) {
				cerr << "Variable " << *it << " not found." << endl;
			} else {
				mixturePrior->changePrior(noPrior, index);
			}
		}
		prior = mixturePrior;
	}

	//Hierarchy prior
	if (arguments.useHierarchy) {
		std::shared_ptr<HierarchicalJointPrior> hierarchicalPrior = std::make_shared<HierarchicalJointPrior>(singlePrior, 2); //Depth of hierarchy fixed at 2 right now
		PriorPtr classPrior = std::make_shared<NormalPrior>(arguments.hyperprior);
		hierarchicalPrior->changePrior(classPrior,1);
        hierarchicalPrior->setHierarchy(
                hierarchyData->returnGetParentMap(),
                hierarchyData->returnGetChildMap()
            );
		hierarchicalPrior->setVariance(0,arguments.hyperprior);
		hierarchicalPrior->setVariance(1,arguments.classHierarchyVariance);
		prior = hierarchicalPrior;
	}
    
	*ccd = new CyclicCoordinateDescent(
 		**modelData /* TODO Change to ref */, 
// 					bsccs::shared_ptr<ModelData>(*modelData),
					**model, prior, logger, error);

#ifdef CUDA
	}
#endif

	(*ccd)->setNoiseLevel(arguments.noiseLevel);

	cout << "Everything loaded and ready to run ..." << endl;	
}

void CmdLineCcdInterface::predictModelImpl(CyclicCoordinateDescent *ccd, AbstractModelData *modelData) {

	bsccs::PredictionOutputWriter predictor(*ccd, *modelData);
	string fileName = getPathAndFileName(arguments, "pred_");
	predictor.writeFile(fileName.c_str());
}
	    
void CmdLineCcdInterface::logModelImpl(CyclicCoordinateDescent *ccd, AbstractModelData *modelData,
	    ProfileInformationMap& profileMap, bool withASE) {

	using namespace bsccs;
	bsccs::EstimationOutputWriter estimates(*ccd, *modelData);
	estimates.addBoundInformation(profileMap);

	string fileName = getPathAndFileName(arguments, "est_");
	estimates.writeFile(fileName.c_str());
}

void CmdLineCcdInterface::diagnoseModelImpl(CyclicCoordinateDescent *ccd, AbstractModelData *modelData,	
		double loadTime,
		double updateTime) {

	using namespace bsccs;
	DiagnosticsOutputWriter diagnostics(*ccd, *modelData);

	string fileName = getPathAndFileName(arguments, "diag_");

	vector<ExtraInformation> extraInfo;
	extraInfo.push_back(ExtraInformation("load_time",loadTime));
	extraInfo.push_back(ExtraInformation("update_time",updateTime));

	diagnostics.addExtraInformation(extraInfo);
	diagnostics.writeFile(fileName.c_str());
}

CmdLineCcdInterface::CmdLineCcdInterface(int argc, char* argv[]) {
    std::vector<std::string> args;
	for (int i = 0; i < argc; i++)
		args.push_back(argv[i]);
	parseCommandLine(args);
}

CmdLineCcdInterface::~CmdLineCcdInterface() {
    // Do nothing
}

} // namespace

