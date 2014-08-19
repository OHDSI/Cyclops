/*
 * RcppCcdInterface.cpp
 *
 * @author Marc Suchard
 */
 
#include <sstream>
 
#include "Rcpp.h"
#include "RcppCyclopsInterface.h"
#include "RcppModelData.h"
//#include "engine/ModelSpecifics.h"
#include "priors/JointPrior.h"
#include "CyclicCoordinateDescent.h"
#include "io/OutputWriter.h"
#include "RcppOutputHelper.h"
#include "RcppProgressLogger.h"

// Rcpp export code

using namespace Rcpp;

// // [[Rcpp::export("test")]]
// size_t ccdTest(SEXP exp) {
// 	if (Rf_isNull(exp)) {	
// 		return 0;
// 	}
// 	std::vector<std::string> strings = as<std::vector<std::string> >(exp);
// 	return strings.size();
// }

// [[Rcpp::export(.cyclopsSetBeta)]]
void cyclopsSetBeta(SEXP inRcppCcdInterface, int beta, double value) {
    using namespace bsccs;
    XPtr<RcppCcdInterface> interface(inRcppCcdInterface);    
    
    interface->getCcd().setBeta(beta - 1, value);
}

// [[Rcpp::export(.cyclopsSetFixedBeta)]]
void cyclopsSetFixedBeta(SEXP inRcppCcdInterface, int beta, bool fixed) {
    using namespace bsccs;
    XPtr<RcppCcdInterface> interface(inRcppCcdInterface);    
    
    interface->getCcd().setFixedBeta(beta - 1, fixed);
}

// [[Rcpp::export(".cyclopsGetIsRegularized")]]
bool cyclopsGetIsRegularized(SEXP inRcppCcdInterface, const int index) {
    using namespace bsccs;
    XPtr<RcppCcdInterface> interface(inRcppCcdInterface);
    return interface->getCcd().getIsRegularized(index);
}

// [[Rcpp::export(".cyclopsGetLogLikelihood")]]
double cyclopsGetLogLikelihood(SEXP inRcppCcdInterface) {
	using namespace bsccs;
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);
	
	return interface->getCcd().getLogLikelihood();
}

// [[Rcpp::export(".cyclopsGetFisherInformation")]]
Eigen::MatrixXd cyclopsGetFisherInformation(SEXP inRcppCcdInterface, const SEXP sexpCovariates) {
	using namespace bsccs;
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);
	
// 	const int p = interface->getCcd().getBetaSize();
// 	std::vector<size_t> indices;
// 	for (int i = 0; i < p; ++i) indices.push_back(i);

    std::vector<size_t> indices;
    if (!Rf_isNull(sexpCovariates)) {
	
    	ProfileVector covariates = as<ProfileVector>(sexpCovariates);    	
    	for (auto it = covariates.begin(); it != covariates.end(); ++it) {
	        size_t index = interface->getModelData().getColumnIndex(*it);
	        indices.push_back(index);	        
	    }
	}
	
    return interface->getCcd().computeFisherInformation(indices);
}

// [[Rcpp::export(".cyclopsSetPrior")]]
void cyclopsSetPrior(SEXP inRcppCcdInterface, const std::string& priorTypeName, double variance, SEXP excludeNumeric) {
	using namespace bsccs;
	
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);
	
//	priors::PriorType priorType = RcppCcdInterface::parsePriorType(priorTypeName);
 	ProfileVector exclude;
 	if (!Rf_isNull(excludeNumeric)) {
 		exclude = as<ProfileVector>(excludeNumeric);
 	}
 	
  interface->setPrior(priorTypeName, variance, exclude);
}

// [[Rcpp::export(".cyclopsProfileModel")]]
List cyclopsProfileModel(SEXP inRcppCcdInterface, SEXP sexpCovariates, double threshold, bool override) {
	using namespace bsccs;
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);
	
	if (!Rf_isNull(sexpCovariates)) {
		ProfileVector covariates = as<ProfileVector>(sexpCovariates);
		
		ProfileInformationMap profileMap;
        interface->profileModel(covariates, profileMap, threshold, override);
                
        std::vector<double> lower;
        std::vector<double> upper;
        std::vector<int> evals;
        
        for (ProfileVector::const_iterator it = covariates.begin();
        		it != covariates.end(); ++it) {        
            ProfileInformation info = profileMap[*it];
            lower.push_back(info.lower95Bound);
            upper.push_back(info.upper95Bound);
            evals.push_back(info.evaluations);
            
 //           std::cout << *it << " " << info.lower95Bound << " " << info.upper95Bound << std::endl;
        }
        return List::create(
            Rcpp::Named("covariate") = covariates,
            Rcpp::Named("lower") = lower,
            Rcpp::Named("upper") = upper,
            Rcpp::Named("evaluations") = evals 
        );
	}
        	
	return List::create();
}

// [[Rcpp::export(".cyclopsPredictModel")]]
List cyclopsPredictModel(SEXP inRcppCcdInterface) {
	using namespace bsccs;
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);
	double timePredict = interface->predictModel();
	
	List list = List::create(			
			Rcpp::Named("timePredict") = timePredict
		);
	RcppCcdInterface::appendRList(list, interface->getResult());
	return list;
}
	

// [[Rcpp::export(".cyclopsSetControl")]]
void cyclopsSetControl(SEXP inRcppCcdInterface, 
		int maxIterations, double tolerance, const std::string& convergenceType,
		bool useAutoSearch, int fold, int foldToCompute, double lowerLimit, double upperLimit, int gridSteps,
		const std::string& noiseLevel, int seed
		) {
	using namespace bsccs;
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);
	// Convergence control
	CCDArguments& args = interface->getArguments();
	interface->getArguments().maxIterations = maxIterations;
	interface->getArguments().tolerance = tolerance;
	interface->getArguments().convergenceType = RcppCcdInterface::parseConvergenceType(convergenceType);
	
	// Cross validation control
	args.useAutoSearchCV = useAutoSearch;
	args.fold = fold;
	args.foldToCompute = foldToCompute;
	args.lowerLimit = lowerLimit;
	args.upperLimit = upperLimit;
	args.gridSteps = gridSteps;
	NoiseLevels noise = RcppCcdInterface::parseNoiseLevel(noiseLevel);
	args.noiseLevel = noise;
	interface->setNoiseLevel(noise);
	args.seed = seed;
}

// [[Rcpp::export(".cyclopsRunCrossValidation")]]
List cyclopsRunCrossValidationl(SEXP inRcppCcdInterface) {	
	using namespace bsccs;
	
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);		
	double timeUpdate = interface->runCrossValidation();
		
	interface->diagnoseModel(0.0, 0.0);
	
	List list = List::create(
			Rcpp::Named("interface")=interface, 
			Rcpp::Named("timeFit")=timeUpdate
		);
	RcppCcdInterface::appendRList(list, interface->getResult());
	return list;
}

// [[Rcpp::export(".cyclopsFitModel")]]
List cyclopsFitModel(SEXP inRcppCcdInterface) {	
	using namespace bsccs;
	
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);		
	double timeUpdate = interface->fitModel();
	
	interface->diagnoseModel(0.0, 0.0);
	
	List list = List::create(
			Rcpp::Named("interface")=interface, 
			Rcpp::Named("timeFit")=timeUpdate
		);
	RcppCcdInterface::appendRList(list, interface->getResult());
	return list;
}

// [[Rcpp::export(".cyclopsLogModel")]]
List cyclopsLogModel(SEXP inRcppCcdInterface) {	
	using namespace bsccs;
	
	XPtr<RcppCcdInterface> interface(inRcppCcdInterface);	
	bool withASE = false;
	//return List::create(interface);
	
	double timeLogModel = interface->logModel(withASE);
	//std::cout << "getResult " << interface->getResult() << std::endl;
	
	CharacterVector names;
	names.push_back("interface");
	names.push_back("timeLog");
	CharacterVector oldNames = interface->getResult().attr("names");
	List list = List::create(interface, timeLogModel);
	for (int i = 0; i < interface->getResult().size(); ++i) {
		list.push_back(interface->getResult()[i]);
		names.push_back(oldNames[i]);
	}
	list.attr("names") = names;
	
	//, interface->getResult());	
	return list;
}

// [[Rcpp::export(".cyclopsInitializeModel")]]
List cyclopsInitializeModel(SEXP inModelData, const std::string& modelType, bool computeMLE = false) {
	using namespace bsccs;

	XPtr<RcppModelData> rcppModelData(inModelData);
	XPtr<RcppCcdInterface> interface(
		new RcppCcdInterface(*rcppModelData));
	
//	interface->getArguments().modelName = "ls"; // TODO Pass as argument	
	interface->getArguments().modelName = modelType;
	if (computeMLE) {
		interface->getArguments().computeMLE = true;
	}	
	double timeInit = interface->initializeModel();
	
//	bsccs::ProfileInformationMap profileMap;
//	// TODO Profile
//	bool withASE = false; //arguments.fitMLEAtMode || arguments.computeMLE || arguments.reportASE;    
//	double timeLogModel = interface->logModel(profileMap, withASE);
//	std::cout << "Done log model" << std::endl;
	
	List list = List::create(
			Rcpp::Named("interface") = interface, 
			Rcpp::Named("data") = rcppModelData, 
			Rcpp::Named("timeInit") = timeInit
		);
	return list;
}

namespace bsccs {
	
void RcppCcdInterface::appendRList(Rcpp::List& list, const Rcpp::List& append) {
	if (append.size() > 0) {
		CharacterVector names = list.attr("names");
		CharacterVector appendNames = append.attr("names");
		for (int i = 0; i < append.size(); ++i) {
			list.push_back(append[i]);
			names.push_back(appendNames[i]);
		}
		list.attr("names") = names;
	}		
}	

void RcppCcdInterface::handleError(const std::string& str) {	
//	Rcpp::stop(str); // TODO Want this to work
	::Rf_error(str.c_str());
}

bsccs::ConvergenceType RcppCcdInterface::parseConvergenceType(const std::string& convergenceName) {
	ConvergenceType type = GRADIENT;
	if (convergenceName == "gradient") {
		type = GRADIENT;
	} else if (convergenceName == "lange") {
		type = LANGE;
	} else if (convergenceName == "mittal") {
		type = MITTAL;
	} else if (convergenceName == "zhang") {
		type = ZHANG_OLES;
	} else {
		handleError("Invalid convergence type."); 	
	}
	return type;
}

bsccs::NoiseLevels RcppCcdInterface::parseNoiseLevel(const std::string& noiseName) {
	using namespace bsccs;
	NoiseLevels level = SILENT;
	if (noiseName == "silent") {
		level = SILENT; 
	} else if (noiseName == "quiet") {
		level = QUIET;
	} else if (noiseName == "noisy") {
		level = NOISY;
	} else {
		handleError("Invalid noise level.");
	}
	return level;
}

bsccs::priors::PriorType RcppCcdInterface::parsePriorType(const std::string& priorName) {
	using namespace bsccs::priors;
	bsccs::priors::PriorType priorType = NONE;
	if (priorName == "none") {
		priorType = NONE;
	} else if (priorName == "laplace") {
		priorType = LAPLACE;		
	} else if (priorName == "normal") {
		priorType = NORMAL;
	} else {
 		handleError("Invalid prior type."); 		
 	}	
 	return priorType;
}

bsccs::ModelType RcppCcdInterface::parseModelType(const std::string& modelName) {
	// Parse type of model 
 	bsccs::ModelType modelType =  bsccs::ModelType::LOGISTIC;
 	if (modelName == "sccs") {
 		modelType = bsccs::ModelType::SELF_CONTROLLED_MODEL;
 	} else if (modelName == "cpr") {
 		modelType = bsccs::ModelType::CONDITIONAL_POISSON;
 	} else if (modelName == "clr") {
 		modelType = bsccs::ModelType::CONDITIONAL_LOGISTIC;
 	} else if (modelName == "lr") {
 		modelType = bsccs::ModelType::LOGISTIC;
 	} else if (modelName == "ls") {
 		modelType = bsccs::ModelType::NORMAL;
 	} else if (modelName == "pr") {
 		modelType = bsccs::ModelType::POISSON;
 	} else if (modelName == "cox") {
 		modelType = bsccs::ModelType::COX;
 	} else if (modelName == "cox_raw") {
 		modelType = bsccs::ModelType::COX_RAW; 		
 	} else {
 		handleError("Invalid model type."); 		
 	}	
 	return modelType;
}

void RcppCcdInterface::setNoiseLevel(bsccs::NoiseLevels noiseLevel) {
    using namespace bsccs;
    ccd->setNoiseLevel(noiseLevel);
}

void RcppCcdInterface::setPrior(const std::string& basePriorName, double baseVariance,
		const ProfileVector& flatPrior) {			
	using namespace bsccs::priors;
	
	JointPriorPtr prior = makePrior(basePriorName, baseVariance, flatPrior);
	ccd->setPrior(prior);
}

priors::JointPriorPtr RcppCcdInterface::makePrior(const std::string& basePriorName, double baseVariance,
		const ProfileVector& flatPrior) {
	using namespace bsccs::priors;
	
 	PriorPtr singlePrior = bsccs::priors::CovariatePrior::makePrior(parsePriorType(basePriorName));
 	singlePrior->setVariance(baseVariance);
 
 	JointPriorPtr prior;
 	if (flatPrior.size() == 0) {
 		prior = bsccs::make_shared<FullyExchangeableJointPrior>(singlePrior);
 	} else {
 		const int length =  modelData->getNumberOfColumns();
 		bsccs::shared_ptr<MixtureJointPrior> mixturePrior = bsccs::make_shared<MixtureJointPrior>(
 						singlePrior, length
 				);
 
 		PriorPtr noPrior = bsccs::make_shared<NoPrior>();
 		for (ProfileVector::const_iterator it = flatPrior.begin();
 				it != flatPrior.end(); ++it) {
 			int index = modelData->getColumnIndexByName(*it);
 			if (index == -1) {
 				std::stringstream error;
 				error << "Variable " << *it << " not found.";
 				handleError(error.str()); 			
 			} else {
 				mixturePrior->changePrior(noPrior, index);
 			}
 		}
 		prior = mixturePrior;
 	}
 	return prior;
}
// TODO Massive code duplicate (to remove) with CmdLineCcdInterface
void RcppCcdInterface::initializeModelImpl(
		ModelData** modelData,
		CyclicCoordinateDescent** ccd,
		AbstractModelSpecifics** model) {
	 	
	 *modelData = &rcppModelData;
	 
	// Parse type of model 
	ModelType modelType = parseModelType(arguments.modelName);
// 	bsccs::Models::ModelType modelType;
// 	if (arguments.modelName == "sccs") {
// 		modelType = bsccs::Models::SELF_CONTROLLED_MODEL;
// 	} else if (arguments.modelName == "clr") {
// 		modelType = bsccs::Models::CONDITIONAL_LOGISTIC;
// 	} else if (arguments.modelName == "lr") {
// 		modelType = bsccs::Models::LOGISTIC;
// 	} else if (arguments.modelName == "ls") {
// 		modelType = bsccs::Models::NORMAL;
// 	} else if (arguments.modelName == "pr") {
// 		modelType = bsccs::Models::POISSON;
// 	} else if (arguments.modelName == "cox") {
// 		modelType = bsccs::Models::COX;
// 	} else {
// 		handleError("Invalid model type."); 		
// 	}
//  
//  	switch (modelType) {
//  		case bsccs::Models::SELF_CONTROLLED_MODEL :
//  			*model = new ModelSpecifics<SelfControlledCaseSeries<real>,real>(**modelData);
//  			break;
//  		case bsccs::Models::CONDITIONAL_LOGISTIC :
//  			*model = new ModelSpecifics<ConditionalLogisticRegression<real>,real>(**modelData);
//  			break;
//  		case bsccs::Models::LOGISTIC :
//  			*model = new ModelSpecifics<LogisticRegression<real>,real>(**modelData);
//  			break;
//  		case bsccs::Models::NORMAL :
//  			*model = new ModelSpecifics<LeastSquares<real>,real>(**modelData);
//  			break;
//  		case bsccs::Models::POISSON :
//  			*model = new ModelSpecifics<PoissonRegression<real>,real>(**modelData);
//  			break;
// 		case bsccs::Models::CONDITIONAL_POISSON :
//  			*model = new ModelSpecifics<ConditionalPoissonRegression<real>,real>(**modelData);
//  			break; 			
//  		case bsccs::Models::COX :
//  			*model = new ModelSpecifics<CoxProportionalHazards<real>,real>(**modelData);
//  			break;
//  		default:
//  			handleError("Invalid model type."); 			
//  	}

	*model = AbstractModelSpecifics::factory(modelType, *modelData);
	if (*model == nullptr) {
		handleError("Invalid model type.");
	}
 
 #ifdef CUDA
 	if (arguments.useGPU) {
 		*ccd = new GPUCyclicCoordinateDescent(arguments.deviceNumber, *reader, **model);
 	} else {
 #endif
 
 // Hierarchy management
// 	HierarchyReader* hierarchyData;
// 	if (arguments.useHierarchy) {
// 		hierarchyData = new HierarchyReader(arguments.hierarchyFileName.c_str(), *modelData);
// 	}
 
 
 	using namespace bsccs::priors;
 	PriorPtr singlePrior;
 	if (arguments.useNormalPrior) {
 		singlePrior = bsccs::make_shared<NormalPrior>();
 	} else if (arguments.computeMLE) {
 		if (arguments.fitMLEAtMode) {
 			handleError("Unable to compute MLE at posterior mode, if mode is not first explored."); 		
 		}
 		singlePrior = bsccs::make_shared<NoPrior>();
 	} else {
 		singlePrior = bsccs::make_shared<LaplacePrior>();
 	}
 	singlePrior->setVariance(arguments.hyperprior);
 
 	JointPriorPtr prior;
 	if (arguments.flatPrior.size() == 0) {
 		prior = bsccs::make_shared<FullyExchangeableJointPrior>(singlePrior);
 	} else {
 		const int length =  (*modelData)->getNumberOfColumns();
 		bsccs::shared_ptr<MixtureJointPrior> mixturePrior = bsccs::make_shared<MixtureJointPrior>(
 						singlePrior, length
 				);
 
 		PriorPtr noPrior = bsccs::make_shared<NoPrior>();
 		for (ProfileVector::const_iterator it = arguments.flatPrior.begin();
 				it != arguments.flatPrior.end(); ++it) {
 			int index = (*modelData)->getColumnIndexByName(*it);
 			if (index == -1) {
 				std::stringstream error;
 				error << "Variable " << *it << " not found.";
 				handleError(error.str()); 			
 			} else {
 				mixturePrior->changePrior(noPrior, index);
 			}
 		}
 		prior = mixturePrior;
 	}
 	 
 	//Hierarchy prior
// 	if (arguments.useHierarchy) {
// 		std::shared_ptr<HierarchicalJointPrior> hierarchicalPrior = std::make_shared<HierarchicalJointPrior>(singlePrior, 2); //Depth of hierarchy fixed at 2 right now
// 		PriorPtr classPrior = std::make_shared<NormalPrior>();
// 		hierarchicalPrior->changePrior(classPrior,1);
//         hierarchicalPrior->setHierarchy(
//                 hierarchyData->returnGetParentMap(),
//                 hierarchyData->returnGetChildMap()
//             );
// 		hierarchicalPrior->setVariance(0,arguments.hyperprior);
// 		hierarchicalPrior->setVariance(1,arguments.classHierarchyVariance);
// 		prior = hierarchicalPrior;
// 	}

  logger = bsccs::make_shared<loggers::RcppProgressLogger>();
  error = bsccs::make_shared<loggers::RcppErrorHandler>();
 
 	*ccd = new CyclicCoordinateDescent(*modelData /* TODO Change to ref */, **model, prior, logger, error);
 
 #ifdef CUDA
 	}
 #endif
 
 	(*ccd)->setNoiseLevel(arguments.noiseLevel);

}

void RcppCcdInterface::predictModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData) {

// 	bsccs::PredictionOutputWriter predictor(*ccd, *modelData);
// 
//     result = List::create();
//     OutputHelper::RcppOutputHelper test(result);
//     predictor.writeStream(test);
    
    NumericVector predictions(ccd->getPredictionSize());
    //std::vector<double> predictions(ccd->getPredictionSize());
    ccd->getPredictiveEstimates(&predictions[0], NULL);
    
    if (modelData->getHasRowLabels()) {
        size_t preds = ccd->getPredictionSize();
        CharacterVector labels(preds);
        for (size_t i = 0; i < preds; ++i) {
            labels[i] = modelData->getRowLabel(i);
        }
        predictions.names() = labels;    
    }
    result = List::create(
        Rcpp::Named("prediction") = predictions
    );
    
//     predictions.resize(ccd.getPredictionSize());
// 		ccd.getPredictiveEstimates(&predictions[0], NULL);
    
}
	    
void RcppCcdInterface::logModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData,
	    ProfileInformationMap& profileMap, bool withASE) {
 		
 		// TODO Move into super-class
  	EstimationOutputWriter estimates(*ccd, *modelData);
  	estimates.addBoundInformation(profileMap);
  	// End move
		
		result = List::create();
		OutputHelper::RcppOutputHelper out(result);  		
  	estimates.writeStream(out);	  	
}

void RcppCcdInterface::diagnoseModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData,	
		double loadTime,
		double updateTime) {
	
		result = List::create();
 		DiagnosticsOutputWriter diagnostics(*ccd, *modelData);
		OutputHelper::RcppOutputHelper test(result);  		
  	diagnostics.writeStream(test);	
}

RcppCcdInterface::RcppCcdInterface(RcppModelData& _rcppModelData) 
	: rcppModelData(_rcppModelData), modelData(NULL), ccd(NULL), modelSpecifics(NULL) {
	arguments.noiseLevel = SILENT; // Change default value from command-line version
}

//RcppCcdInterface::RcppCcdInterface() {
//    // Do nothing
//}

RcppCcdInterface::~RcppCcdInterface() {
//	std::cout << "~RcppCcdInterface() called." << std::endl;
	if (ccd) delete ccd;
	if (modelSpecifics) delete modelSpecifics;		
	// Do not delete modelData
}

} // namespace

