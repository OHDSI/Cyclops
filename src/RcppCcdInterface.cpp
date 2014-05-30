/*
 * RcppCcdInterface.cpp
 *
 * @author Marc Suchard
 */
 
#include <sstream>
 
#include "Rcpp.h"
#include "RcppCcdInterface.h"
#include "RcppModelData.h"
#include "engine/ModelSpecifics.h"
#include "priors/JointPrior.h"
#include "CyclicCoordinateDescent.h"
#include "io/OutputWriter.h"
#include "RcppOutputHelper.h"
#include "RcppProgressLogger.h"

// Rcpp export code

using namespace Rcpp;

//// [[Rcpp::export]]
//List ccd_hello_world() {
//
//   using namespace bsccs;
//    CharacterVector x = CharacterVector::create( "foo", "bar" )  ;
//    NumericVector y   = NumericVector::create( 0.0, 1.0 ) ;
//    XPtr<RcppCcdInterface> ptr(new RcppCcdInterface());
//    List z            = List::create( x, y, ptr ) ;
////     List z = List::create(x, y);
//
//    return z ;
//}

// [[Rcpp::export]]
List ccdInitializeModelImpl(SEXP inModelData) {

	using namespace bsccs;

	XPtr<RcppModelData> rcppModelData(inModelData);
	XPtr<RcppCcdInterface> interface(
		new RcppCcdInterface(*rcppModelData));
	
	interface->getArguments().modelName = "ls"; // TODO Pass as argument
	
	double timeInit = interface->initializeModel();
	std::cout << "Done init" << std::endl;
	double timeUpdate = interface->fitModel();
	std::cout << "Done fit" << std::endl;
//	
//	bsccs::ProfileInformationMap profileMap;
//	// TODO Profile
//	bool withASE = false; //arguments.fitMLEAtMode || arguments.computeMLE || arguments.reportASE;    
//	double timeLogModel = interface->logModel(profileMap, withASE);
//	std::cout << "Done log model" << std::endl;
	
	List list = List::create(interface, rcppModelData, timeInit);
	return list;
}

namespace bsccs {

void RcppCcdInterface::handleError(const std::string& str) {	
	::Rf_error(str.c_str()); 
//	std::cerr << str << std::endl;
//	exit(-1);
}

// TODO Massive code duplicate (to remove) with CmdLineCcdInterface
void RcppCcdInterface::initializeModelImpl(
		ModelData** modelData,
		CyclicCoordinateDescent** ccd,
		AbstractModelSpecifics** model) {
	 	
	 *modelData = &rcppModelData;
	 
	// Parse type of model 
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
 		handleError("Invalid model type."); 		
 	}
 
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

  loggers::ProgressLoggerPtr logger = bsccs::make_shared<loggers::RcppProgressLogger>();
  loggers::ErrorHandlerPtr error = bsccs::make_shared<loggers::RcppErrorHandler>();

 
 	*ccd = new CyclicCoordinateDescent(*modelData /* TODO Change to ref */, **model, prior, logger, error);
 
 #ifdef CUDA
 	}
 #endif
 
 	(*ccd)->setNoiseLevel(arguments.noiseLevel);

}

void RcppCcdInterface::predictModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData) {

// 	struct timeval time1, time2;
// 	gettimeofday(&time1, NULL);
// 
// // 	bsccs::PredictionOutputWriter predictor(*ccd, *modelData);
// // 
// // 	string fileName = getPathAndFileName(arguments, "pred_");
// // 
// // 	predictor.writeFile(fileName.c_str());
// 
// 	gettimeofday(&time2, NULL);
// 	return calculateSeconds(time1, time2);
}

//  	struct Test { //: public std::ostream {  		
//  		//std::ostream os;  		
//  		std::stringstream s;
//  	};
  	
//  	template <typename T>
//  	Test& operator<<(Test& test, const T& obj) {
//  		return test;
//  	}
//  	
//  	template <>
//  	Test& operator<<(Test& test, const std::string& string) {  		
//  		return test;
//  	}
  	
//  	template <>
//  	Test& operator<<(Test& test, const char* string) {
//  		return test;
//  	}
	    
void RcppCcdInterface::logModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData,
	    ProfileInformationMap& profileMap, bool withASE) {
 		
 		// TODO Move into super-class
  	EstimationOutputWriter estimates(*ccd, *modelData);
  	estimates.addBoundInformation(profileMap);
  	// End move

		OutputHelper::RcppOutputHelper test(result);  	
  	estimates.writeStream(test);		
}

void RcppCcdInterface::diagnoseModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData,	
		double loadTime,
		double updateTime) {

// 	struct timeval time1, time2;
// 	gettimeofday(&time1, NULL);
// 	
// //	using namespace bsccs;	
// // 	DiagnosticsOutputWriter diagnostics(*ccd, *modelData);
// // 
// // 	string fileName = getPathAndFileName(arguments, "diag_");
// // 
// // 	vector<ExtraInformation> extraInfo;
// // 	extraInfo.push_back(ExtraInformation("load_time",loadTime));
// // 	extraInfo.push_back(ExtraInformation("update_time",updateTime));
// // 
// // 	diagnostics.addExtraInformation(extraInfo);
// // 	diagnostics.writeFile(fileName.c_str());
// 
// 
// 	gettimeofday(&time2, NULL);
// 	return calculateSeconds(time1, time2);

}

RcppCcdInterface::RcppCcdInterface(RcppModelData& _rcppModelData) 
	: rcppModelData(_rcppModelData), modelData(NULL), ccd(NULL), modelSpecifics(NULL) {
	// Do nothing
}

//RcppCcdInterface::RcppCcdInterface() {
//    // Do nothing
//}

RcppCcdInterface::~RcppCcdInterface() {
	std::cout << "~RcppCcdInterface() called." << std::endl;
	if (ccd) delete ccd;
	if (modelSpecifics) delete modelSpecifics;		
	// Do not delete modelData
}

} // namespace

