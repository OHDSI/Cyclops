/*
 * RcppCcdInterface.cpp
 *
 * @author Marc Suchard
 */
 
#include "Rcpp.h"
#include "RcppCcdInterface.h"

// Rcpp export code

using namespace Rcpp;

// [[Rcpp::export]]
List ccd_hello_world() {
   
   using namespace bsccs;
    CharacterVector x = CharacterVector::create( "foo", "bar" )  ;
    NumericVector y   = NumericVector::create( 0.0, 1.0 ) ;    
    XPtr<RcppCcdInterface> ptr(new RcppCcdInterface());    
    List z            = List::create( x, y, ptr ) ;
//     List z = List::create(x, y);
    
    return z ;
}

namespace bsccs {

void RcppCcdInterface::initializeModelImpl(
		ModelData** modelData,
		CyclicCoordinateDescent** ccd,
		AbstractModelSpecifics** model) {
	
// 	struct timeval time1, time2;
// 	gettimeofday(&time1, NULL);
// 
// Parse type of model
// 	//using namespace bsccs::Models;
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
// 		cerr << "Invalid model type." << endl;
// 		exit(-1);
// 	}
// 
// 	InputReader* reader;
// 	if (arguments.fileFormat == "sccs") {
// 		reader = new SCCSInputReader();
// 	} else if (arguments.fileFormat == "clr") {
// 		reader = new NewCLRInputReader();
// 	} else if (arguments.fileFormat == "csv") {
// 		reader = new RTestInputReader();
// 	} else if (arguments.fileFormat == "cc") {
// 		reader = new CCTestInputReader();
// 	} else if (arguments.fileFormat == "cox-csv") {
// 		reader = new CoxInputReader();
// 	} else if (arguments.fileFormat == "bbr") {
// 		reader = new BBRInputReader<NoImputation>();
// 	} else if (arguments.fileFormat == "generic") {
// 		reader = new NewGenericInputReader(modelType);
// 	} else if (arguments.fileFormat == "new-cox") {
// 		reader = new NewCoxInputReader();
// 	} else {
// 		cerr << "Invalid file format." << endl;
// 		exit(-1);
// 	}
// 
// 	reader->readFile(arguments.inFileName.c_str()); // TODO Check for error
// delete reader;
// 	*modelData = reader->getModelData();
// 
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
// 
// #ifdef CUDA
// 	if (arguments.useGPU) {
// 		*ccd = new GPUCyclicCoordinateDescent(arguments.deviceNumber, *reader, **model);
// 	} else {
// #endif
// 
// Hierarchy management
// 	HierarchyReader* hierarchyData;
// 	if (arguments.useHierarchy) {
// 		hierarchyData = new HierarchyReader(arguments.hierarchyFileName.c_str(), *modelData);
// 	}
// 
// 
// 	using namespace bsccs::priors;
// 	PriorPtr singlePrior;
// 	if (arguments.useNormalPrior) {
// 		singlePrior = std::make_shared<NormalPrior>();
// 	} else if (arguments.computeMLE) {
// 		if (arguments.fitMLEAtMode) {
// 			cerr << "Unable to compute MLE at posterior mode, if mode is not first explored." << endl;
// 			exit(-1);
// 		}
// 		singlePrior = std::make_shared<NoPrior>();
// 	} else {
// 		singlePrior = std::make_shared<LaplacePrior>();
// 	}
// 	singlePrior->setVariance(arguments.hyperprior);
// 
// 	JointPriorPtr prior;
// 	if (arguments.flatPrior.size() == 0) {
// 		prior = std::make_shared<FullyExchangeableJointPrior>(singlePrior);
// 	} else {
// 		const int length =  (*modelData)->getNumberOfColumns();
// 		std::shared_ptr<MixtureJointPrior> mixturePrior = std::make_shared<MixtureJointPrior>(
// 						singlePrior, length
// 				);
// 
// 		PriorPtr noPrior = std::make_shared<NoPrior>();
// 		for (ProfileVector::const_iterator it = arguments.flatPrior.begin();
// 				it != arguments.flatPrior.end(); ++it) {
// 			int index = (*modelData)->getColumnIndexByName(*it);
// 			if (index == -1) {
// 				cerr << "Variable " << *it << " not found." << endl;
// 			} else {
// 				mixturePrior->changePrior(noPrior, index);
// 			}
// 		}
// 		prior = mixturePrior;
// 	}
// 
// 	//Hierarchy prior
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
// 
// 	*ccd = new CyclicCoordinateDescent(*modelData /* TODO Change to ref */, **model, prior);
// 
// #ifdef CUDA
// 	}
// #endif
// 
// 	(*ccd)->setNoiseLevel(arguments.noiseLevel);
// 
// 	gettimeofday(&time2, NULL);
// 	double sec1 = calculateSeconds(time1, time2);
// 	
// 	return sec1;
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
	    
void RcppCcdInterface::logModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData,
	    ProfileInformationMap& profileMap, bool withASE) {

// 	struct timeval time1, time2;
// 	gettimeofday(&time1, NULL);
// 	
// //	using namespace bsccs;
// // 	bsccs::EstimationOutputWriter estimates(*ccd, *modelData);
// // 	estimates.addBoundInformation(profileMap);
// // 
// // 	string fileName = getPathAndFileName(arguments, "est_");
// // 	estimates.writeFile(fileName.c_str());
// 	
// 	gettimeofday(&time2, NULL);
// 	return calculateSeconds(time1, time2);				
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

RcppCcdInterface::RcppCcdInterface() {
    // Do nothing
}

RcppCcdInterface::~RcppCcdInterface() {
    // Do nothing
}

} // namespace

