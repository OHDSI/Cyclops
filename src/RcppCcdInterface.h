/*
 * RcppCcdInterface.h
 *
 * @author Marc Suchard
 */

#ifndef RCPP_CCD_INTERFACE_H_
#define RCPP_CCD_INTERFACE_H_

#include "CcdInterface.h"
#include "priors/JointPrior.h"

namespace bsccs {

class RcppModelData; // forward reference

class RcppCcdInterface : public CcdInterface {

public:

	RcppCcdInterface();

    RcppCcdInterface(RcppModelData& modelData);

    virtual ~RcppCcdInterface();
    
    double initializeModel() {
//    	ModelData* tmp;
    	return CcdInterface::initializeModel(&modelData, &ccd, &modelSpecifics);
    }

    double fitModel() {
    	return CcdInterface::fitModel(ccd);
    }
        
    double runFitMLEAtMode() {
    	return CcdInterface::runFitMLEAtMode(ccd);
    }

    double predictModel() {
    	return CcdInterface::predictModel(ccd, modelData);
    }

    double profileModel(/*ProfileInformationMap &profileMap*/) {
    	return CcdInterface::profileModel(ccd, modelData, profileMap);
    }
           
    double runCrossValidation() {
    	return CcdInterface::runCrossValidation(ccd, modelData);
    }           
        
    double runBoostrap(std::vector<double>& savedBeta) {
    	return CcdInterface::runBoostrap(ccd, modelData, savedBeta);
    }         		
    
    void setZeroBetaAsFixed() {
    	CcdInterface::setZeroBetaAsFixed(ccd);
    }            
        
    double logModel(/*ProfileInformationMap &profileMap,*/ bool withProfileBounds) {
    	return CcdInterface::logModel(ccd, modelData, profileMap, withProfileBounds);
    }    
        
    double diagnoseModel(double loadTime, double updateTime) {
    	return CcdInterface::diagnoseModel(ccd, modelData, loadTime, updateTime);
    }
    
    const Rcpp::List& getResult() const {
    	return result;
    }
    
	void setPrior(
				const std::string& basePriorName, 
				double baseVariance,
				const ProfileVector& flatPrior);    
    
    
    static void appendRList(Rcpp::List& list, const Rcpp::List& append);
    
    static Models::ModelType parseModelType(const std::string& modelName);
    static priors::PriorType parsePriorType(const std::string& priorName);
    static ConvergenceType parseConvergenceType(const std::string& convergenceName);
                        
protected:            
		
		static void handleError(const std::string& str);
		
		priors::JointPriorPtr makePrior(
				const std::string& basePriorName, 
				double baseVariance,
				const ProfileVector& flatPrior);
            
    void initializeModelImpl(
            ModelData** modelData,
            CyclicCoordinateDescent** ccd,
            AbstractModelSpecifics** model);
            
    void predictModelImpl(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData); 
            
    void logModelImpl(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData,
            ProfileInformationMap& profileMap,
            bool withASE); 
            
     void diagnoseModelImpl(
            CyclicCoordinateDescent *ccd, 
            ModelData *modelData,	
    		double loadTime,
    		double updateTime);  

private:

			RcppModelData& rcppModelData; // TODO Make const?
			
//      Rcpp::XPtr<RcppModelData> data;
//      Rcpp::XPtr<CyclicCoordinateDescent> ccd;
//      Rcpp::XPtr<AbstractModelSpecifics> model;
	 		
	 		ModelData* modelData;
			CyclicCoordinateDescent* ccd;
			AbstractModelSpecifics* modelSpecifics;
						
			bsccs::ProfileInformationMap profileMap;
			Rcpp::List result;

}; // class RcppCcdInterface

} // namespace

#endif /* CCD_H_ */
