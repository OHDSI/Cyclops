/*
 * RcppCcdInterface.h
 *
 * @author Marc Suchard
 */

#ifndef RCPP_CCD_INTERFACE_H_
#define RCPP_CCD_INTERFACE_H_

#include "CcdInterface.h"
#include "priors/JointPrior.h"
#include "priors/PriorFunction.h"

namespace bsccs {

// typedef std::vector<Rcpp::List> NeighborhoodMap;
// typedef Rcpp::List NeighborhoodMap;

class AbstractModelData; // forward reference

class RcppCcdInterface : public CcdInterface {

public:

	RcppCcdInterface();

    RcppCcdInterface(AbstractModelData& modelData);

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

    double profileModel(const ProfileVector& profileCI, ProfileInformationMap& profileMap,
            int threads,
            double threshold, bool override, bool includePenalty) {
    	return CcdInterface::profileModel(ccd, modelData, profileCI, profileMap, threads, threshold,
    			override, includePenalty);
    }

    double evaluateProfileModel(const IdType covariate,
                                const std::vector<double>& points,
                                std::vector<double>& values,
                                int threads, bool includePenalty) {
        return CcdInterface::evaluateProfileModel(ccd, modelData, covariate, points, values, threads, includePenalty);
    }

    double runCrossValidation() {
    	return CcdInterface::runCrossValidation(ccd, modelData);
    }

    double runBoostrap(std::vector<double>& savedBeta, std::string& treatmentId) {
    	return CcdInterface::runBoostrap(ccd, modelData, savedBeta, treatmentId);
    }

    void logResultsToFile(const std::string& fileName, bool withASE);

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

    void setParameterizedPrior(const std::vector<std::string>& priorName,
                              bsccs::priors::PriorFunctionPtr& priorFunctionPtr,
                              const ProfileVector& flatPrior);

	void setPrior(
				const std::vector<std::string>& basePriorName,
				const std::vector<double>& baseVariance,
				const ProfileVector& flatPrior,
				const HierarchicalChildMap& map,
				const NeighborhoodMap& neighborhood);

    void setNoiseLevel(bsccs::NoiseLevels noiseLevel);

    // For debug purposes
    CyclicCoordinateDescent& getCcd() { return *ccd; }
    AbstractModelData& getModelData() { return *modelData; }

    static void appendRList(Rcpp::List& list, const Rcpp::List& append);

    static ModelType parseModelType(const std::string& modelName);
    static priors::PriorType parsePriorType(const std::string& priorName);
    static ConvergenceType parseConvergenceType(const std::string& convergenceName);
    static NoiseLevels parseNoiseLevel(const std::string& noiseName);
  	static SelectorType parseSelectorType(const std::string& selectorName);
  	static NormalizationType parseNormalizationType(const std::string& normalizationName);
  	static void handleError(const std::string& str);

protected:

    priors::JointPriorPtr makePrior(
            const std::vector<std::string>& priorName,
            bsccs::priors::PriorFunctionPtr& priorFunctionPtr,
            const ProfileVector& flatPrior);

		priors::JointPriorPtr makePrior(
				const std::vector<std::string>& basePriorName,
				const std::vector<double>& baseVariance,
				const ProfileVector& flatPrior,
				const HierarchicalChildMap& map,
				const NeighborhoodMap& neighborhood);

    void initializeModelImpl(
            AbstractModelData** modelData,
            CyclicCoordinateDescent** ccd,
            AbstractModelSpecifics** model);

    void predictModelImpl(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData);

    void logModelImpl(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData,
            ProfileInformationMap& profileMap,
            bool withASE);

     void diagnoseModelImpl(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData,
    		double loadTime,
    		double updateTime);

private:

			AbstractModelData& rcppModelData; // TODO Make const?

	 		AbstractModelData* modelData;

// 	 		ModelData* modelData;

			CyclicCoordinateDescent* ccd;
			AbstractModelSpecifics* modelSpecifics;

			bsccs::ProfileInformationMap profileMap;
			Rcpp::List result;

}; // class RcppCcdInterface

} // namespace

#endif /* CCD_H_ */
