/*
 * ModelSpecifics.h
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef MODELSPECIFICS_H_
#define MODELSPECIFICS_H_

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "AbstractModelSpecifics.h"

namespace bsccs {

class SparseIterator; // forward declaration

template <class BaseModel, typename WeightType>
class ModelSpecifics : public AbstractModelSpecifics, BaseModel {
public:
	ModelSpecifics(const ModelData& input);

	virtual ~ModelSpecifics();

	void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian,  bool useWeights);
			
	void computeMMGradientAndHessian(int index, double *ogradient,
			double *ohessian,  double scale, bool useWeights);			

protected:
	void computeNumeratorForGradient(int index);

	void computeFisherInformation(int indexOne, int indexTwo, double *oinfo, bool useWeights);

	void updateXBeta(real realDelta, int index, bool useWeights);
	
	void computeXBeta(double* beta); 
	
	void computeRemainingStatistics(bool useWeights);

	void computeAccumlatedNumerDenom(bool useWeights);

	void computeFixedTermsInLogLikelihood(bool useCrossValidation);

	void computeFixedTermsInGradientAndHessian(bool useCrossValidation);

	double getLogLikelihood(bool useCrossValidation);

	double getPredictiveLogLikelihood(real* weights);

	void getPredictiveEstimates(real* y, real* weights);

	bool allocateXjY(void);

	bool allocateXjX(void);

	bool allocateNtoKIndices(void);

	bool sortPid(void);

	void setWeights(real* inWeights, bool useCrossValidation);

	void doSortPid(bool useCrossValidation);
	
	bool initializeAccumulationVectors(void);
	
	bool hasResetableAccumulators(void);

private:
	template <class IteratorType, class Weights>
	void computeGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian, Weights w);
			
	template <class IteratorType, class Weights>
	void computeMMGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian, double scale, Weights w);			

	template <class IteratorType>
	void incrementNumeratorForGradientImpl(int index);

	template <class IteratorType>
	void updateXBetaImpl(real delta, int index, bool useWeights);
	
	template <class IteratorType>
	void computeXBetaImpl(double *beta);

	template <class OutType, class InType>
	void incrementByGroup(OutType* values, int* groups, int k, InType inc) {
		values[BaseModel::getGroup(groups, k)] += inc; // TODO delegate to BaseModel (different in tied-models)
	}

	template <typename IteratorTypeOne, class Weights>
	void dispatchFisherInformation(int indexOne, int indexTwo, double *oinfo, Weights w);

	template <class IteratorTypeOne, class IteratorTypeTwo, class Weights>
	void computeFisherInformationImpl(int indexOne, int indexTwo, double *oinfo, Weights w);

	template<class IteratorType>
	SparseIterator getSubjectSpecificHessianIterator(int index);

	void computeXjY(bool useCrossValidation);

	void computeXjX(bool useCrossValidation);

	void computeNtoKIndices(bool useCrossValidation);
	
	void initializeMM(std::vector<bool>& fixBeta, std::vector<double>& ccdBeta, std::vector<double>& mmBeta);
	
	void copyBetaMM(std::vector<bool> ccdBeta);
	
	void computeNorms(void);
	
	template <class InteratorType>
	void incrementNormsImpl(int index);

	std::vector<WeightType> hNWeight;
	std::vector<WeightType> hKWeight;

	std::vector<int> nPid;
	std::vector<real> nY;
	std::vector<int> hNtoK;
	
	std::vector<double> *hBetaCCD;
	std::vector<double> *hBetaMM;

	
	std::vector<real> norm;

	struct WeightedOperation {
		const static bool isWeighted = true;
	} weighted;

	struct UnweightedOperation {
		const static bool isWeighted = false;
	} unweighted;

};

template <typename WeightType>
class CompareSurvivalTuples {
	bool useCrossValidation;
	std::vector<WeightType>& weight;
	const std::vector<real>& z;
public:
	CompareSurvivalTuples(bool _ucv,
			std::vector<WeightType>& _weight,
			const std::vector<real>& _z)
		: useCrossValidation(_ucv), weight(_weight), z(_z) {
		// Do nothing
	}
	bool operator()(size_t i, size_t j) { // return true if element[i] is before element[j]
		if (useCrossValidation) {
			if (weight[i] > weight[j]) { 			// weight[i] = 1, weight[j] = 0
				return true; // weighted first
			} else if (weight[i] < weight[j]) {		// weight[i] = 0, weight[j] = 1
				return false;
			}
		}
		return z[i] > z[j]; // TODO Throw error if non-unique y
	}
};

struct GroupedData {
public:
	const static bool hasStrataCrossTerms = true;

	const static bool hasNtoKIndices = true;
	
	const static bool exactTies = false;

	int getGroup(int* groups, int k) {
		return groups[k];
	}
};

struct GroupedWithTiesData : GroupedData {
public:	
	const static bool exactTies = true;
};

struct OrderedData {
public:
	const static bool hasStrataCrossTerms = true;
	
	const static bool hasNtoKIndices = false;	
	
	const static bool exactTies = false;	

	int getGroup(int* groups, int k) {
		return k; // No ties
	}
	
	const static bool hasResetableAccumulators = true;
};

struct OrderedWithTiesData {
public:
	const static bool hasStrataCrossTerms = true;
	
	const static bool hasNtoKIndices = false;	
	
	const static bool exactTies = false;

	int getGroup(int* groups, int k) {
		return groups[k];
	}	
	
	const static bool hasResetableAccumulators = false;	
};

struct IndependentData {
public:
	const static bool hasStrataCrossTerms = false;

	const static bool hasNtoKIndices = false;
	
	const static bool exactTies = false;	

	int getGroup(int* groups, int k) {
		return k;
	}
};

struct FixedPid {
	const static bool sortPid = false;

	const static bool cumulativeGradientAndHessian = false;		
	
	const static bool hasResetableAccumulators = false;
};

struct SortedPid {
	const static bool sortPid = true;

	const static bool cumulativeGradientAndHessian = true;	
};

struct NoFixedLikelihoodTerms {
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
// 		std::cerr << "Error!" << std::endl;
// 		exit(-1);
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}
};

struct GLMProjection {
public:
	const static bool precomputeGradient = true; // XjY

	const static bool likelihoodHasDenominator = true;

	const static bool hasTwoNumeratorTerms = true;

	real gradientNumeratorContrib(real x, real predictor, real xBeta, real y) {
		return predictor * x;
	}

	real logLikeNumeratorContrib(int yi, real xBetai) {
		return yi * xBetai;
	}

	real gradientNumerator2Contrib(real x, real predictor) {
		return predictor * x * x;
	}
	
	template <class IteratorType, class Weights>	
	inline void incrementMMGradientAndHessian(
			real& gradient, real& hessian,
			real expXBeta, real denominator, 
			real weight, real x, real xBeta, real y, real norm, real oldBeta, real newBeta) {

        throw new std::logic_error("Not model-specific");
	}
};

template <typename WeightType>
struct Survival {
public: /***/
	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		*information += weight * predictor / denom * it.value();
	}
};

template <typename WeightType>
struct Logistic {
public:
	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		const real g = predictor / denom;
		*information += weight *
				 (predictor / denom - g * g)
//
//				predictor / (denom * denom)
				 * it.value();
	}

	template <class IteratorType, class Weights> // TODO Code duplication with LR
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights w,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {

		const real g = numer / denom;
		if (Weights::isWeighted) {
			*gradient += weight * g;
		} else {
			*gradient += g;
		}
		if (IteratorType::isIndicator) {
			if (Weights::isWeighted) {
				*hessian += weight * g * (static_cast<real>(1.0) - g);
			} else {
				*hessian += g * (static_cast<real>(1.0) - g);
			}
		} else {
			if (Weights::isWeighted) {
				*hessian += weight * (numer2 / denom - g * g); // Bounded by x_j^2
			} else {
				*hessian += (numer2 / denom - g * g); // Bounded by x_j^2
			}
		}
	}
};

template <typename WeightType>
struct SelfControlledCaseSeries : public GroupedData, GLMProjection, FixedPid, Survival<WeightType> {
public:
	const static bool precomputeHessian = false; // XjX

#define TEST_CONSTANT_SCCS
#ifdef TEST_CONSTANT_SCCS
	const static bool likelihoodHasFixedTerms = true;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
		return yi * std::log(offseti);
	}
#else
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
// 		std::cerr << "Error!" << std::endl;
// 		exit(-1);
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}
#endif

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}
	
	template <class IteratorType, class Weights>
	inline void incrementMMGradientAndHessian(
			real& gradient, real& hessian,
			real expXBeta, real denominator, 
			real weight, real x, real xBeta, real y, real norm, real oldBeta, real newBeta) {
		
		if (IteratorType::isIndicator) {
			gradient += exp(norm*(newBeta - oldBeta)) *weight * expXBeta / denominator; //exp(norm*(newBeta - oldBeta)) *
			hessian += exp(norm*(newBeta - oldBeta)) *weight * expXBeta / denominator * norm; //exp(norm*(newBeta - oldBeta)) * 
		} else {
			gradient += weight * expXBeta * x / denominator;
			hessian += weight * expXBeta * x * x / denominator * norm;		
		}
	}	

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return offs[k] * std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		//do nothing for now
	}
	
	void copyBetaToMM(real& yi, real xBeta){
		//do nothing for now
	}

};

template <typename WeightType>
struct ConditionalPoissonRegression : public GroupedData, GLMProjection, FixedPid, Survival<WeightType> {
public:
	const static bool precomputeHessian = false; // XjX
	
	const static bool likelihoodHasFixedTerms = true;
		
// 	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
// 		return yi * std::log(offseti);
// 	}
	
	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
		real logLikeFixedTerm = 0.0;
		for(int i = 2; i <= (int)yi; i++)
			logLikeFixedTerm += -log((real)i);
		return logLikeFixedTerm;
	}	

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		//do nothing for now
	}

};

#if 0
template <typename WeightType>
struct ConditionalLogisticRegression : public GroupedData, GLMProjection, Logistic<WeightType>, FixedPid,
	NoFixedLikelihoodTerms { // TODO Implement likelihood terms
public:
	const static bool precomputeHessian = false; // XjX

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

//	real logLikeDenominatorContrib(WeightType ni, real denom) {
//		return std::log(denom);
//	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		 // TODO Can this be optimized for CLR?
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		// do nothing for now
	}

	template <class IteratorType, class Weights> // TODO Code duplication with LR
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights w,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {

		const real g = numer / denom;
//		if (Weights::isWeighted) {
			*gradient += weight * g;
//		} else {
//			*gradient += g;
//		}
		if (IteratorType::isIndicator) {
//			if (Weights::isWeighted) {
				*hessian += weight * g * (static_cast<real>(1.0) - g);
//			} else {
//				*hessian += g * (static_cast<real>(1.0) - g);
//			}
		} else {
//			if (Weights::isWeighted) {
				*hessian += weight * (numer2 / denom - g * g); // Bounded by x_j^2
//			} else {
//				*hessian += (numer2 / denom - g * g); // Bounded by x_j^2
//			}
		}
	}

	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		*information += weight * predictor / denom * it.value();
	}
};
#else
template <typename WeightType>
struct ConditionalLogisticRegression : public GroupedData, GLMProjection, FixedPid, Survival<WeightType> {
public:
	const static bool precomputeHessian = false; // XjX
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
// 		std::cerr << "Error!" << std::endl;
// 		exit(-1);
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
	    // Do nothing
		//yi = xBeta; // Returns the linear predictor;  ###relative risk		
	}

};
#endif

template <typename WeightType>
struct TiedConditionalLogisticRegression : public GroupedWithTiesData, GLMProjection, FixedPid, Survival<WeightType> {
public:
	const static bool precomputeGradient = true; // XjY   // TODO Until tied calculations are only used for ties
	const static bool precomputeHessian = false; // XjX
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
	    // Do nothing
		//yi = xBeta; // Returns the linear predictor;  ###relative risk		
	}

};

template <typename WeightType>
struct LogisticRegression : public IndependentData, GLMProjection, Logistic<WeightType>, FixedPid,
	NoFixedLikelihoodTerms {
public:
	const static bool precomputeHessian = false;

	static real getDenomNullValue () { return static_cast<real>(1.0); }

	real observationCount(real yi) {
		return static_cast<real>(1);
	}

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		real t = exp(xBeta);
		yi = t/(t+1);
	}
};

template <typename WeightType>
struct CoxProportionalHazards : public OrderedData, GLMProjection, SortedPid, NoFixedLikelihoodTerms, Survival<WeightType> {
public:
	const static bool precomputeHessian = false;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

    bool resetAccumulators(int* pid, int k, int currentPid) { return false; } // No stratification

	real observationCount(real yi) {
		return static_cast<real>(yi);  
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (not censured indicator)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real accDenom) {
		return ni*std::log(accDenom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)])); // TODO Wrong
	}

	void predictEstimate(real& yi, real xBeta){
		// do nothing for now
	}
};

template <typename WeightType>
struct StratifiedCoxProportionalHazards : public CoxProportionalHazards<WeightType> {
public:
    bool resetAccumulators(int* pid, int k, int currentPid) { 
        return pid[k] != currentPid;
    }
};

template <typename WeightType>
struct BreslowTiedCoxProportionalHazards : public OrderedWithTiesData, GLMProjection, SortedPid, NoFixedLikelihoodTerms, Survival<WeightType> {
public:
	const static bool precomputeHessian = false;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

    bool resetAccumulators(int* pid, int k, int currentPid) { 
        return pid[k] != currentPid;
    }
    
	real observationCount(real yi) {
		return static_cast<real>(yi);  
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (not censured indicator)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real accDenom) {
		return ni*std::log(accDenom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)])); // TODO Wrong
	}

	void predictEstimate(real& yi, real xBeta){
		// do nothing for now
	}
};

template <typename WeightType>
struct LeastSquares : public IndependentData, FixedPid, NoFixedLikelihoodTerms {
public:
	const static bool precomputeGradient = false; // XjY

	const static bool precomputeHessian = true; // XjX

	const static bool likelihoodHasDenominator = false;

	const static bool hasTwoNumeratorTerms = false;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(1);
	}

	real logLikeNumeratorContrib(real yi, real xBetai) {
		real residual = yi - xBetai;
		return - (residual * residual);
	}

	real gradientNumeratorContrib(real x, real predictor, real xBeta, real y) {
			return static_cast<real>(2) * x * (xBeta - y);
	}

	real gradientNumerator2Contrib(real x, real predictor) {
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}

	template <class IteratorType, class Weights>
	inline void incrementMMGradientAndHessian(
			real& gradient, real& hessian,
			real expXBeta, real denominator, 
			real weight, real x, real xBeta, real y, real norm, real oldBeta, real newBeta) {
												
		throw new std::logic_error("Not implemented.");			
	}

	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		*information += weight * it.value();
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			const Weights& w,
			real* gradient, real* hessian,
			real numer, real numer2, real denom, WeightType weight,
			real x, real xBeta, real y
			) {
		// Reduce contribution here
		if (Weights::isWeighted) {
			*gradient += weight * numer;
		} else {
			*gradient += numer;
		}
	}

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
// 		std::cerr << "Error!" << std::endl;
// 		exit(-1);
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}

	real logLikeDenominatorContrib(int ni, real denom) {
		return std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		real residual = ji - xBetai;
		return - (residual * residual * weighti);
	}

	void predictEstimate(real& yi, real xBeta){
		yi = xBeta;
	}
};

template <typename WeightType>
struct PoissonRegression : public IndependentData, GLMProjection, FixedPid {
public:

	const static bool precomputeHessian = false; // XjX

	const static bool likelihoodHasFixedTerms = true;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(1);
	}

	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		*information += weight * predictor * it.value();
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
		const IteratorType& it,
		const Weights& w,
		real* gradient, real* hessian,
		real numer, real numer2, real denom, WeightType weight,
		real x, real xBeta, real y
		) {
			// Reduce contribution here
			if (IteratorType::isIndicator) {
				if (Weights::isWeighted) {
					const real value = weight * numer;
					*gradient += value;
					*hessian += value;
				} else {
					*gradient += numer;
					*hessian += numer;
				}
#ifdef DEBUG_POISSON
				std::cerr << (*gradient) << std::endl;
#endif
			} else {
				if (Weights::isWeighted) {
					*gradient += weight * numer;
					*hessian += weight * numer2;
				} else {
					*gradient += numer;
					*hessian += numer2;
				}
#ifdef DEBUG_POISSON
				std::cerr << (*gradient) << std::endl;
#endif
			}
	}
	
	template <class IteratorType, class Weights>
	inline void incrementMMGradientAndHessian(
			real& gradient, real& hessian,
			real expXBeta, real denominator, 
			real weight, real x, real xBeta, real y, real norm, real oldBeta, real newBeta) {

		if (IteratorType::isIndicator) {
			if (Weights::isWeighted) {
				gradient += weight * expXBeta;
				hessian  += weight * expXBeta * norm;
			} else {
				gradient +=  expXBeta;
				hessian  +=  expXBeta * norm;			
			}
		} else {
			if (Weights::isWeighted) {
				gradient += weight * expXBeta * x;
				hessian  += weight * expXBeta * x * x * norm;		
			} else {
				gradient +=  expXBeta * x;
				hessian  +=  expXBeta * x * x * norm;				
			}
		}
	}	

	real getOffsExpXBeta(real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(int ni, real denom) {
		return denom;
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
		int* groups, int i) {
			return (ji*xBetai - exp(xBetai))*weighti;
	}

	void predictEstimate(real& yi, real xBeta){
		yi = exp(xBeta);
	}

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
		real logLikeFixedTerm = 0.0;
		for(int i = 2; i <= (int)yi; i++)
			logLikeFixedTerm += -log((real)i);
		return logLikeFixedTerm;
	}

};

} // namespace

#include "ModelSpecifics.hpp"

#endif /* MODELSPECIFICS_H_ */
