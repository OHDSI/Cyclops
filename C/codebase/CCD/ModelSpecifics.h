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

#include "AbstractModelSpecifics.h"

namespace bsccs {

template <class BaseModel, typename WeightType>
class ModelSpecifics : public AbstractModelSpecifics, BaseModel {
public:
	ModelSpecifics(const ModelData& input);

	virtual ~ModelSpecifics();

	void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian,  bool useWeights);

protected:
	void computeNumeratorForGradient(int index);

	void updateXBeta(real realDelta, int index, bool useWeights);

	void computeRemainingStatistics(bool useWeights);

	void computeAccumlatedNumerDenom(bool useWeights);

	void computeFixedTermsInLogLikelihood(bool useCrossValidation);

	void computeFixedTermsInGradientAndHessian(bool useCrossValidation);

	double getLogLikelihood(bool useCrossValidation);

	double getPredictiveLogLikelihood(real* weights);

	void getPredictiveEstimates(real* y, real* weights);

	bool allocateXjY(void);

	bool allocateXjX(void);

	bool sortPid(void);

	void setWeights(real* inWeights, bool useCrossValidation);

	void doSortPid(bool useCrossValidation);

private:
	template <class IteratorType, class Weights>
	void computeGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian, Weights w);

	template <class IteratorType>
	void incrementNumeratorForGradientImpl(int index);

	template <class IteratorType>
	void updateXBetaImpl(real delta, int index, bool useWeights);

	template <class OutType, class InType>
	void incrementByGroup(OutType* values, int* groups, int k, InType inc) {
		values[BaseModel::getGroup(groups, k)] += inc;
	}

	void computeXjY(bool useCrossValidation);

	void computeXjX(bool useCrossValidation);

	std::vector<WeightType> hNWeight;
	std::vector<WeightType> hKWeight;

	std::vector<int> nPid;
	std::vector<real> nY;

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
	bool operator()(size_t i, size_t j) { // return turn if element[i] is before element[j]
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
	int getGroup(int* groups, int k) {
		return groups[k];
	}
};

struct OrderedData {
public:
	int getGroup(int* groups, int k) {
		return groups[k];
	}
};

struct IndependentData {
public:
	int getGroup(int* groups, int k) {
		return k;
	}
};

struct FixedPid {
	const static bool sortPid = false;

	const static bool cumulativeGradientAndHessian = false;
};

struct SortedPid {
	const static bool sortPid = true;

	const static bool cumulativeGradientAndHessian = true;
};

struct NoFixedLikelihoodTerms {
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti){
		std::cerr << "Error!" << std::endl;
		exit(-1);
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
};

template <typename WeightType>
struct Logistic {
public:
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
struct SelfControlledCaseSeries : public GroupedData, GLMProjection, FixedPid {
public:
	const static bool precomputeHessian = false; // XjX

#define TEST_CONSTANT_SCCS
#ifdef TEST_CONSTANT_SCCS
	const static bool likelihoodHasFixedTerms = true;

	real logLikeFixedTermsContrib(real yi, real offseti){
		return yi * std::log(offseti);
	}
#else
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti){
		std::cerr << "Error!" << std::endl;
		exit(-1);
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

	real getOffsExpXBeta(int* offs, real xBeta, real y, int k) {
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

};

template <typename WeightType>
struct ConditionalLogisticRegression : public GroupedData, GLMProjection, Logistic<WeightType>, FixedPid,
	NoFixedLikelihoodTerms { // TODO Implement likelihood terms
public:
	const static bool precomputeHessian = false; // XjX

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	real getOffsExpXBeta(int* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		 // TODO Can this be optimized for CLR?
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		// do nothing for now
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

	real getOffsExpXBeta(int* offs, real xBeta, real y, int k) {
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
struct CoxProportionalHazards : public OrderedData, GLMProjection, SortedPid, NoFixedLikelihoodTerms {
public:
	const static bool precomputeHessian = false;

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
		const real g = nEvents * t; // Always use weights (not censured indicator)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(int* offs, real xBeta, real y, int k) {
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
		std::cerr << "Error!" << std::endl;
		exit(-1);
		return static_cast<real>(0);
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

	real getOffsExpXBeta(int* offs, real xBeta, real y, int k) {
		std::cerr << "Error!" << std::endl;
		exit(-1);
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

	real getOffsExpXBeta(int* offs, real xBeta, real y, int k) {
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

	real logLikeFixedTermsContrib(real yi, real offseti){
		real logLikeFixedTerm = 0.0;
		for(int i = 2; i <= (int)yi; i++)
			logLikeFixedTerm += -log((real)i);
		return logLikeFixedTerm;
	}

};

} // namespace

#include "ModelSpecifics.hpp"

#endif /* MODELSPECIFICS_H_ */
