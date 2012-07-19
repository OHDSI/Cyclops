/*
 * ModelSpecifics.h
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef MODELSPECIFICS_H_
#define MODELSPECIFICS_H_

#include <cmath>

#include "AbstractModelSpecifics.h"

template <class BaseModel, typename WeightType>
class ModelSpecifics : public AbstractModelSpecifics, BaseModel {
public:
	ModelSpecifics();
	virtual ~ModelSpecifics();

	void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian,  bool useWeights);

protected:
	void computeNumeratorForGradient(int index);

	void updateXBeta(real realDelta, int index);

	void computeRemainingStatistics(void);

	void computeFixedTermsInGradientAndHessian(bool useCrossValidation);

	double getLogLikelihood(bool useCrossValidation);

	double getPredictiveLogLikelihood(real* weights);

	bool allocateXjY(void);

	bool allocateXjX(void);

	void setWeights(real* inWeights, bool useCrossValidation);

	void sortPid(bool useCrossValidation);

private:
	template <class IteratorType, class Weights>
	void computeGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian, Weights w);

	template <class IteratorType>
	void incrementNumeratorForGradientImpl(int index);

	template <class IteratorType>
	void updateXBetaImpl(real delta, int index);

	template <class OutType, class InType>
	void incrementByGroup(OutType* values, int* groups, int k, InType inc) {
		values[BaseModel::getGroup(groups, k)] += inc;
	}

	std::vector<WeightType> hNWeight;
	std::vector<WeightType> hKWeight;

	struct WeightedOperation {
		const static bool isWeighted = true;
	} unweighted;

	struct UnweightedOperation {
		const static bool isWeighted = false;
	} weighted;

};

struct GroupedData {
public:
	int getGroup(int* groups, int k) {
		return groups[k];
	}
	// TODO Should include grouping, i.e. nevents functions as well
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

struct GLMProjection {
public:
	const static bool precomputeGradient = true; // XjY

	const static bool likelihoodHasDenominator = true;

	const static bool hasTwoNumeratorTerms = true;

	real logLikeNumeratorContrib(int yi, real xBetai) {
		return yi * xBetai;
	}

	real gradientNumeratorContrib(real x, real predictor, real xBeta, real y) {
		return predictor * x;
	}

	real gradientNumerator2Contrib(real x, real predictor) {
		return predictor * x * x;
	}
};

template <typename WeightType>
struct SelfControlledCaseSeries : public GroupedData, GLMProjection {
public:
	const static bool precomputeHessian = false; // XjX

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	int observationCount(real yi) {
		return static_cast<int>(yi);
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
};

template <typename WeightType>
struct LogisticRegression : public IndependentData, GLMProjection {
public:
	const static bool precomputeHessian = false;

	static real getDenomNullValue () { return static_cast<real>(1.0); }

	int observationCount(real yi) {
		return static_cast<int>(1);
	}

	template <class IteratorType, class Weights>
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
};

template <typename WeightType>
struct CoxProportionalHazards : public OrderedData, GLMProjection {
public:
	const static bool precomputeHessian = false;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	int observationCount(real yi) {
		return static_cast<int>(1);
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
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}
};

template <typename WeightType>
struct LeastSquares : public IndependentData {
public:
	const static bool precomputeGradient = false; // XjY

	const static bool precomputeHessian = true; // XjX

	const static bool likelihoodHasDenominator = false;

	const static bool hasTwoNumeratorTerms = false;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	int observationCount(real yi) {
		return static_cast<int>(1);
	}

	real logLikeNumeratorContrib(int yi, real xBetai) {
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
};

#include "ModelSpecifics.hpp"

#endif /* MODELSPECIFICS_H_ */
