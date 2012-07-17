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

template <class BaseModel>
class ModelSpecifics : public AbstractModelSpecifics, BaseModel {
public:
	ModelSpecifics();
	virtual ~ModelSpecifics();

	void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian);

protected:
	void computeNumeratorForGradient(int index);

	void updateXBeta(real realDelta, int index);

	void computeRemainingStatistics(void);

	void computeFixedTermsInGradientAndHessian(bool useCrossValidation);

	double getLogLikelihood(bool useCrossValidation);

	double getPredictiveLogLikelihood(real* weights);

	bool allocateXjY(void);

	bool allocateXjX(void);

private:
	template <class IteratorType>
	void computeGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian);

	template <class IteratorType>
	void incrementNumeratorForGradientImpl(int index);

	template <class IteratorType>
	void updateXBetaImpl(real delta, int index);

	void incrementByGroup(real* values, int* groups, int k, real inc) {
		values[BaseModel::getGroup(groups, k)] += inc;
	}
};

struct GroupedData {
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


struct SelfControlledCaseSeries : public GroupedData, GLMProjection {
public:
	const static real denomNullValue = 0.0;

	const static bool precomputeHessian = false; // XjX

	template <class IteratorType>
	void incrementGradientAndHessian(
			const IteratorType& it,
			real* gradient, real* hessian,
			real numer, real numer2, real denom, int nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t;
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

	real logLikeDenominatorContrib(int ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}
};

struct LogisticRegression : public IndependentData, GLMProjection {
public:
	const static real denomNullValue = 1.0;

	const static bool precomputeHessian = false;

	template <class IteratorType>
	void incrementGradientAndHessian(
			const IteratorType& it,
			real* gradient, real* hessian,
			real numer, real numer2, real denom, int nEvents,
			real x, real xBeta, real y) {

		const real g = numer / denom;
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - g);
		} else {
			*hessian += (numer2 / denom - g * g); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(int* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(int ni, real denom) {
		return std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}
};

struct LeastSquares : public IndependentData {
public:
	const static real denomNullValue = 0.0;

	const static bool precomputeGradient = false; // XjY

	const static bool precomputeHessian = true; // XjX

	const static bool likelihoodHasDenominator = false;

	const static bool hasTwoNumeratorTerms = false;

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

	template <class IteratorType>
	void incrementGradientAndHessian(
			const IteratorType& it,
			real* gradient, real* hessian,
			real numer, real numer2, real denom, int nEvents,
			real x, real xBeta, real y
			) {
		// Compute contribution here, numerators and denominators are unnecessary
		*gradient += numer;
//		*gradient += static_cast<real>(2) * x * (xBeta - y);
	}

	real getOffsExpXBeta(int* offs, real xBeta, real y, int k) {
		return static_cast<real>(2) * (xBeta - y);
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
