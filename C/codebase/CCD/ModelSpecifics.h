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

	void computeNumeratorForGradient(int index);

	void updateXBeta(real realDelta, int index);

	void computeRemainingStatistics(void);

	double getLogLikelihood(bool useCrossValidation);

	double getPredictiveLogLikelihood(real* weights);

private:
	template <class IteratorType>
	void computeGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian);

	template <class IteratorType>
	void incrementGradientAndHessian(
			real* gradient, real* hessian,
			real numer, real numer2, real denom, int nEvents);

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

struct LinearProjection {
public:
	real logLikeNumeratorContrib(int yi, real xBetai) {
		return yi * xBetai;
	}
};

struct SelfControlledCaseSeries : public GroupedData, LinearProjection {
public:
	const static real denomNullValue = 0.0;

	template <class IteratorType>
	void incrementGradientAndHessian(
			const IteratorType& it,
			real* gradient, real* hessian,
			real numer, real numer2, real denom, int nEvents) {

		const real t = numer / denom;
		const real g = nEvents * t;
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(int* offs, real xBeta, int k) {
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

struct LogisticRegression : public IndependentData, LinearProjection {
public:
	const static real denomNullValue = 1.0;

	template <class IteratorType>
	void incrementGradientAndHessian(
			const IteratorType& it,
			real* gradient, real* hessian,
			real numer, real numer2, real denom, int nEvents) {

		const real g = numer / denom;
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - g);
		} else {
			*hessian += (numer2 / denom - g * g); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(int* offs, real xBeta, int k) {
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

struct LeastSquares : public IndependentData, LinearProjection {
public:
	const static real denomNullValue = 0.0; // TODO No need to compute denominators

	template <class IteratorType>
	void incrementGradientAndHessian(
			const IteratorType& it,
			real* gradient, real* hessian,
			real numer, real numer2, real denom, int nEvents) {

		const real g = numer / denom;
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - g);
		} else {
			*hessian += (numer2 / denom - g * g); // Bounded by x_j^2
		}
	}

	real getOffsExpXBeta(int* offs, real xBeta, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(int ni, real denom) {
		return std::log(denom);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, real* denoms,
			int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[i]));
	}
};

#include "ModelSpecifics.hpp"

#endif /* MODELSPECIFICS_H_ */
