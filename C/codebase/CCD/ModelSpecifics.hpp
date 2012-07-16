/*
 * ModelSpecifics.hpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef MODELSPECIFICS_HPP_
#define MODELSPECIFICS_HPP_

#include <cmath>
#include <cstdlib>

#include "ModelSpecifics.h"
#include "Iterators.h"

template <class BaseModel>
ModelSpecifics<BaseModel>::ModelSpecifics() : AbstractModelSpecifics(), BaseModel() {
	// TODO Memory allocation here
}

template <class BaseModel>
ModelSpecifics<BaseModel>::~ModelSpecifics() {
	// TODO Memory release here
}

template <class BaseModel>
bool ModelSpecifics<BaseModel>::allocateXjY(void) { return BaseModel::precomputeGradient; }

template <class BaseModel>
bool ModelSpecifics<BaseModel>::allocateXjX(void) { return BaseModel::precomputeHessian; }

template <class BaseModel>
void ModelSpecifics<BaseModel>::computeFixedTermsInGradientAndHessian(bool useCrossValidation) {

	if (allocateXjY()) {
		for (int j = 0; j < J; ++j) {
			hXjY[j] = 0;
			GenericIterator it(*hXI, j);

			if (useCrossValidation) {
				for (; it; ++it) {
					const int k = it.index();
					hXjY[j] += it.value() * hY[k] * hWeights[k];
				}
			} else {
				for (; it; ++it) {
					const int k = it.index();
					hXjY[j] += it.value() * hY[k];
				}
			}
		}
	}

	if (allocateXjX()) {
		for (int j = 0; j < J; ++j) {
			hXjX[j] = 0;
			GenericIterator it(*hXI, j);

			if (useCrossValidation) {
				for (; it; ++it) {
					const int k = it.index();
					hXjX[j] += it.value() * it.value() * hWeights[k];
				}
			} else {
				for (; it; ++it) {
					const int k = it.index();
					hXjX[j] += it.value() * it.value();
				}
			}
		}
	}
}

template <class BaseModel>
double ModelSpecifics<BaseModel>::getLogLikelihood(bool useCrossValidation) {

	real logLikelihood = static_cast<real>(0.0);
	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			logLikelihood += BaseModel::logLikeNumeratorContrib(hY[i], hXBeta[i]) * hWeights[i];
		}
	} else {
		for (int i = 0; i < K; i++) {
			logLikelihood += BaseModel::logLikeNumeratorContrib(hY[i], hXBeta[i]);
		}
	}

	for (int i = 0; i < N; i++) {
		// Weights modified in computeNEvents()
		logLikelihood -= BaseModel::logLikeDenominatorContrib(hNEvents[i], denomPid[i]);
	}

	return static_cast<double>(logLikelihood);
}

template <class BaseModel>
double ModelSpecifics<BaseModel>::getPredictiveLogLikelihood(real* weights) {
	real logLikelihood = static_cast<real>(0.0);

	for (int k = 0; k < K; ++k) {
		logLikelihood += BaseModel::logPredLikeContrib(hY[k], weights[k], hXBeta[k], denomPid, hPid, k);
	}

	return static_cast<double>(logLikelihood);
}

// TODO The following function is an example of a double-dispatch, rewrite without need for virtual function
template <class BaseModel>
void ModelSpecifics<BaseModel>::computeGradientAndHessian(int index, double *ogradient,
		double *ohessian) {
	// Run-time dispatch, so virtual call should not effect speed, TODO Check
	switch (hXI->getFormatType(index)) {
		case INDICATOR :
			computeGradientAndHessianImpl<IndicatorIterator>(index, ogradient, ohessian);
			break;
		case SPARSE :
			computeGradientAndHessianImpl<SparseIterator>(index, ogradient, ohessian);
			break;
		case DENSE :
			computeGradientAndHessianImpl<DenseIterator>(index, ogradient, ohessian);
			break;
	}
}

//incrementGradientAndHessian<SparseIterator>();

template <class BaseModel> template <class IteratorType>
void ModelSpecifics<BaseModel>::computeGradientAndHessianImpl(int index, double *ogradient,
		double *ohessian) {
	real gradient = 0;
	real hessian = 0;

	IteratorType it(*(*sparseIndices)[index], N); // TODO How to create with different constructor signatures?
	for (; it; ++it) {
		const int k = it.index();
		// Compile-time delegation
		BaseModel::incrementGradientAndHessian(it,
						&gradient, &hessian,
						numerPid[k], numerPid2[k], denomPid[k], hNEvents[k]
				);
	}

	if (BaseModel::precomputeGradient) { // Compile-time switch
		gradient -= hXjY[index];
	}

	if (BaseModel::precomputeHessian) { // Compile-time switch
		hessian += hXjX[index];
	}

	*ogradient = static_cast<double>(gradient);
	*ohessian = static_cast<double>(hessian);
}

template <class BaseModel>
void ModelSpecifics<BaseModel>::computeNumeratorForGradient(int index) {
	// Run-time delegation
	switch (hXI->getFormatType(index)) {
		case INDICATOR : {
			IndicatorIterator it(*(*sparseIndices)[index]);
			for (; it; ++it) { // Only affected entries
				numerPid[it.index()] = static_cast<real>(0.0);
			}
			incrementNumeratorForGradientImpl<IndicatorIterator>(index);
			}
			break;
		case DENSE :
			zeroVector(numerPid, N);
			zeroVector(numerPid2, N);
			incrementNumeratorForGradientImpl<DenseIterator>(index);
			break;
		case SPARSE : {
			IndicatorIterator it(*(*sparseIndices)[index]);
			for (; it; ++it) { // Only affected entries
				numerPid[it.index()] = static_cast<real>(0.0);
				numerPid2[it.index()] = static_cast<real>(0.0); // TODO Does this invalid the cache line too much?
			}
			incrementNumeratorForGradientImpl<SparseIterator>(index); }
			break;
		default :
			// throw error
			exit(-1);
	}
}

template <class BaseModel> template <class IteratorType>
void ModelSpecifics<BaseModel>::incrementNumeratorForGradientImpl(int index) {
	IteratorType it(*hXI, index);
	for (; it; ++it) {
		const int k = it.index();
		incrementByGroup(numerPid, hPid, k, offsExpXBeta[k] * it.value());
		if (!IteratorType::isIndicator) {
			incrementByGroup(numerPid2, hPid, k, offsExpXBeta[k] * it.value() * it.value());
		}
	}
}

template <class BaseModel>
void ModelSpecifics<BaseModel>::updateXBeta(real realDelta, int index) {
	// Run-time dispatch to implementation depending on covariate FormatType
	switch(hXI->getFormatType(index)) {
		case INDICATOR :
			updateXBetaImpl<IndicatorIterator>(realDelta, index);
			break;
		case DENSE :
			updateXBetaImpl<DenseIterator>(realDelta, index);
			break;
		case SPARSE :
			updateXBetaImpl<SparseIterator>(realDelta, index);
			break;
		default :
			// throw error
			exit(-1);
	}
}

template <class BaseModel> template <class IteratorType>
inline void ModelSpecifics<BaseModel>::updateXBetaImpl(real realDelta, int index) {
	IteratorType it(*hXI, index);
	for (; it; ++it) {
		const int k = it.index();
		hXBeta[k] += realDelta * it.value();
		// Update denominators as well
		real oldEntry = offsExpXBeta[k];
		real newEntry = offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs, hXBeta[k], k);
		incrementByGroup(denomPid, hPid, k, (newEntry - oldEntry));
	}
}

template <class BaseModel>
void ModelSpecifics<BaseModel>::computeRemainingStatistics(void) {
	fillVector(denomPid, N, BaseModel::denomNullValue);
	for (int k = 0; k < K; ++k) {
		offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs, hXBeta[k], k);
		incrementByGroup(denomPid, hPid, k, offsExpXBeta[k]);
	}
}

#endif /* MODELSPECIFICS_HPP_ */
