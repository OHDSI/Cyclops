/*
 * ModelSpecifics.hpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef MODELSPECIFICS_HPP_
#define MODELSPECIFICS_HPP_

#include <cmath>

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

template <class BaseModel> template <class IteratorType>
void ModelSpecifics<BaseModel>::computeGradientAndHessianImpl(int index, double *ogradient,
		double *ohessian) {
	real gradient = 0;
	real hessian = 0;

	IteratorType it(*(*sparseIndices)[index], N); // TODO How to create with different constructor signatures?
	for (; it; ++it) {
		const int k = it.index();
		incrementGradientAndHessian<IteratorType>(
				&gradient, &hessian,
				numerPid[k], numerPid2[k], denomPid[k], hNEvents[k]
		);
	}
	gradient -= hXjEta[index];

	*ogradient = static_cast<double>(gradient);
	*ohessian = static_cast<double>(hessian);
}

template <class BaseModel> template <class IteratorType>
inline void ModelSpecifics<BaseModel>::incrementGradientAndHessian(
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
		numerPid[hPid[k]] += offsExpXBeta[k] * it.value();
		if (!IteratorType::isIndicator) {
			numerPid2[hPid[k]] += offsExpXBeta[k] * it.value() * it.value();
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
void ModelSpecifics<BaseModel>::updateXBetaImpl(real realDelta, int index) {
	IteratorType it(*hXI, index);
	for (; it; ++it) {
		const int k = it.index();
		hXBeta[k] += realDelta * it.value();
		// Update denominators as well
		real oldEntry = offsExpXBeta[k];
		real newEntry = offsExpXBeta[k] = hOffs[k] * exp(hXBeta[k]);
		denomPid[hPid[k]] += (newEntry - oldEntry);
	}
}

template <class BaseModel>
void ModelSpecifics<BaseModel>::computeRemainingStatistics(void) {
	fillVector(denomPid, N, BaseModel::denomNullValue);
	for (int i = 0; i < K; i++) {
		offsExpXBeta[i] = hOffs[i] * exp(hXBeta[i]);
		denomPid[hPid[i]] += offsExpXBeta[i];
	}
}

#endif /* MODELSPECIFICS_HPP_ */