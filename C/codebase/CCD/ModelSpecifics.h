/*
 * ModelSpecifics.h
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef MODELSPECIFICS_H_
#define MODELSPECIFICS_H_

#include "AbstractModelSpecifics.h"

template <class Model>
class ModelSpecifics : public AbstractModelSpecifics, Model {
public:
	ModelSpecifics();
	virtual ~ModelSpecifics();

	virtual void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian);

	virtual void computeNumeratorForGradient(int index);

	virtual void updateXBeta(real realDelta, int index);

	virtual void computeRemainingStatistics(void);

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
};

struct DefaultModel {
	const static real denomNullValue = 0.0;
};

struct LogisticRegression {
	const static real denomNullValue = 1.0;
};

#include "ModelSpecifics.hpp"

#endif /* MODELSPECIFICS_H_ */
