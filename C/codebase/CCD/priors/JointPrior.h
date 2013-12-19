/*
 * JointPrior.h
 *
 *  Created on: Nov 11, 2013
 *      Author: msuchard
 */

#ifndef JOINTPRIOR_H_
#define JOINTPRIOR_H_

#include "CyclicCoordinateDescent.h"
#include "priors/CovariatePrior.h"

namespace bsccs {
namespace priors {

typedef std::vector<double> DoubleVector;

class JointPrior {
public:
	JointPrior() { }
	virtual ~JointPrior() { }

	virtual void setVariance(double x) = 0; // pure virtual

	virtual double getVariance() const = 0; // pure virtual

	virtual double logDensity(const DoubleVector& beta) const = 0; // pure virtual

	virtual double getDelta(const GradientHessian gh, const double beta, const int index) const = 0; // pure virtual

	virtual const std::string getDescription() const = 0; // pure virtual
};

class MixtureJointPrior : public JointPrior {
public:
	typedef std::vector<PriorPtr> PriorList;

	MixtureJointPrior(PriorPtr defaultPrior, int length) : JointPrior(),
			listPriors(length, defaultPrior) {
		// Do nothing
	}

	void changePrior(PriorPtr newPrior, int index) {
		// TODO assert(index < listPriors.size());
		listPriors[index] = newPrior;
	}

	const std::string getDescription() const {
		return "Mixture";
	}

	void setVariance(double x) {
		for (PriorList::iterator it = listPriors.begin(); it != listPriors.end(); ++it) {
			(*it)->setVariance(x);
			// TODO Should only update unique pointers
		}
	}

	double getVariance() const {
		return 0.0;
	}

	double logDensity(const DoubleVector& beta) const {

		// TODO assert(beta.size() == listPriors.size());
		double result = 0.0;
		for (int i = 0; i < beta.size(); ++i) {
			result += listPriors[i]->logDensity(beta[i]);
		}
		return result;
	}

	double getDelta(const GradientHessian gh, const double beta, const int index) const {
		return listPriors[index]->getDelta(gh, beta);
	}

private:
	PriorList listPriors;

};

class FullyExchangeableJointPrior : public JointPrior {
public:
	FullyExchangeableJointPrior(PriorPtr _one) : JointPrior(), singlePrior(_one) {
		// Do nothing
	}

	void setVariance(double x) {
		singlePrior->setVariance(x);
	}

	double getVariance() const {
		return singlePrior->getVariance();
	}

	double logDensity(const DoubleVector& beta) const {
		double result = 0.0;
		for (DoubleVector::const_iterator it = beta.begin(); it != beta.end(); ++it) {
			result += singlePrior->logDensity(*it);
		}
		return result;
	}

	double getDelta(const GradientHessian gh, const double beta, const int index) const {
		return singlePrior->getDelta(gh, beta);
	}

	const std::string getDescription() const {
		return singlePrior->getDescription();
	}

private:
	PriorPtr singlePrior;
};

typedef std::shared_ptr<JointPrior> JointPriorPtr;

} /* namespace priors */
} /* namespace bsccs */
#endif /* JOINTPRIOR_H_ */
