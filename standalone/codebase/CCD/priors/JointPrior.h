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
#include "io/HierarchyReader.h"

namespace bsccs {
namespace priors {

typedef std::vector<double> DoubleVector;

class JointPrior {
public:
	JointPrior() { }
	virtual ~JointPrior() { }

	virtual void setVariance(double x) = 0; // pure virtual

	virtual void setVariance(int level, double x) {}; // only works if of type HierarchicalPrior

	virtual double getVariance() const = 0; // pure virtual

	virtual double logDensity(const DoubleVector& beta) const = 0; // pure virtual

	virtual double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const = 0; // pure virtual

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

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const {
		return listPriors[index]->getDelta(gh, beta[index]);
	}

private:
	PriorList listPriors;

};

class HierarchicalJointPrior : public JointPrior {
public:
	typedef std::vector<PriorPtr> PriorList;

	HierarchicalJointPrior(PriorPtr defaultPrior, int length) : JointPrior(),
			hierarchyPriors(length, defaultPrior) {
		hierarchyDepth = length;
	}

	void setVariance(double x) {
		setVariance(0,x);
	}

	void setVariance(int level, double x) {
		hierarchyPriors[level]->setVariance(x);
	}

	void changePrior(PriorPtr newPrior, int index) {
		// TODO assert(index < listPriors.size());
		hierarchyPriors[index] = newPrior;
	}

	const std::string getDescription() const {
		stringstream info;
		for (int i = 0; i < hierarchyDepth; i ++) {
			info << "Hierarchy level " << i << " has prior " << hierarchyPriors[i]->getDescription() << " ";
		}
		return info.str();
	}

	void setHierarchy(HierarchyReader* hierarchyReader) {
		getParentMap = hierarchyReader->returnGetParentMap();
		getChildMap = hierarchyReader->returnGetChildMap();
	}

	double getVariance() const{
		return getVariance(0);
	}

	double getVariance(int level) const {
		return hierarchyPriors[level]->getVariance();
	}

	double logDensity(const DoubleVector& beta) const {
		double result = 0.0;
		for (DoubleVector::const_iterator it = beta.begin(); it != beta.end(); ++it) {
			result += hierarchyPriors[0]->logDensity(*it);
		}
		return result;
	}

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const {
		double t1 = 1/hierarchyPriors[0]->getVariance(); // this is the hyperparameter that is used in the original code
		double t2 = 1/hierarchyPriors[1]->getVariance();

		int parent = getParentMap.at(index);
		const vector<int>& siblings = getChildMap.at(parent);
		double sumBetas = 0;
		int nSiblingsOfInterest = 0; //Different from siblings length if weights used
		for (int i = 0; i < siblings.size(); i++) {
			sumBetas += beta[siblings[i]];
		}
		double hessian = t1 - t1 / (siblings.size() + t2/t1);

		double gradient = t1*beta[index] - t1*t1*sumBetas / (siblings.size()*t1 + t2);

		return (- (gh.first + gradient)/(gh.second + hessian));
	}

private:
	PriorList hierarchyPriors;
	int hierarchyDepth;
	std::map<int, int> getParentMap;
	std::map<int, vector<int> > getChildMap;

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

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const {
		return singlePrior->getDelta(gh, beta[index]);
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
