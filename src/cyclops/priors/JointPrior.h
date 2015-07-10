/*
 * JointPrior.h
 *
 *  Created on: Nov 11, 2013
 *      Author: msuchard
 */

#ifndef JOINTPRIOR_H_
#define JOINTPRIOR_H_

#include "Types.h"
#include "priors/CovariatePrior.h"

namespace bsccs {
namespace priors {

typedef std::vector<double> DoubleVector;

class JointPrior {
public:
	JointPrior() { }
	virtual ~JointPrior() { }

	virtual void setVariance(double x) = 0; // pure virtual

	virtual void setVariance(int level, double x) {}; // only works if of type HierarchicalPrior

	virtual std::vector<double> getVariance() const = 0; // pure virtual

	virtual double logDensity(const DoubleVector& beta) const = 0; // pure virtual

	virtual double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const = 0; // pure virtual

	virtual const std::string getDescription() const = 0; // pure virtual
	
	virtual bool getIsRegularized(const int index) const = 0; // pure virtual
	
	virtual bool getSupportsKktSwindle(const int index) const = 0; // pure virtual
	
	virtual bool getSupportsKktSwindle(void) const = 0; // pure virtual
	
	virtual double getKktBoundary(const int index) const = 0; // pure virtual
	
 	virtual JointPrior* clone() const = 0; // pure virtual
};

class MixtureJointPrior : public JointPrior {
public:
	typedef std::vector<PriorPtr> PriorList;

	MixtureJointPrior(PriorPtr defaultPrior, int length) : JointPrior(),
			listPriors(length, defaultPrior), uniquePriors(1, defaultPrior) {
		// Do nothing
	}

	void changePrior(PriorPtr newPrior, int index) {
		// TODO assert(index < listPriors.size());
		listPriors[index] = newPrior;
		uniquePriors.push_back(newPrior);
	}

	const std::string getDescription() const {
	    std::ostringstream stream;
	    for (PriorPtr prior : uniquePriors) {
	        stream << prior->getDescription() << " ";
	    }
		return stream.str();
	}

	void setVariance(double x) {
		for (PriorList::iterator it = uniquePriors.begin(); it != uniquePriors.end(); ++it) {
			(*it)->setVariance(x);			
		}
	}

	std::vector<double> getVariance() const {
	    std::vector<double> variances;
	    for (PriorPtr prior : uniquePriors) {
	        variances.push_back(prior->getVariance());
	    }
		return std::move(variances);
	}

	double logDensity(const DoubleVector& beta) const {

		// TODO assert(beta.size() == listPriors.size());
		double result = 0.0;
		for (size_t i = 0; i < beta.size(); ++i) {
			result += listPriors[i]->logDensity(beta[i]);
		}
		return result;
	}
	
	bool getIsRegularized(const int index) const {	    
	    return listPriors[index]->getIsRegularized();
	}

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const {
		return listPriors[index]->getDelta(gh, beta[index]);
	}
	
	bool getSupportsKktSwindle(const int index) const {
		return listPriors[index]->getSupportsKktSwindle();
	}
	
	double getKktBoundary(const int index) const {
		return listPriors[index]->getKktBoundary();
	}
	
	bool getSupportsKktSwindle(void) const {
		// Return true if *any* prior supports swindle
		for (auto&prior : uniquePriors) {
			if (prior->getSupportsKktSwindle()) {
				return true;
			}
		}
		return false;	
	}
	
	JointPrior* clone() const {
		PriorList newListPriors(listPriors.size());
																
		PriorList newUniquePriors;				
		for (auto& prior : uniquePriors) {
			PriorPtr newPriorPtr = PriorPtr(prior->clone());
			newUniquePriors.push_back(newPriorPtr);			
			for(size_t i = 0; i < listPriors.size(); ++i) {
				if (listPriors[i] == prior) {
					newListPriors[i] = newPriorPtr;
				}
			}			
		}
		
		// CovariatePrior* oldPtr = nullptr;
// 		for (auto& prior : listPriors) {
// 			PriorPtr newPriorPtr; 
// 			if (oldPtr == nullptr || oldPtr != &*prior) {
// 				oldPtr = &*prior;
// 				newPriorPtr = = PriorPtr(prior->clone());
// 				newUniquePriors.push_back(newPriorPtr);
// 			}
// 			newListPriors.push_back(newPriorPtr);
// 			std::cerr << "cloned " << &*prior* << " -> " << &*newPriorPtr << std::endl;
// 		}
				
		return new MixtureJointPrior(newListPriors, newUniquePriors);
	}

private:

	MixtureJointPrior(PriorList listPriors, PriorList uniquePriors) : 
		listPriors(listPriors), uniquePriors(uniquePriors) { }
		
	PriorList listPriors;
	PriorList uniquePriors;

};

class LaplaceFusedJointPrior : public HierarchicalJointPrior {

// TODO Implement

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
		std::stringstream info;
		for (int i = 0; i < hierarchyDepth; i ++) {
			info << "Hierarchy level " << i << " has prior " << hierarchyPriors[i]->getDescription() << " ";
		}
		return info.str();
	}

// 	void setHierarchy(HierarchyReader* hierarchyReader) {
// 		getParentMap = hierarchyReader->returnGetParentMap();
// 		getChildMap = hierarchyReader->returnGetChildMap();
// 	}

    void setHierarchy(HierarchicalParentMap parentMap,
             HierarchicalChildMap childMap) {
        // TODO getParentMap, getChildMap should be refs; no need to copy
        getParentMap = parentMap;
        getChildMap = childMap;
        
//         std::cout << "parentMap:" << std::endl;
//         std::for_each(begin(parentMap), end(parentMap), [](int i) {
//         	std::cout << " " << i;
//         });
//         std::cout << std::endl << "childMap:" << std::endl;
//         for(auto& v : childMap) {
//         	for(auto x : v) {
//         		std::cout << " " << x;
//         	}
//         	std::cout << std::endl;
//         }            
    }

	std::vector<double> getVariance() const{
		return std::vector<double> {
		    getVariance(0),
		    getVariance(1)
		};
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
	
	bool getIsRegularized(const int index) const {
	    return true;
	}
	
	bool getSupportsKktSwindle(const int index) const {
		return false; // TODO fix
	}	
	
	bool getSupportsKktSwindle(void) const {		
		return false; // TODO fix	
	}	
	
	double getKktBoundary(const int index) const {
		return 0.0; // TODO fix
	}	

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const {
		double t1 = 1/hierarchyPriors[0]->getVariance(); // this is the hyperparameter that is used in the original code
		double t2 = 1/hierarchyPriors[1]->getVariance();

		int parent = getParentMap.at(index);
		const std::vector<int>& siblings = getChildMap.at(parent);
		double sumBetas = 0;
//		int nSiblingsOfInterest = 0; //Different from siblings length if weights used
		for (size_t i = 0; i < siblings.size(); i++) {
			sumBetas += beta[siblings[i]];
		}
		double hessian = t1 - t1 / (siblings.size() + t2/t1);

		double gradient = t1*beta[index] - t1*t1*sumBetas / (siblings.size()*t1 + t2);

		return (- (gh.first + gradient)/(gh.second + hessian));
	}
	
	JointPrior* clone() const {
		PriorList newHierarchyPriors;	
		
		for (auto& prior : hierarchyPriors) {
			newHierarchyPriors.push_back(PriorPtr(prior->clone()));
		}	
	
		return new HierarchicalJointPrior(newHierarchyPriors, hierarchyDepth, getParentMap, 
			getChildMap);
	}	

private:

	HierarchicalJointPrior(PriorList hierarchyPriors, int hierarchyDepth, 
		HierarchicalParentMap getParentMap,
		HierarchicalChildMap getChildMap) : hierarchyPriors(hierarchyPriors),
		hierarchyDepth(hierarchyDepth), getParentMap(getParentMap), getChildMap(getChildMap) { }

	PriorList hierarchyPriors;
	int hierarchyDepth;
	HierarchicalParentMap getParentMap;
	HierarchicalChildMap getChildMap;

};

class FullyExchangeableJointPrior : public JointPrior {
public:
	FullyExchangeableJointPrior(PriorPtr _one) : JointPrior(), singlePrior(_one) {
		// Do nothing
	}

	void setVariance(double x) {
		singlePrior->setVariance(x);
	}

	std::vector<double> getVariance() const {
		return std::vector<double> {
		    singlePrior->getVariance()
		};
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
	
	bool getIsRegularized(const int index) const {
	    return singlePrior->getIsRegularized();
	}

	const std::string getDescription() const {
		return singlePrior->getDescription();
	}
	
	bool getSupportsKktSwindle(const int index) const {
		return singlePrior->getSupportsKktSwindle();
	}
	
	bool getSupportsKktSwindle(void) const {
		return singlePrior->getSupportsKktSwindle();
	}	
	
	double getKktBoundary(const int index) const {
		return singlePrior->getKktBoundary();
	}	
		
	JointPrior* clone() const {
		return new FullyExchangeableJointPrior(PriorPtr(singlePrior->clone()));
	}		

private:
	PriorPtr singlePrior;
};

typedef bsccs::shared_ptr<JointPrior> JointPriorPtr;

} /* namespace priors */
} /* namespace bsccs */
#endif /* JOINTPRIOR_H_ */
