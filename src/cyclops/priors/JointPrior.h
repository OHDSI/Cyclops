/*
 * PriorFunction.h
 *
 *  Created on: Nov 11, 2013
 *      Author: msuchard
 */

#ifndef JOINTPRIOR_H_
#define JOINTPRIOR_H_

#include <algorithm>

#include "Types.h"
#include "priors/CovariatePrior.h"

namespace bsccs {
namespace priors {

typedef std::vector<double> DoubleVector;

class JointPrior {
public:
	JointPrior() { }
	virtual ~JointPrior() { }

// 	virtual void setVariance(double x) = 0; // pure virtual

	// virtual void setVariance(int level, double x) = 0; // pure virtual

	// virtual std::vector<double> getVariance() const = 0; // pure virtual

	virtual double logDensity(const DoubleVector& beta) const = 0; // pure virtual

	virtual double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const = 0; // pure virtual

	virtual const std::string getDescription() const = 0; // pure virtual

	virtual bool getIsRegularized(const int index) const = 0; // pure virtual

	virtual bool getSupportsKktSwindle(const int index) const = 0; // pure virtual

	virtual bool getSupportsKktSwindle(void) const = 0; // pure virtual

	virtual double getKktBoundary(const int index) const = 0; // pure virtual

//  	virtual JointPrior* clone() const = 0; // pure virtual

    void addVarianceParameter(const VariancePtr& ptr) {
        if (std::find(variance.begin(), variance.end(), ptr) == variance.end()) {
            variance.push_back(ptr);
        }
    }

	void addVarianceParameters(const std::vector<VariancePtr>& ptrs) {
	    for (auto ptr : ptrs) {
		    addVarianceParameter(ptr);
	    }
	}

	void setVariance(int index, double x) {
		// TODO Check bounds
		variance[index].set(x);
	}

	std::vector<double> getVariance() const {
	    std::vector<double> tmp;
	    for (auto v : variance) {
	        tmp.push_back(v.get());
	    }
		return tmp;
	}

protected:

    std::vector<VariancePtr> variance;
	// std::vector<double> variance;
	// std::vector<std::vector<int>> varianceMap;

};

class MixtureJointPrior : public JointPrior {
public:
	typedef std::vector<PriorPtr> PriorList;

	MixtureJointPrior(PriorPtr defaultPrior, int length) : JointPrior(),
			listPriors(length, defaultPrior), uniquePriors(1, defaultPrior) {
		addVarianceParameters(defaultPrior->getVarianceParameters());
	}

	void changePrior(PriorPtr newPrior, int index) {
		// TODO assert(index < listPriors.size());
		listPriors[index] = newPrior;
		uniquePriors.push_back(newPrior);
		addVarianceParameters(newPrior->getVarianceParameters());
	}

	const std::string getDescription() const {
	    std::ostringstream stream;
	    for (PriorPtr prior : uniquePriors) {
	        stream << prior->getDescription() << " ";
	    }
		return stream.str();
	}

// 	void setVariance(int index, double x) {
// // 		for (PriorList::iterator it = uniquePriors.begin(); it != uniquePriors.end(); ++it) {
// // 			(*it)->setVariance(level, x);
// // 		}
//         *variance[index] = x;
// 	}

// 	std::vector<double> getVariance() const {
// 	    std::vector<double> tmp;
// 	    for (auto v : variance) {
// 	        tmp.push_back(*v);
// 	    }
// // 	    for (PriorPtr prior : uniquePriors) {
// // 	        variances.push_back(prior->getVariance(0));
// // 	    }
// 		return std::move(tmp);
// 	}

	double logDensity(const DoubleVector& beta) const {

		// TODO assert(beta.size() == listPriors.size());
		double result = 0.0;
		for (size_t i = 0; i < beta.size(); ++i) {
			result += listPriors[i]->logDensity(beta, i);
		}
		return result;
	}

	bool getIsRegularized(const int index) const {
	    return listPriors[index]->getIsRegularized();
	}

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const {
		return listPriors[index]->getDelta(gh, beta, index);
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

// 	JointPrior* clone() const {
// 		PriorList newListPriors(listPriors.size());
//
// 	    // Deep copy of variances
// 	    std::vector<VariancePtr> nVariance;
// 	    for (auto x : variance) {
// 	        nVariance.push_back(bsccs::make_shared<double>(*x));
// 	    }
//
// 		PriorList newUniquePriors;
// 		for (auto& prior : uniquePriors) {
//
//             // prior connects to which variances?
//             auto varianceForPrior = prior->getVarianceParameters();
//
//
// 			PriorPtr newPriorPtr = PriorPtr(prior->clone());
// 			newUniquePriors.push_back(newPriorPtr);
// 			for(size_t i = 0; i < listPriors.size(); ++i) {
// 				if (listPriors[i] == prior) {
// 					newListPriors[i] = newPriorPtr;
// 				}
// 			}
// 		}

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
//
// 		return new MixtureJointPrior(newListPriors, newUniquePriors);
// 	}

private:

	MixtureJointPrior(PriorList listPriors, PriorList uniquePriors) :
		listPriors(listPriors), uniquePriors(uniquePriors) { }

	PriorList listPriors;
	PriorList uniquePriors;
};

// class LaplaceFusedJointPrior : public JointPrior {
// public:
// 	typedef std::vector<PriorPtr> PriorList;
//
// 	LaplaceFusedJointPrior(PriorPtr defaultPrior, int length) : JointPrior(),
// // 	        basePrior(defaultPriot),
// 			hierarchyPriors(length, defaultPrior) {
// 		hierarchyDepth = length;
// 	}
//
// 	void setVariance(double x) {
// 		setVariance(0,x);
// 	}
//
// 	void setVariance(int level, double x) {
// 		hierarchyPriors[level]->setVariance(0, x);
// 	}
//
// 	void changePrior(PriorPtr newPrior, int index) {
// 		// TODO assert(index < listPriors.size());
// 		hierarchyPriors[index] = newPrior;
// //         basePrior = newPrior;
// 	}
//
// 	const std::string getDescription() const {
// 		std::stringstream info;
// 		for (int i = 0; i < hierarchyDepth; i ++) {
// 			info << "Laplace fused level " << i << " has prior " << hierarchyPriors[i]->getDescription() << " ";
// 		}
// 		return info.str();
// 	}
//
// // 	void setHierarchy(HierarchyReader* hierarchyReader) {
// // 		getParentMap = hierarchyReader->returnGetParentMap();
// // 		getChildMap = hierarchyReader->returnGetChildMap();
// // 	}
//
//     void setHierarchy(HierarchicalParentMap parentMap,
//              HierarchicalChildMap childMap) {
//         // TODO getParentMap, getChildMap should be refs; no need to copy
//         getParentMap = parentMap;
//         getChildMap = childMap;
//
// //         std::cout << "parentMap:" << std::endl;
// //         std::for_each(begin(parentMap), end(parentMap), [](int i) {
// //         	std::cout << " " << i;
// //         });
// //         std::cout << std::endl << "childMap:" << std::endl;
// //         for(auto& v : childMap) {
// //         	for(auto x : v) {
// //         		std::cout << " " << x;
// //         	}
// //         	std::cout << std::endl;
// //         }
//     }
//
//     void setNeighborhoods(std::vector<std::vector<int>> neighbors) {
//         neighborMap = neighbors; // TODO No need to copy
//     }
//
// 	std::vector<double> getVariance() const{
// 		return std::vector<double> {
// 		    getVariance(0),
// 		    getVariance(1)
// 		};
// 	}
//
// 	double getVariance(int level) const {
// 		return hierarchyPriors[level]->getVariance(0);
// 	}
//
// 	double logDensity(const DoubleVector& beta) const {
// 		double result = 0.0;
// 		for (DoubleVector::const_iterator it = beta.begin(); it != beta.end(); ++it) {
// 			result += hierarchyPriors[0]->logDensity(*it);
// 		}
// 		return result;
// 	}
//
// 	bool getIsRegularized(const int index) const {
// 	    return true;
// 	}
//
// 	bool getSupportsKktSwindle(const int index) const {
// 		return false; // TODO fix
// 	}
//
// 	bool getSupportsKktSwindle(void) const {
// 		return false; // TODO fix
// 	}
//
// 	double getKktBoundary(const int index) const {
// 		return 0.0; // TODO fix
// 	}
//
// // 		double delta = 0.0;
// //
// // 		double neg_update = - (gh.first - lambda) / gh.second;
// // 		double pos_update = - (gh.first + lambda) / gh.second;
// //
// // 		int signBetaIndex = sign(beta);
// //
// // 		if (signBetaIndex == 0) {
// //
// // 			if (neg_update < 0) {
// // 				delta = neg_update;
// // 			} else if (pos_update > 0) {
// // 				delta = pos_update;
// // 			} else {
// // 				delta = 0;
// // 			}
// // 		} else { // non-zero entry
// //
// // 			if (signBetaIndex < 0) {
// // 				delta = neg_update;
// // 			} else {
// // 				delta = pos_update;
// // 			}
// //
// // 			if ( sign(beta + delta) != signBetaIndex ) {
// // 				delta = - beta;
// // 			}
// // 		}
// // 		return delta;
//
// 	double getDelta(const GradientHessian gh, const DoubleVector& betaVector, const int index) const {
// 		double t1 = 1/hierarchyPriors[0]->getVariance(0); // this is the hyperparameter that is used in the original code
// 		double t2 = 1/hierarchyPriors[1]->getVariance(0);
//
// 		const double beta = betaVector[index];
//
//
// 		typedef std::pair<double,double> PointWeightPair;
//
// 		// Get attraction points
// 		std::vector<PointWeightPair> attractionPoints;
// 		attractionPoints.push_back(std::make_pair(0.0, t1));
//
// 		for (const auto i : neighborMap[index]) {
// 		    attractionPoints.push_back(std::make_pair(betaVector[i], t2));
// 		}
// 		std::sort(std::begin(attractionPoints), std::end(attractionPoints));
//
// 		// Compact to unique points
// 		std::vector<PointWeightPair> uniqueAttractionPoints;
// 		uniqueAttractionPoints.push_back(attractionPoints[0]);
// 		for (auto it = std::begin(attractionPoints) + 1; it != std::end(attractionPoints); ++it) {
// 		    if (uniqueAttractionPoints.back().first == (*it).first) {
// 		        uniqueAttractionPoints.back().second += (*it).second;
// 		    } else {
// 		        uniqueAttractionPoints.push_back(*it);
// 		    }
// 		}
//
// 		bool onAttractionPoint = false;
// 		int pointsToLeft = 0;
//
// 		for (; pointsToLeft < uniqueAttractionPoints.size(); ++pointsToLeft) {
// 		    if (beta <= attractionPoints[pointsToLeft].first) {
// 		        if (beta == attractionPoints[pointsToLeft].first) {
// 		            onAttractionPoint = true;
// 		        }
// 		        break;
// 		    }
// 		}
//
// 	    double leftWeight = 0.0;
// 	    double rightWeight = 0.0;
//
// 	    for (int i = 0; i < pointsToLeft; ++i) {
// 	        leftWeight += uniqueAttractionPoints[i].second;
// 	    }
//
// 	    for (int i = pointsToLeft; i < uniqueAttractionPoints.size(); ++i) {
// 	        rightWeight += uniqueAttractionPoints[i].second;
// 	    }
//
// 		if (onAttractionPoint) {
// 		    // Can we leave point?
//
// 		} else {
// 		    // Move
//
// 		}
//
// 		int parent = getParentMap.at(index);
// 		const std::vector<int>& siblings = getChildMap.at(parent);
// 		double sumBetas = 0;
// //		int nSiblingsOfInterest = 0; //Different from siblings length if weights used
// 		for (size_t i = 0; i < siblings.size(); i++) {
// 			sumBetas += betaVector[siblings[i]];
// 		}
// 		double hessian = t1 - t1 / (siblings.size() + t2/t1);
//
// 		double gradient = t1*betaVector[index] - t1*t1*sumBetas / (siblings.size()*t1 + t2);
//
// 		return (- (gh.first + gradient)/(gh.second + hessian));
// 	}
//
// 	JointPrior* clone() const {
// 		PriorList newHierarchyPriors;
//
// 		for (auto& prior : hierarchyPriors) {
// 			newHierarchyPriors.push_back(PriorPtr(prior->clone()));
// 		}
//
// 		return new LaplaceFusedJointPrior(newHierarchyPriors, hierarchyDepth, getParentMap,
// 			getChildMap);
// 	}
//
// private:
//
//     LaplaceFusedJointPrior(PriorList hierarchyPriors, int hierarchyDepth,
// 		HierarchicalParentMap getParentMap,
// 		HierarchicalChildMap getChildMap) : hierarchyPriors(hierarchyPriors),
// 		hierarchyDepth(hierarchyDepth), getParentMap(getParentMap), getChildMap(getChildMap) { }
//
// //     PriorPtr basePrior;
//
// 	PriorList hierarchyPriors;
// 	int hierarchyDepth;
// 	HierarchicalParentMap getParentMap;
// 	HierarchicalChildMap getChildMap;
//
// 	std::vector<std::vector<int>> neighborMap;
//
// };

class HierarchicalJointPrior : public JointPrior {
public:
	typedef std::vector<PriorPtr> PriorList;

	HierarchicalJointPrior(PriorPtr defaultPrior, int length) : JointPrior(),
			hierarchyPriors(length, defaultPrior) {
		hierarchyDepth = length;
		addVarianceParameters(defaultPrior->getVarianceParameters());
	}

// 	void setVariance(double x) {
// 		setVariance(0,x);
// 	}

// 	void setVariance(int level, double x) {
// 		hierarchyPriors[level]->setVariance(0, x);
// 	}
//
//     void setVariance(int index, double x) {
//         *variance[index] = x;
//     }

	void changePrior(PriorPtr newPrior, int index) {
		// TODO assert(index < listPriors.size());
		hierarchyPriors[index] = newPrior;
		addVarianceParameters(newPrior->getVarianceParameters());
	}

	// void addVariance

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

// 	std::vector<double> getVariance() const{
// 		return std::vector<double> {
// 		    getVariance(0),
// 		    getVariance(1)
// 		};
// 	}

// 	double getVariance(int level) const {
// 		return hierarchyPriors[level]->getVariance(0);
// 	}

	double logDensity(const DoubleVector& beta) const {
		double result = 0.0;
		// for (DoubleVector::const_iterator it = beta.begin(); it != beta.end(); ++it) {
	    for (size_t i = 0; i < beta.size(); ++i) {
			result += hierarchyPriors[0]->logDensity(beta, i);
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
// 		double t1 = 1/hierarchyPriors[0]->getVariance(0); // this is the hyperparameter that is used in the original code
// 		double t2 = 1/hierarchyPriors[1]->getVariance(0);

		double t1 = 1.0 / variance[0].get();
		double t2 = 1.0 / variance[1].get();

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

// 	JointPrior* clone() const {
// 		PriorList newHierarchyPriors;
//
// 		for (auto& prior : hierarchyPriors) {
//
// 		    std::vector<VariancePtr> newPtrs;
// 		    for (auto x : prior->getVarianceParameters()) {
// 		        newPtrs.push_back(bsccs::make_shared<double>(*x)); // deep copy of variances
// 		    }
//
// 			newHierarchyPriors.push_back(PriorPtr(prior->clone(newPtrs)));
// 		}
//
// 		return new HierarchicalJointPrior(newHierarchyPriors, hierarchyDepth, getParentMap,
// 			getChildMap);
// 	}

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

// class LaplaceFusedJointPrior2 : public HierarchicalJointPrior {
//
//     // TODO Implement
//
// };

class FullyExchangeableJointPrior : public JointPrior {
public:
	FullyExchangeableJointPrior(PriorPtr _one) : JointPrior(), singlePrior(_one) {
		addVarianceParameters(singlePrior->getVarianceParameters());
	}

// 	void setVariance(int index, double x) {
// 		variances[index] = x;
// 		singlePrior->setVariance(index, x);
// 	}

// 	std::vector<double> getVariance() const {
// 		return variances;
// 	}

	double logDensity(const DoubleVector& beta) const {
		double result = 0.0;
		// for (DoubleVector::const_iterator it = beta.begin(); it != beta.end(); ++it) {
	    for (size_t i = 0; i < beta.size(); ++i) {
			result += singlePrior->logDensity(beta, i);
		}
		return result;
	}

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const {
		return singlePrior->getDelta(gh, beta, index);
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

// 	JointPrior* clone() const {
// 	    std::vector<VariancePtr> newPtrs;
// 	    for (auto x : variance) {
// 	        newPtrs.push_back(bsccs::make_shared<double>(*x)); // deep copy of variances
// 	    }
//
// 		return new FullyExchangeableJointPrior(PriorPtr(singlePrior->clone(newPtrs)));
// 	}

private:

	PriorPtr singlePrior;
};

typedef bsccs::shared_ptr<JointPrior> JointPriorPtr;

} /* namespace priors */
} /* namespace bsccs */
#endif /* JOINTPRIOR_H_ */
