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

	virtual double logDensity(const DoubleVector& beta, CyclicCoordinateDescent& ccd) const = 0; // pure virtual

	virtual double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const = 0; // pure virtual

	virtual const std::string getDescription() const = 0; // pure virtual

	virtual bool getIsRegularized(const int index) const = 0; // pure virtual

	virtual bool getSupportsKktSwindle(const int index) const = 0; // pure virtual

	virtual bool getSupportsKktSwindle(void) const = 0; // pure virtual

	virtual double getKktBoundary(const int index) const = 0; // pure virtual

	// See CovariatePrior::getKktResidual(); NaN when not provided
	virtual GradientHessian getKktResidual(const int index, double gradient, double beta) const {
		return GradientHessian(std::numeric_limits<double>::quiet_NaN(), 0.0);
	}

	virtual PriorType getPriorType(const int index) {return PriorType::NONE;};

	// Precision of a quadratic (normal or flat) prior on coordinate index; false for other priors
	virtual bool getNormalPrecision(const int index, double* precision) const {
		return false;
	}

	// Rate of an independent Laplace prior on coordinate index; false for other priors
	virtual bool getLaplaceLambda(const int index, double* lambda) const {
		return false;
	}

	// See CovariatePrior::getNewtonPenalty()
	virtual bool getNewtonPenalty(const int index, double* precision, double* lambda) const {
		return false;
	}

	// See CovariatePrior::setOutcomeScale()
	virtual void setOutcomeScale(double scale) { }

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

	const std::vector<VariancePtr>& getVarianceParameters() const {
		return variance;
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

	PriorType getPriorType(int index) {
		return listPriors[index]->getPriorType();
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

	double logDensity(const DoubleVector& beta, CyclicCoordinateDescent& ccd) const {

		// TODO assert(beta.size() == listPriors.size());
		double result = 0.0;
		for (size_t i = 0; i < beta.size(); ++i) {
			result += listPriors[i]->logDensity(beta, i, ccd);
		}
		return result;
	}

	bool getIsRegularized(const int index) const {
	    return listPriors[index]->getIsRegularized();
	}

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
		return listPriors[index]->getDelta(gh, beta, index, ccd);
	}

	bool getNormalPrecision(const int index, double* precision) const {
		return listPriors[index]->getNormalPrecision(precision);
	}

	bool getLaplaceLambda(const int index, double* lambda) const {
		return listPriors[index]->getLaplaceLambda(lambda);
	}

	bool getNewtonPenalty(const int index, double* precision, double* lambda) const {
		return listPriors[index]->getNewtonPenalty(precision, lambda);
	}

	bool getSupportsKktSwindle(const int index) const {
		return listPriors[index]->getSupportsKktSwindle();
	}

	double getKktBoundary(const int index) const {
		return listPriors[index]->getKktBoundary();
	}

	GradientHessian getKktResidual(const int index, double gradient, double beta) const {
		return listPriors[index]->getKktResidual(gradient, beta);
	}

	void setOutcomeScale(double scale) {
		for (auto& prior : uniquePriors) {
			prior->setOutcomeScale(scale);
		}
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

	double logDensity(const DoubleVector& beta, CyclicCoordinateDescent& ccd) const {
		double result = 0.0;
		// for (DoubleVector::const_iterator it = beta.begin(); it != beta.end(); ++it) {
	    for (size_t i = 0; i < beta.size(); ++i) {
			result += hierarchyPriors[0]->logDensity(beta, i, ccd);
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

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
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

	double logDensity(const DoubleVector& beta, CyclicCoordinateDescent& ccd) const {
		double result = 0.0;
		// for (DoubleVector::const_iterator it = beta.begin(); it != beta.end(); ++it) {
	    for (size_t i = 0; i < beta.size(); ++i) {
			result += singlePrior->logDensity(beta, i, ccd);
		}
		return result;
	}

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
		return singlePrior->getDelta(gh, beta, index, ccd);
	}

	bool getNormalPrecision(const int index, double* precision) const {
		return singlePrior->getNormalPrecision(precision);
	}

	bool getLaplaceLambda(const int index, double* lambda) const {
		return singlePrior->getLaplaceLambda(lambda);
	}

	bool getNewtonPenalty(const int index, double* precision, double* lambda) const {
		return singlePrior->getNewtonPenalty(precision, lambda);
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

	GradientHessian getKktResidual(const int index, double gradient, double beta) const {
		return singlePrior->getKktResidual(gradient, beta);
	}

	void setOutcomeScale(double scale) {
		singlePrior->setOutcomeScale(scale);
	}

	PriorType getPriorType(int index) {
		return singlePrior->getPriorType();
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

// Exact-duplicate columns merged into one coordinate for a fit (Newton plan 2.8, Phase 5). The representative of a
// group of k columns carries the group's total b, and the other members stay fixed at 0. A penalty
// lambda |b_i| + precision / 2 b_i^2 on each member with precision > 0 (Normal, elastic net) is then exactly
// lambda |b| + precision / (2 k) b^2 on b: both terms are least at the equal split, the unique MAP. A Laplace or flat
// prior is exact as it is, with all the mass on the representative. Everything else is the wrapped prior's.
class AliasedJointPrior : public JointPrior {
public:
	AliasedJointPrior(JointPriorPtr wrapped, std::vector<int> groupSize)
		: wrapped(wrapped), groupSize(std::move(groupSize)) {
		addVarianceParameters(this->wrapped->getVarianceParameters());
	}

	// sum over the members at the equal split: the wrapped prior has precision / 2 b^2 on the representative
	double logDensity(const DoubleVector& beta, CyclicCoordinateDescent& ccd) const {
		double value = wrapped->logDensity(beta, ccd);
		for (size_t j = 0; j < groupSize.size(); ++j) {
			double precision = 0.0, lambda = 0.0;
			if (merged(static_cast<int>(j), &precision, &lambda)) {
				value += 0.5 * precision * groupSize[j] * beta[j] * beta[j] * (1.0 - 1.0 / groupSize[j]);
			}
		}
		return value;
	}

	double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
		double precision = 0.0, lambda = 0.0;
		if (!merged(index, &precision, &lambda)) {
			return wrapped->getDelta(gh, beta, index, ccd);
		}
		const double b = beta[index];
		const double gradient = gh.first + precision * b;
		const double hessian = gh.second + precision;
		if (lambda == 0.0) {
			return -gradient / hessian;
		}
		// the elastic-net step: soft-threshold, and stop at 0 rather than cross it
		const double down = -(gradient - lambda) / hessian;
		const double up = -(gradient + lambda) / hessian;
		if (b == 0.0) {
			return down < 0.0 ? down : (up > 0.0 ? up : 0.0);
		}
		const double delta = b < 0.0 ? down : up;
		return ((b + delta) * b <= 0.0) ? -b : delta;
	}

	GradientHessian getKktResidual(const int index, double gradient, double beta) const {
		double precision = 0.0, lambda = 0.0;
		if (!merged(index, &precision, &lambda)) {
			return wrapped->getKktResidual(index, gradient, beta);
		}
		const double residual = (beta == 0.0 && lambda > 0.0) ? std::max(std::abs(gradient) - lambda, 0.0) :
			std::abs(gradient + precision * beta + (beta > 0.0 ? lambda : (beta < 0.0 ? -lambda : 0.0)));
		return GradientHessian(residual, precision);
	}

	bool getNormalPrecision(const int index, double* precision) const {
		if (!wrapped->getNormalPrecision(index, precision)) {
			return false;
		}
		if (groupSize[index] > 1) {
			*precision /= groupSize[index];
		}
		return true;
	}

	bool getLaplaceLambda(const int index, double* lambda) const { return wrapped->getLaplaceLambda(index, lambda); }
	bool getNewtonPenalty(const int index, double* precision, double* lambda) const {
		return merged(index, precision, lambda) || wrapped->getNewtonPenalty(index, precision, lambda);
	}
	const std::string getDescription() const { return wrapped->getDescription(); }
	bool getIsRegularized(const int index) const { return wrapped->getIsRegularized(index); }
	bool getSupportsKktSwindle(const int index) const { return wrapped->getSupportsKktSwindle(index); }
	bool getSupportsKktSwindle(void) const { return wrapped->getSupportsKktSwindle(); }
	double getKktBoundary(const int index) const { return wrapped->getKktBoundary(index); }
	void setOutcomeScale(double scale) { wrapped->setOutcomeScale(scale); }
	PriorType getPriorType(const int index) { return wrapped->getPriorType(index); }

private:
	// a merged group's representative with a quadratic part: the penalty of the group total
	bool merged(const int index, double* precision, double* lambda) const {
		if (groupSize[index] <= 1 || !wrapped->getNewtonPenalty(index, precision, lambda) || !(*precision > 0.0)) {
			return false;
		}
		*precision /= groupSize[index];
		return true;
	}

	JointPriorPtr wrapped;
	std::vector<int> groupSize; // per coefficient: the size of the group it represents (1: not merged)
};

} /* namespace priors */
} /* namespace bsccs */
#endif /* JOINTPRIOR_H_ */
