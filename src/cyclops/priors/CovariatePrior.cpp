#include <algorithm>

#include "priors/CovariatePrior.h"

namespace bsccs {
	namespace priors {

	double FusedLaplacePrior::logDensity(const DoubleVector& betaVector, const int index) const {
	    auto x = betaVector[index];
	    auto lambda = getLambda();
	    auto epsilon = getEpsilon();

	    double logDensity = std::log(0.5 * lambda) - lambda * std::abs(x);

	    for (const auto x0 : neighborList) {
	        logDensity += std::log(0.5 * epsilon) - epsilon * std::abs(x - x0) - std::log(2.0);
	    }

	    return logDensity;
	}

	namespace details {
	inline void processAttractionPoint(const double x, const double x0, const double weight,
                                double& lambda, double& leftWeight, double& rightWeight, bool& onAttractionPoint,
                                double& leftClosestAttractionPoint, double& rightClosestAttractionPoint) {
	    if (x > x0) {
	        lambda += weight;
	        if (x0 > leftClosestAttractionPoint) {
	            leftClosestAttractionPoint = x0;
	        }
	    } else if (x < x0) {
	        lambda -= weight;
	        if (x0 < rightClosestAttractionPoint) {
	            rightClosestAttractionPoint = x0;
	        }
	    } else {
	        onAttractionPoint = true;
	        leftWeight  = -weight;
	        rightWeight =  weight;
	    }
	}
	}; // namespace details

	double FusedLaplacePrior::getDelta(const GradientHessian gh, const DoubleVector& betaVector, const int index) const {
	    const auto t1 = getLambda();
	    const auto t2 = getEpsilon();
	    const auto beta = betaVector[index];

// 	    typedef std::pair<double,double> PointWeightPair;
//
// 	    // Get attraction points
// 	    std::vector<PointWeightPair> allAttractionPoints;
// 	    allAttractionPoints.push_back(std::make_pair(0.0, t1));
//
// 	    for (const auto i : neighborList) {
// 	        allAttractionPoints.push_back(std::make_pair(betaVector[i], t2));
// 	    }
// 	    std::sort(std::begin(allAttractionPoints), std::end(allAttractionPoints));
//
// 	    // Compact to unique points
// 	    std::vector<PointWeightPair> uniqueAttractionPoints;
// 	    uniqueAttractionPoints.push_back(allAttractionPoints[0]);
// 	    for (auto it = std::begin(allAttractionPoints) + 1; it != std::end(allAttractionPoints); ++it) {
// 	        if (uniqueAttractionPoints.back().first == (*it).first) {
// 	            uniqueAttractionPoints.back().second += (*it).second;
// 	        } else {
// 	            uniqueAttractionPoints.push_back(*it);
// 	        }
// 	    }
//
// 	    bool onAttractionPoint = false;
// 	    int pointsToLeft = 0;
// 	    double lambda = 0.0;
// 	    // double rightWeight = 0.0;
//
// 	    for (int i = 0; i < uniqueAttractionPoints.size(); ++i) {
// 	        if (beta > uniqueAttractionPoints[i].first) {
// 	            lambda += uniqueAttractionPoints[i].second;
// 	            ++pointsToLeft;
// 	        } else if (beta == uniqueAttractionPoints[i].first) {
// 	            onAttractionPoint = true;
// 	        } else { // smaller
// 	            lambda -= uniqueAttractionPoints[i].second;
// 	        }
// 	    }

	    double lambda2 = 0.0;
	    double leftWeight2 = 0.0;
	    double rightWeight2 = 0.0;

	    bool onAttractionPoint2 = false;
	    double leftClosestAttractionPoint  = -std::numeric_limits<double>::max();
	    double rightClosestAttractionPoint =  std::numeric_limits<double>::max();

	    // Handle lasso @ 0
	    details::processAttractionPoint(beta, 0.0, t1,
                               lambda2, leftWeight2, rightWeight2, onAttractionPoint2,
                               leftClosestAttractionPoint, rightClosestAttractionPoint);

	    // Handle fused lasso @ neighbors
	    for (const auto i : neighborList) {
	        details::processAttractionPoint(beta, betaVector[i], t2,
                                   lambda2, leftWeight2, rightWeight2, onAttractionPoint2,
                                   leftClosestAttractionPoint, rightClosestAttractionPoint);
	    }

	    leftWeight2 += lambda2;
	    rightWeight2 += lambda2;

// 	    std::cerr << "Attraction points for variable " << index << " at " << beta << ":";
// 	    for (auto x : uniqueAttractionPoints) {
// 	        std::cerr << " " << x.first << "=" << x.second;
// 	    }
// 	    std::cerr << std::endl;
// 	    std::cerr << "pts to left: " << pointsToLeft << std::endl;
// 	    std::cerr << "lambda: " << lambda << std::endl;
// 	    std::cerr << "Gradient: " << gh.first << std::endl;

//         if (lambda != lambda2) {
//             std::cerr << "Unequal lambda" << std::endl;
//         }
//
//         if (onAttractionPoint != onAttractionPoint2) {
//             std::cerr << "Unequal attraction point" << std::endl;
//         }


	    double delta = 0;
	    if (onAttractionPoint2) {
	        // Can we leave point?

// 	        auto leftWeight = lambda - uniqueAttractionPoints[pointsToLeft].second;
// 	        auto rightWeight = lambda + uniqueAttractionPoints[pointsToLeft].second;
//
// 	        if (leftWeight != leftWeight2) {
// 	            std::cerr << "Unequal left weight" << std::endl;
// 	        }
// 	        if (rightWeight != rightWeight2) {
// 	            std::cerr << "Unequal right weight" << std::endl;
// 	        }

	        double neg_update = - (gh.first + leftWeight2) / gh.second;
	        double pos_update = - (gh.first + rightWeight2) / gh.second;

	        if (neg_update < 0) {
	            delta = neg_update;
	        } else if (pos_update > 0) {
	            delta = pos_update;
	        } else {
	            delta = 0.0;
	        }

	    } else {
	        // Move
	        delta = - (gh.first + lambda2) / gh.second;
// 	        std::cerr << "Not on point: " << delta << std::endl;
	    }

// 	    if (pointsToLeft > 0) {
// 	        if (uniqueAttractionPoints[pointsToLeft - 1].first != leftClosestAttractionPoint) {
// 	            std::cerr << "Unequal left attraction points" << std::endl;
// 	        }
// 	    }

// 	    if (pointsToLeft < uniqueAttractionPoints.size()) {
// 	        if (uniqueAttractionPoints[pointsToLeft].first != rightClosestAttractionPoint) {
// 	            std::cerr << "Unequal right attraction points" << std::endl;
// 	            std::cerr << uniqueAttractionPoints[pointsToLeft].first
//                        << " " << rightClosestAttractionPoint << " : " << beta << std::endl;
//
// 	            for (const auto x : uniqueAttractionPoints) {
// 	                std::cerr << " " << x.first;
// 	            }
// 	            std::cerr << " : " << pointsToLeft << std::endl << std::endl;
// 	        }
// 	    }

	    // Go no further than next point
// 	    if (pointsToLeft > 0 && beta + delta < uniqueAttractionPoints[pointsToLeft - 1].first) {
// 	        delta = uniqueAttractionPoints[pointsToLeft - 1].first - beta;
// // 	        std::cerr << "Truncate to left" << std::endl;
// 	    } else if (pointsToLeft < uniqueAttractionPoints.size()
//                     && beta + delta > uniqueAttractionPoints[pointsToLeft].first) {
// 	        delta = uniqueAttractionPoints[pointsToLeft].first - beta;
// // 	        std::cerr << "Truncate to right" << std::endl;
// 	    }
        if (beta + delta < leftClosestAttractionPoint) {
            delta = leftClosestAttractionPoint - beta;
        } else if (beta + delta > rightClosestAttractionPoint) {
            delta = rightClosestAttractionPoint - beta;
        }

// 	    std::cerr << std::endl;

	    return delta;
	}

// 	double neg_update = - (gh.first - lambda) / gh.second;
// 	double pos_update = - (gh.first + lambda) / gh.second;
//
// 	int signBetaIndex = sign(beta);
//
// 	if (signBetaIndex == 0) {
//
// 	    if (neg_update < 0) {
// 	        delta = neg_update;
// 	    } else if (pos_update > 0) {
// 	        delta = pos_update;
// 	    } else {
// 	        delta = 0;
// 	    }
// 	} else { // non-zero entry
//
// 	    if (signBetaIndex < 0) {
// 	        delta = neg_update;
// 	    } else {
// 	        delta = pos_update;
// 	    }
//
// 	    if ( sign(beta + delta) != signBetaIndex ) {
// 	        delta = - beta;
// 	    }
// 	}

    double HierarchicalNormalPrior::logDensity(const DoubleVector& beta, const int index) const {
        auto x = beta[index];
        double sigma2Beta = getVariance();
        return -0.5 * std::log(2.0 * PI * sigma2Beta) - 0.5 * x * x / sigma2Beta;
    }

	double HierarchicalNormalPrior::getDelta(GradientHessian gh, const DoubleVector& betaVector, const int index) const {
	    double sigma2Beta = getVariance();
	    double beta = betaVector[index];
	    return - (gh.first + (beta / sigma2Beta)) /
	    (gh.second + (1.0 / sigma2Beta));
	}

   PriorPtr CovariatePrior::makePrior(PriorType priorType, double variance) {
        PriorPtr prior;
        switch (priorType) {
            case NONE :
                prior = bsccs::make_shared<NoPrior>();
                break;
            case LAPLACE :
                prior = bsccs::make_shared<LaplacePrior>(variance);
                break;
            case NORMAL :
                prior = bsccs::make_shared<NormalPrior>(variance);
                break;
            default : break;
        }
        return prior;
    }
	} // namespace priors
} // namespace bsccs
