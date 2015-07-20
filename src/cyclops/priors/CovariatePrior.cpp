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

	double FusedLaplacePrior::getDelta(const GradientHessian gh, const DoubleVector& betaVector, const int index) const {
	    double t1 = getLambda();
	    double t2 = getEpsilon();

	    const double beta = betaVector[index];

	    typedef std::pair<double,double> PointWeightPair;

	    // Get attraction points
	    std::vector<PointWeightPair> allAttractionPoints;
	    allAttractionPoints.push_back(std::make_pair(0.0, t1));

	    for (const auto i : neighborList) {
	        allAttractionPoints.push_back(std::make_pair(betaVector[i], t2));
	    }
	    std::sort(std::begin(allAttractionPoints), std::end(allAttractionPoints));

	    // Compact to unique points
	    std::vector<PointWeightPair> uniqueAttractionPoints;
	    uniqueAttractionPoints.push_back(allAttractionPoints[0]);
	    for (auto it = std::begin(allAttractionPoints) + 1; it != std::end(allAttractionPoints); ++it) {
	        if (uniqueAttractionPoints.back().first == (*it).first) {
	            uniqueAttractionPoints.back().second += (*it).second;
	        } else {
	            uniqueAttractionPoints.push_back(*it);
	        }
	    }

	    bool onAttractionPoint = false;
	    int pointsToLeft = 0;
	    double lambda = 0.0;
	    // double rightWeight = 0.0;

	    for (int i = 0; i < uniqueAttractionPoints.size(); ++i) {
	        if (beta > uniqueAttractionPoints[i].first) {
	            lambda += uniqueAttractionPoints[i].second;
	            ++pointsToLeft;
	        } else if (beta == uniqueAttractionPoints[i].first) {
	            onAttractionPoint = true;
	        } else { // smaller
	            lambda -= uniqueAttractionPoints[i].second;
	        }
	    }

// 	    std::cerr << "Attraction points for variable " << index << " at " << beta << ":";
// 	    for (auto x : uniqueAttractionPoints) {
// 	        std::cerr << " " << x.first << "=" << x.second;
// 	    }
// 	    std::cerr << std::endl;
// 	    std::cerr << "pts to left: " << pointsToLeft << std::endl;
// 	    std::cerr << "lambda: " << lambda << std::endl;
// 	    std::cerr << "Gradient: " << gh.first << std::endl;

	    double delta = 0;
	    if (onAttractionPoint) {
	        // Can we leave point?

// 	        std::cerr << "On point" << std::endl;
	        auto leftWeight = lambda - uniqueAttractionPoints[pointsToLeft].second;
	        auto rightWeight = lambda + uniqueAttractionPoints[pointsToLeft].second;
// 	        std::cerr << "lw: " << leftWeight << " rw: " << rightWeight << " gh: " << gh.first << std::endl;

	        double neg_update = - (gh.first + leftWeight) / gh.second;
	        double pos_update = - (gh.first + rightWeight) / gh.second;

	        if (neg_update < 0) {
	            delta = neg_update;
	        } else if (pos_update > 0) {
	            delta = pos_update;
	        } else {
	            delta = 0.0;
	        }

	    } else {
	        // Move
	        delta = - (gh.first + lambda) / gh.second;
// 	        std::cerr << "Not on point: " << delta << std::endl;
	    }

	    // Go no further than next point
	    if (pointsToLeft > 0 && beta + delta < uniqueAttractionPoints[pointsToLeft - 1].first) {
	        delta = uniqueAttractionPoints[pointsToLeft - 1].first - beta;
// 	        std::cerr << "Truncate to left" << std::endl;
	    } else if (pointsToLeft + 1 < uniqueAttractionPoints.size()
                    && beta + delta > uniqueAttractionPoints[pointsToLeft + 1].first) {
	        delta = uniqueAttractionPoints[pointsToLeft + 1].first - beta;
// 	        std::cerr << "Truncate to right" << std::endl;
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
