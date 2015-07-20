/*
 * CovariatePrior.h
 *
 *  Created on: Nov 5, 2013
 *      Author: msuchard
 */

#ifndef COVARIATEPRIOR_H_
#define COVARIATEPRIOR_H_

#include <memory>
#include <string>
#include <cmath>
#include <sstream>
#include <limits>

#include <iostream> // TODO Remove

#ifndef PI
#define PI	3.14159265358979323851280895940618620443274267017841339111328125
#endif

#include "Types.h"

namespace bsccs {
namespace priors {


typedef std::pair<double, double> GradientHessian;
typedef std::vector<double> DoubleVector;

typedef bsccs::shared_ptr<double> VariancePtr;

class CovariatePrior; // forward declaration
typedef bsccs::shared_ptr<CovariatePrior> PriorPtr;

class CovariatePrior {
public:
	CovariatePrior() {
		// Do nothing
	}
	virtual ~CovariatePrior() {
		// Do nothing
	}

// 	virtual void setVariance(double x) = 0; // pure virtual

// 	virtual void setVariance(int level, double x) = 0; // pure virtual

// 	virtual double getVariance(int level) const = 0; // pure virtual

	virtual const std::string getDescription() const = 0; // pure virtual

	// virtual double logDensity(double x) const = 0; // pure virtual

	virtual double logDensity(const DoubleVector& beta, const int index) const = 0; // pure virtual

	virtual double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index) const = 0; // pure virtual

//	virtual double logDensity(const DoubleVector& vector) const = 0; // pure virtual

	virtual bool getIsRegularized() const = 0; // pure virtual

// 	virtual CovariatePrior* clone() const = 0; // pure virtual

	virtual bool getSupportsKktSwindle() const = 0; // pure virtual

	virtual double getKktBoundary() const = 0; // pure virtual

	virtual std::vector<VariancePtr> getVarianceParameters() const = 0 ; // pure virtual

// 	virtual int getNumberVarianceParameters() const { return 1; }

	static PriorPtr makePrior(PriorType priorType, double variance);

	static VariancePtr makeVariance(double variance) {
	    return bsccs::make_shared<double>(variance);
	}
};

class NoPrior : public CovariatePrior {
public:
	NoPrior() : CovariatePrior() {
		// Do nothing
	}

	virtual ~NoPrior() {
		// Do nothing
	}

// 	double getVariance(int level) const {
// 		return std::numeric_limits<double>::infinity();
// 	}

// 	void setVariance(int level, double x) {
// 		// Do nothing
// 	}

	std::vector<VariancePtr> getVarianceParameters() const {
		return std::vector<VariancePtr>();
	}

	const std::string getDescription() const {
		return "None";
	}

    double logDensity(const DoubleVector& beta, const int index) const {
        return 0.0;
    }

// 	double logDensity(double x) const {
// 		return 0.0;
// 	}
//
// 	double logDensity(const DoubleVector& vector) const {
// 		return 0.0;
// 	}

	bool getIsRegularized() const {
	    return false;
	}

	double getDelta(GradientHessian gh,const DoubleVector& beta, const int index) const {
		return -(gh.first / gh.second); // No regularization
	}

	bool getSupportsKktSwindle() const {
		return false;
	}

	double getKktBoundary() const {
		return 0.0;
	}

// 	CovariatePrior* clone() const {
// 		return new NoPrior(*this);
// 	}
};


// class FusedLaplacePrior : public CovariatePrior {
// public:
// 	FusedLaplacePrior() : CovariatePrior() { }
//
// };

class LaplacePrior : public CovariatePrior {
public:

	LaplacePrior(double variance) : CovariatePrior(), variance(makeVariance(variance)) {
    //LaplacePrior(makeVariance(variance)) { // Not for gcc 4.6.3
		// Do nothing
	}

	LaplacePrior(VariancePtr ptr) : CovariatePrior(), variance(ptr) {
		// Do nothing
	}

	virtual ~LaplacePrior() {
		// Do nothing
	}

// 	double getVariance(int level) const {
// 		if (level == 0) {
// 			return convertHyperparameterToVariance(lambda);
// 		} else {
// 			return std::numeric_limits<double>::infinity();
// 		}
// 	}

	const std::string getDescription() const {
	    double lambda = getLambda();
		std::stringstream info;
		info << "Laplace(" << lambda << ")";
		return info.str();
	}

// 	void setVariance(int level, double x) {
// 		if (level == 0) {
// 			lambda = convertVarianceToHyperparameter(x);
// 		}
// 	}

    double logDensity(const DoubleVector& beta, const int index) const {
        auto x = beta[index];
        auto lambda = getLambda();
        return std::log(0.5 * lambda) - lambda * std::abs(x);
    }

// 	double logDensity(double x) const {
// 	    double lambda = getLambda();
// 		return std::log(0.5 * lambda) - lambda * std::abs(x);
// 	}
//
// 	double logDensity(const DoubleVector& vector) const {
// 		return logIndependentDensity(vector);
// 	}

	bool getIsRegularized() const {
	    return true;
	}

	bool getSupportsKktSwindle() const {
		return true;
	}

	double getKktBoundary() const {
	    double lambda = getLambda();
		return lambda;
	}

	double getDelta(GradientHessian gh, const DoubleVector& betaVector, const int index) const {

		double beta = betaVector[index];
		double lambda = getLambda();
		double delta = 0.0;

		double neg_update = - (gh.first - lambda) / gh.second;
		double pos_update = - (gh.first + lambda) / gh.second;

		int signBetaIndex = sign(beta);

		if (signBetaIndex == 0) {

			if (neg_update < 0) {
				delta = neg_update;
			} else if (pos_update > 0) {
				delta = pos_update;
			} else {
				delta = 0;
			}
		} else { // non-zero entry

			if (signBetaIndex < 0) {
				delta = neg_update;
			} else {
				delta = pos_update;
			}

			if ( sign(beta + delta) != signBetaIndex ) {
				delta = - beta;
			}
		}
		return delta;
	}

	std::vector<VariancePtr> getVarianceParameters() const {
	    auto tmp = std::vector<VariancePtr>();
	    tmp.push_back(variance);
		return std::move(tmp);
	}

// 	CovariatePrior* clone() const {
// 		return new LaplacePrior(*this);
// 	}

protected:
	double convertVarianceToHyperparameter(double value) const {
		return std::sqrt(2.0 / value);
	}

	double convertHyperparameterToVariance(double value) const {
		return 2.0 / (value * value);
	}

	double getLambda() const {
		return convertVarianceToHyperparameter(*variance);
	}

private:
	template <typename Vector>
	typename Vector::value_type logIndependentDensity(const Vector& vector) const {
	    double lambda = getLambda();
		return vector.size() * std::log(0.5 * lambda) - lambda * oneNorm(vector);
	}

	template <typename Vector>
	typename Vector::value_type oneNorm(const Vector& vector) const {
		typename Vector::value_type result = 0.0;
		for (typename Vector::const_iterator it; it != vector.end(); ++it) {
			result += std::abs(*it);
		}
		return result;
	}

	int sign(double x) const {
		if (x == 0) {
			return 0;
		}
		if (x < 0) {
			return -1;
		}
		return 1;
	}

	VariancePtr variance;
// 	double lambda;
};

class FusedLaplacePrior : public LaplacePrior {
public:

	typedef std::vector<int> NeighborList;

	FusedLaplacePrior(double variance1, double variance2,
				NeighborList neighborList) :
	        //FusedLaplacePrior(
		    //	makeVariance(variance1), makeVariance(variance2),
			//   neighborList)  // Not with gcc 4.6.3
			LaplacePrior(makeVariance(variance1)), variance2(makeVariance(variance2)),
			neighborList(neighborList) {
		// Do nothing
	}

	FusedLaplacePrior(VariancePtr ptr1, VariancePtr ptr2,
				NeighborList neighborList) : LaplacePrior(ptr1), variance2(ptr2),
						neighborList(neighborList) {
		// Do nothing
	}

	virtual ~FusedLaplacePrior() {
		// Do nothing
	}

	const std::string getDescription() const {
		std::stringstream info;
		info << "Fused(" << getLambda() << "," << getEpsilon() << ":";
		for (auto i : neighborList) {
			info << "-" << i;
		}
		info << ")";
		return info.str();
	}

	double logDensity(const DoubleVector& beta, const int index) const;

	double getDelta(const GradientHessian gh, const DoubleVector& betaVector, const int index) const;

private:
	double getEpsilon() const {
		return convertVarianceToHyperparameter(*variance2);
	}

	VariancePtr variance2;
	NeighborList neighborList;
};

class NormalPrior : public CovariatePrior {
public:

	NormalPrior(double variance) : CovariatePrior(), variance(makeVariance(variance)) {
     //NormalPrior(makeVariance(variance)) { // Not with gcc 4.6.3
		// Do nothing
	}

	NormalPrior(VariancePtr ptr) : CovariatePrior(), variance(ptr) {
		// Do nothing
	}

	virtual ~NormalPrior() {
		// Do nothing
	}

// 	double getVariance(int level) const {
// 		if (level == 0) {
// 			return sigma2Beta;
// 		} else {
// 			return std::numeric_limits<double>::infinity();
// 		}
// 	}

	const std::string getDescription() const {
	    double sigma2Beta = getVariance();
		std::stringstream info;
		info << "Normal(" << sigma2Beta << ")";
		return info.str();
	}

// 	void setVariance(int level, double x) {
// 		if (level == 0) {
// 			sigma2Beta = x;
// 		}
// 	}

	bool getIsRegularized() const {
	    return true;
	}

	bool getSupportsKktSwindle() const {
		return false;
	}

	double getKktBoundary() const {
		return 0.0;
	}

// 	double logDensity(double x) const {
// 		double sigma2Beta = getVariance();
// 		return -0.5 * std::log(2.0 * PI * sigma2Beta) - 0.5 * x * x / sigma2Beta;
// 	}

    double logDensity(const DoubleVector& beta, const int index) const {
        auto x = beta[index];
        double sigma2Beta = getVariance();
        return -0.5 * std::log(2.0 * PI * sigma2Beta) - 0.5 * x * x / sigma2Beta;
    }

// 	double logDensity(const DoubleVector& vector) const {
// 		return logIndependentDensity(vector);
// 	}

	double getDelta(GradientHessian gh, const DoubleVector& betaVector, const int index) const {
		double sigma2Beta = getVariance();
		double beta = betaVector[index];
		return - (gh.first + (beta / sigma2Beta)) /
				  (gh.second + (1.0 / sigma2Beta));
	}

	std::vector<VariancePtr> getVarianceParameters() const {
	    auto tmp = std::vector<VariancePtr>();
	    tmp.push_back(variance);
		return std::move(tmp);
	}

// 	CovariatePrior* clone() const {
// 		return new NormalPrior(*this);
// 	}

private:
// 	double sigma2Beta;

	double getVariance() const {
		return *variance;
	}

	template <typename Vector>
	typename Vector::value_type logIndependentDensity(const Vector& vector) const {
	    double sigma2Beta = getVariance();
		return -0.5 * vector.size() * std::log(2.0 * PI * sigma2Beta)
				- 0.5 * twoNormSquared(vector) / sigma2Beta;
	}

	template <typename Vector>
	typename Vector::value_type twoNormSquared(const Vector& vector) const {
		typename Vector::value_type norm = 0.0;
		for (typename Vector::const_iterator it = vector.begin();
				it != vector.end(); ++it) {
			norm += (*it) * (*it);
		}
		return norm;
	}

	VariancePtr variance;
};

} /* namespace priors */
} /* namespace bsccs */
#endif /* COVARIATEPRIOR_H_ */
