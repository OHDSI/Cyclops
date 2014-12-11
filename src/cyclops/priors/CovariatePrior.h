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

#ifndef PI
#define PI	3.14159265358979323851280895940618620443274267017841339111328125
#endif

#include "Types.h"

namespace bsccs {
namespace priors {


typedef std::pair<double, double> GradientHessian;
typedef std::vector<double> DoubleVector;

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

	virtual void setVariance(double x) = 0; // pure virtual

	virtual double getVariance() const = 0; // pure virtual

	virtual const std::string getDescription() const = 0; // pure virtual

	virtual double logDensity(double x) const = 0; // pure virtual

	virtual double getDelta(GradientHessian gh, double beta) const = 0; // pure virtual

	virtual double logDensity(const DoubleVector& vector) const = 0; // pure virtual
	
	virtual bool getIsRegularized() const = 0; // pure virtual
	
	static PriorPtr makePrior(PriorType priorType);	    
};



class NoPrior : public CovariatePrior {
public:
	NoPrior() : CovariatePrior() {
		// Do nothing
	}

	virtual ~NoPrior() {
		// Do nothing
	}

	double getVariance() const {
		return std::numeric_limits<double>::infinity();
	}

	void setVariance(double x) {
		// Do nothing
	}

	const std::string getDescription() const {
		return "None";
	}

	double logDensity(double x) const {
		return 0.0;
	}

	double logDensity(const DoubleVector& vector) const {
		return 0.0;
	}
	
	bool getIsRegularized() const {    
	    return false;
	}

	double getDelta(GradientHessian gh, double beta) const {
		return -(gh.first / gh.second); // No regularization
	}
};

class LaplacePrior : public CovariatePrior {
public:
	LaplacePrior() : CovariatePrior(), lambda(1) {
		// Do nothing
	}

	virtual ~LaplacePrior() {
		// Do nothing
	}

	double getVariance() const {
		return convertHyperparameterToVariance(lambda);
	}

	const std::string getDescription() const {
		std::stringstream info;
		info << "Laplace(" << lambda << ")";
		return info.str();
	}

	void setVariance(double x) {
		lambda = convertVarianceToHyperparameter(x);
	}

	double logDensity(double x) const {
		return std::log(0.5 * lambda) - lambda * std::abs(x);
	}

	double logDensity(const DoubleVector& vector) const {
		return logIndependentDensity(vector);
	}
	
	bool getIsRegularized() const {
	    return true;
	}	

	double getDelta(GradientHessian gh, double beta) const {

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

private:

	template <typename Vector>
	typename Vector::value_type logIndependentDensity(const Vector& vector) const {
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

	double convertVarianceToHyperparameter(double value) const {
		return std::sqrt(2.0 / value);
	}

	double convertHyperparameterToVariance(double value) const {
		return 2.0 / (value * value);
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

	double lambda;
};

class NormalPrior : public CovariatePrior {
public:
	NormalPrior() : CovariatePrior(), sigma2Beta(1) {
		// Do nothing
	}

	virtual ~NormalPrior() {
		// Do nothing
	}

	double getVariance() const {
		return sigma2Beta;
	}

	const std::string getDescription() const {
		std::stringstream info;
		info << "Normal(" << sigma2Beta << ")";
		return info.str();
	}

	void setVariance(double x) {
		sigma2Beta = x;
	}
	
	bool getIsRegularized() const {
	    return true;
	}	

	double logDensity(double x) const {
		return -0.5 * std::log(2.0 * PI * sigma2Beta) - 0.5 * x * x / sigma2Beta;
	}

	double logDensity(const DoubleVector& vector) const {
		return logIndependentDensity(vector);
	}

	double getDelta(GradientHessian gh, double beta) const {
		return - (gh.first + (beta / sigma2Beta)) /
				  (gh.second + (1.0 / sigma2Beta));
	}

private:
	double sigma2Beta;

	template <typename Vector>
	typename Vector::value_type logIndependentDensity(const Vector& vector) const {
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
};

} /* namespace priors */
} /* namespace bsccs */
#endif /* COVARIATEPRIOR_H_ */
