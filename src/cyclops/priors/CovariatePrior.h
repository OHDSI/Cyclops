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

class CyclicCoordinateDescent; // forward reference

namespace priors {


typedef std::pair<double, double> GradientHessian;
typedef std::vector<double> DoubleVector;

//typedef bsccs::shared_ptr<double> VariancePtr;

template <typename T, typename C>
class CallbackSharedPtr {
public:
    typedef  bsccs::shared_ptr<T> SharedPtr;

public:
 //   CallbackSharedPtr(T* t) : ptr(t), callback(nullptr) { }

//    CallbackSharedPtr(T* t, C* c) : ptr(t), callback(c) { }

    // CallbackSharedPtr(const CallbackSharedPtr& refptr) :  ptr(refptr),
    //     callback(refptr.callback) { }

    CallbackSharedPtr(const SharedPtr& refptr) : ptr(refptr), callback(nullptr) { }

    CallbackSharedPtr(const SharedPtr& refptr, C* c) : ptr(refptr), callback(c) { }

    // const T& operator*() const { return *ptr; }
    //
    // T& operator*() {
    //     std::cerr << "A ";
    //     signal();
    //     return *ptr;
    // }

    void set(const T& t) {
        signal();
        *ptr = t;
    }

    const T& get() const {
        return *ptr;
    }

    // const T* operator->() const { return ptr.operator->(); }
    //
    // T* operator->() {
    //     std::cerr << "B ";
    //     signal();
    //     return ptr.operator->();
    // }

    void setCallback(C* c) {
        callback = c;
    }

    bool operator==(const CallbackSharedPtr<T,C>& rhs) noexcept {
        return ptr == rhs.ptr;
    }

private:
    void signal() {
        // std::cerr << "signal()" << std::endl;
        if (callback != nullptr) {
            callback->callback();
        }
    }

    SharedPtr ptr;
    C* callback;
};

struct Cachable {

    Cachable() : valid(false) { }

    bool isValid() const { return valid; }

protected:
    void setValid(bool v) { valid = v; }

private:
    bool valid;
};

struct CacheCallback : public Cachable {

    CacheCallback() : Cachable() {
        // std::cerr <<"ctor Cachable" << std::endl;
    }

    void callback() {
        // std::cerr << "callback()" << std::endl;
        setValid(false);
    }
};

// struct NoCallback { // TODO Use enable_if
//     void callback() { }
// };

typedef CallbackSharedPtr<double,CacheCallback> VariancePtr;

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

	virtual const std::string getDescription() const = 0; // pure virtual

	virtual double logDensity(const DoubleVector& beta, const int index,
                             CyclicCoordinateDescent& ccd) const = 0; // pure virtual

	virtual double getDelta(const GradientHessian gh, const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const = 0; // pure virtual

	virtual bool getIsRegularized() const = 0; // pure virtual

	virtual bool getSupportsKktSwindle() const = 0; // pure virtual

	virtual double getKktBoundary() const = 0; // pure virtual

	virtual std::vector<VariancePtr> getVarianceParameters() const = 0 ; // pure virtual

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

	std::vector<VariancePtr> getVarianceParameters() const {
		return std::vector<VariancePtr>();
	}

	const std::string getDescription() const {
		return "None";
	}

    double logDensity(const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
        return 0.0;
    }

	bool getIsRegularized() const {
	    return false;
	}

	double getDelta(GradientHessian gh,const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
		return -(gh.first / gh.second); // No regularization
	}

	bool getSupportsKktSwindle() const {
		return false;
	}

	double getKktBoundary() const {
		return 0.0;
	}
};

class JeffreysPrior : public CovariatePrior {
public:
    JeffreysPrior() : CovariatePrior() { }

    virtual ~JeffreysPrior() { }

    const std::string getDescription() const {
        return "Jeffreys (univariable)";
    }

    double logDensity(const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const;

    // double logDensity(const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
    //     double gradient, negativeHessian;
    //
    //     double hessian = ccd.getHessianDiagonal(index);
    //
    //     // modelSpecifics.updateXBeta(0.0, index, false);
    //     // modelSpecifics.computeRemainingStatistics(false);
    //     // modelSpecifics.computeGradientAndHessian(index, &gradient, &negativeHessian, false);
    //
    //
    //     double penalty = -0.5 * std::log(std::abs(hessian));
    //     std::cerr << hessian << " " << penalty << std::endl;
    //     return≥ penalty;
    // }

    double getDelta(GradientHessian gh,const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const;

    bool getIsRegularized() const { return true; }

    bool getSupportsKktSwindle() const { return false; }

    double getKktBoundary() const { return 0.0; }

    std::vector<VariancePtr> getVarianceParameters() const {
        return std::vector<VariancePtr>();
    }
};

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

	const std::string getDescription() const {
	    double lambda = getLambda();
		std::stringstream info;
		info << "Laplace(" << lambda << ")";
		return info.str();
	}
    double logDensity(const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
        auto x = beta[index];
        auto lambda = getLambda();
        return std::log(0.5 * lambda) - lambda * std::abs(x);
    }

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

	double getDelta(GradientHessian gh, const DoubleVector& betaVector, const int index, CyclicCoordinateDescent& ccd) const {

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
		return tmp;
	}

protected:
	double convertVarianceToHyperparameter(double value) const {
		return std::sqrt(2.0 / value);
	}

	double convertHyperparameterToVariance(double value) const {
		return 2.0 / (value * value);
	}

	double getLambda() const {
		return convertVarianceToHyperparameter(variance.get());
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

	virtual ~FusedLaplacePrior() { }

	FusedLaplacePrior(VariancePtr ptr1, VariancePtr ptr2,
				NeighborList neighborList) : LaplacePrior(ptr1), variance2(ptr2),
						neighborList(neighborList) {
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

	double logDensity(const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const;

	double getDelta(const GradientHessian gh, const DoubleVector& betaVector, const int index, CyclicCoordinateDescent& ccd) const;

private:
	double getEpsilon() const {
		return convertVarianceToHyperparameter(variance2.get());
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

	const std::string getDescription() const {
	    double sigma2Beta = getVariance();
		std::stringstream info;
		info << "Normal(" << sigma2Beta << ")";
		return info.str();
	}

	bool getIsRegularized() const {
	    return true;
	}

	bool getSupportsKktSwindle() const {
		return false;
	}

	double getKktBoundary() const {
		return 0.0;
	}

    double logDensity(const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
        auto x = beta[index];
        double sigma2Beta = getVariance();
        return -0.5 * std::log(2.0 * PI * sigma2Beta) - 0.5 * x * x / sigma2Beta;
    }

	double getDelta(GradientHessian gh, const DoubleVector& betaVector, const int index, CyclicCoordinateDescent& ccd) const {
		double sigma2Beta = getVariance();
		double beta = betaVector[index];
		return - (gh.first + (beta / sigma2Beta)) /
				  (gh.second + (1.0 / sigma2Beta));
	}

	std::vector<VariancePtr> getVarianceParameters() const {
	    auto tmp = std::vector<VariancePtr>();
	    tmp.push_back(variance);
		return tmp;
	}

protected:
    double getVariance() const {
        return variance.get();
    }

private:
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

class BarUpdatePrior : public CovariatePrior {
public:

    BarUpdatePrior(double variance) : CovariatePrior(), variance(makeVariance(variance)) {
         // Do nothing
    }

    BarUpdatePrior(VariancePtr ptr) : CovariatePrior(), variance(ptr) {
        // Do nothing
    }

    virtual ~BarUpdatePrior() {
        // Do nothing
    }

    const std::string getDescription() const {
        double sigma2Beta = getVariance();
        std::stringstream info;
        info << "BarUpdate(" << sigma2Beta << ")";
        return info.str();
    }

    bool getIsRegularized() const {
        return true;
    }

    bool getSupportsKktSwindle() const {
        return false;
    }

    double getKktBoundary() const {
        return 0.0;
    }

    double logDensity(const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const {
        auto x = beta[index];
        double sigma2Beta = getVariance();
        return -0.5 * std::log(2.0 * PI * sigma2Beta) - 0.5 * x * x / sigma2Beta;
    }

    double getDelta(GradientHessian gh, const DoubleVector& betaVector, const int index, CyclicCoordinateDescent& ccd) const {

        double beta = betaVector[index];
        double lambda = 1 / getVariance();
        double delta = 0.0;
        double beta2 = (gh.second * beta) - gh.first;
        int signBeta2 = sign(beta2);
        if (std::abs(beta2) < (2 * std::sqrt(lambda * gh.second))) {
            delta = - beta;
        } else if (signBeta2 < 0) {
            delta = - (beta / 2) - gh.first / (2 * gh.second) - std::sqrt(beta2 * beta2 - (4 * lambda * gh.second)) / (2 * gh.second);
        } else if (signBeta2 > 0) {
            delta = - (beta / 2) - gh.first / (2 * gh.second) + std::sqrt(beta2 * beta2 - (4 * lambda * gh.second)) / (2 * gh.second);
        }
        return delta;
    }

    std::vector<VariancePtr> getVarianceParameters() const {
        auto tmp = std::vector<VariancePtr>();
        tmp.push_back(variance);
        return tmp;
    }

protected:
    double getVariance() const {
        return variance.get();
    }

private:
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
};





class HierarchicalNormalPrior : public NormalPrior {
public:
    typedef std::vector<int> NeighborList;

    HierarchicalNormalPrior(VariancePtr ptr1, VariancePtr ptr2, NeighborList neighborList) :
        NormalPrior(ptr1), variance2(ptr2), neighborList(neighborList) { }

    virtual ~HierarchicalNormalPrior() { }

    double logDensity(const DoubleVector& beta, const int index, CyclicCoordinateDescent& ccd) const;

    double getDelta(GradientHessian gh, const DoubleVector& betaVector, const int index, CyclicCoordinateDescent& ccd) const;

    std::vector<VariancePtr> getVarianceParameters() const {
        auto tmp = NormalPrior::getVarianceParameters();
        tmp.push_back(variance2);
        return tmp;
    }

protected:
    double getVariance2() const { return variance2.get(); }

private:
    VariancePtr variance2;
    NeighborList neighborList;
};

} /* namespace priors */
} /* namespace bsccs */
#endif /* COVARIATEPRIOR_H_ */
