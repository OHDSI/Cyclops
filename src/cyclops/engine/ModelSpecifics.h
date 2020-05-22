/*
 * ModelSpecifics.h
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef MODELSPECIFICS_H_
#define MODELSPECIFICS_H_

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <complex>

// #define CYCLOPS_DEBUG_TIMING
// #define CYCLOPS_DEBUG_TIMING_LOW

#ifdef CYCLOPS_DEBUG_TIMING
    #include "Timing.h"
#endif

#include <type_traits>
#include <iterator>
//#include <boost/tuple/tuple.hpp>
//#include <boost/iterator/zip_iterator.hpp>
//#include <boost/range/iterator_range.hpp>

//#include <boost/iterator/permutation_iterator.hpp>
//#include <boost/iterator/transform_iterator.hpp>
//#include <boost/iterator/zip_iterator.hpp>
//#include <boost/iterator/counting_iterator.hpp>

#include "AbstractModelSpecifics.h"
#include "Iterators.h"
#include "ParallelLoops.h"

#define Fraction std::complex

namespace bsccs {

template <typename RealType>
class Storage {
public:
    typedef typename CompressedDataMatrix<RealType>::RealVector RealVector;

    Storage(const RealVector& y, const RealVector& offs) : hY(y), hOffs(offs) { }

protected:
    const RealVector& hY;
    const RealVector& hOffs;

    RealVector hXBeta;
    RealVector offsExpXBeta;
    RealVector denomPid;
    RealVector numerPid;
    RealVector numerPid2;

    RealVector hNWeight;
    RealVector hKWeight;
};

template <class BaseModel, typename RealType>
class ModelSpecifics : public AbstractModelSpecifics, BaseModel {
public:

    typedef typename CompressedDataMatrix<RealType>::RealVector RealVector;

	ModelSpecifics(const ModelData<RealType>& input);

	virtual ~ModelSpecifics();

	void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian,  bool useWeights);

	virtual void computeMMGradientAndHessian(
			std::vector<GradientHessian>& gh,
			const std::vector<bool>& fixBeta,
			bool useWeights);

	AbstractModelSpecifics* clone() const;

	virtual const std::vector<double> getXBeta();

	virtual const std::vector<double> getXBetaSave();

	virtual void saveXBeta();

	virtual void zeroXBeta();

	virtual void axpyXBeta(const double beta, const int j);

	virtual void computeXBeta(double* beta, bool useWeights);

	//virtual double getGradientObjective();

	virtual void deviceInitialization();

	void initialize(
	        int iN,
	        int iK,
	        int iJ,
	        const void* iXi,
	        // const CompressedDataMatrix<double>* iXI, // TODO Change to const&
	        double* iNumerPid,
	        double* iNumerPid2,
	        double* iDenomPid,
	        double* iXjY,
	        std::vector<std::vector<int>* >* iSparseIndices,
	        const int* iPid,
	        double* iOffsExpXBeta,
	        double* iXBeta,
	        double* iOffs,
	        double* iBeta,
	        const double* iY);
protected:

    const ModelData<RealType>& modelData;
    const CompressedDataMatrix<RealType>& hX;

    typedef bsccs::shared_ptr<CompressedDataMatrix<RealType>> CdmPtr;
    CdmPtr hXt;

    // Moved from AMS
    RealVector accDenomPid;
    RealVector accNumerPid;
    RealVector accNumerPid2;

    // const RealVector& hY;
    // const RealVector& hOffs;
    // 	const std::vector<int>& hPid;
    using BaseModel::hY;
    using BaseModel::hOffs;
    using BaseModel::hXBeta;

 //   RealVector hXBeta; // TODO Delegate to ModelSpecifics
    RealVector hXBetaSave; // Delegate

    using BaseModel::offsExpXBeta;
    using BaseModel::denomPid;
    using BaseModel::numerPid;
    using BaseModel::numerPid2;

    // RealVector offsExpXBeta;
    // RealVector denomPid;
    // RealVector numerPid;
    // RealVector numerPid2;

    RealVector hXjY;
    RealVector hXjX;
    RealType logLikelihoodFixedTerm;

    typedef bsccs::shared_ptr<CompressedDataColumn<RealType>> CDCPtr;
    typedef std::map<int, CDCPtr> HessianSparseMap;
    HessianSparseMap hessianSparseCrossTerms;

    // End of AMS move

	template <typename IteratorType>
	void axpy(RealType* y, const RealType alpha, const int index);

	void computeNumeratorForGradient(int index, bool useWeights);

	void computeFisherInformation(int indexOne, int indexTwo, double *oinfo, bool useWeights);

	void updateXBeta(double delta, int index, bool useWeights);

	void computeRemainingStatistics(bool useWeights);

	void computeAccumlatedNumerator(bool useWeights);

	void computeAccumlatedDenominator(bool useWeights);

	void computeFixedTermsInLogLikelihood(bool useCrossValidation);

	void computeFixedTermsInGradientAndHessian(bool useCrossValidation);

	double getLogLikelihood(bool useCrossValidation);

	double getPredictiveLogLikelihood(double* weights);

	double getGradientObjective(bool useCrossValidation);

	void getPredictiveEstimates(double* y, double* weights);

	bool allocateXjY(void);

	bool allocateXjX(void);

	bool allocateNtoKIndices(void);

	bool sortPid(void);

	void setWeights(double* inWeights, bool useCrossValidation);

	void doSortPid(bool useCrossValidation);

	template <typename AnyRealType>
	void setPidForAccumulation(const AnyRealType* weights);

	void setupSparseIndices(const int max);

	bool initializeAccumulationVectors(void);

	bool hasResetableAccumulators(void);

	void printTiming(void);

	using Storage<RealType>::hNWeight;
	using Storage<RealType>::hKWeight;

	// std::vector<RealType> hNWeight;
	// std::vector<RealType> hKWeight;

#ifdef CYCLOPS_DEBUG_TIMING
	//	std::vector<double> duration;
	std::map<std::string,long long> duration;
#endif

private:

    template <class Weights>
    void computeRemainingStatisticsImpl();

	template <class IteratorType>
	void computeXBetaImpl(double *beta);

	template <class IteratorType, class Weights>
	void computeGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian, Weights w);

	template <class IteratorType, class Weights>
	void computeMMGradientAndHessianImpl(
			int index, double *ogradient,
            double *ohessian, Weights w);

	template <class IteratorType, class Weights>
	void incrementNumeratorForGradientImpl(int index);

	template <class IteratorType, class Weights>
	void updateXBetaImpl(RealType delta, int index);

	template <class OutType, class InType>
	void incrementByGroup(OutType* values, int* groups, int k, InType inc) {
	    values[BaseModel::getGroup(groups, k)] += inc; // TODO delegate to BaseModel (different in tied-models)
	}

	template <typename IteratorTypeOne, class Weights>
	void dispatchFisherInformation(int indexOne, int indexTwo, double *oinfo, Weights w);

	template <class IteratorTypeOne, class IteratorTypeTwo, class Weights>
	void computeFisherInformationImpl(int indexOne, int indexTwo, double *oinfo, Weights w);

	template<class IteratorType>
	SparseIterator<RealType> getSubjectSpecificHessianIterator(int index);

	void computeXjY(bool useCrossValidation);

	void computeXjX(bool useCrossValidation);

	void computeNtoKIndices(bool useCrossValidation);

	void initializeMmXt();

	void initializeMM(
	    MmBoundType boundType,
		const std::vector<bool>& fixBeta
	);

	void computeNorms(void);

	template <class InteratorType>
	void incrementNormsImpl(int index);

	std::vector<int> hNtoK;

	RealVector norm;

	struct WeightedOperation {
		const static bool isWeighted = true;
	} weighted;

	struct UnweightedOperation {
		const static bool isWeighted = false;
	} unweighted;

	ParallelInfo info;
};

template <typename RealType>
class CompareSurvivalTuples {
	bool useCrossValidation;
	std::vector<RealType>& weight;
	const std::vector<RealType>& z;
public:
	CompareSurvivalTuples(bool _ucv,
			std::vector<RealType>& _weight,
			const std::vector<RealType>& _z)
		: useCrossValidation(_ucv), weight(_weight), z(_z) {
		// Do nothing
	}
	bool operator()(size_t i, size_t j) { // return true if element[i] is before element[j]
		if (useCrossValidation) {
			if (weight[i] > weight[j]) { 			// weight[i] = 1, weight[j] = 0
				return true; // weighted first
			} else if (weight[i] < weight[j]) {		// weight[i] = 0, weight[j] = 1
				return false;
			}
		}
		return z[i] > z[j]; // TODO Throw error if non-unique y
	}
};

struct GroupedData {
public:
	const static bool hasStrataCrossTerms = true;

	const static bool hasNtoKIndices = true;

	const static bool exactTies = false;

	int getGroup(const int* groups, int k) {
		return groups[k];
	}

	const static bool hasIndependentRows = false;
};

struct GroupedWithTiesData : GroupedData {
public:
	const static bool exactTies = true;
};

struct OrderedData {
public:
	const static bool hasStrataCrossTerms = true;

	const static bool hasNtoKIndices = false;

	const static bool exactTies = false;

	int getGroup(const int* groups, int k) {
		return k; // No ties
	}

	const static bool hasResetableAccumulators = true;

	const static bool hasIndependentRows = false;
};

struct OrderedWithTiesData {
public:
	const static bool hasStrataCrossTerms = true;

	const static bool hasNtoKIndices = false;

	const static bool exactTies = false;

	int getGroup(const int* groups, int k) {
		return groups[k];
	}

	const static bool hasResetableAccumulators = false;

	const static bool hasIndependentRows = false;
};

struct IndependentData {
public:
	const static bool hasStrataCrossTerms = false;

	const static bool hasNtoKIndices = false;

	const static bool exactTies = false;

	int getGroup(const int* groups, int k) {
		return k;
	}

	const static bool hasIndependentRows = true;
};

struct FixedPid {
	const static bool sortPid = false;

	const static bool cumulativeGradientAndHessian = false;

	const static bool hasResetableAccumulators = false;
};

struct SortedPid {
	const static bool sortPid = true;

	const static bool cumulativeGradientAndHessian = true;
};

struct NoFixedLikelihoodTerms {
	const static bool likelihoodHasFixedTerms = false;

    template <typename RealType>
	RealType logLikeFixedTermsContrib(RealType yi, RealType offseti, RealType logoffseti) {
   	throw new std::logic_error("Not model-specific");
		return static_cast<RealType>(0);
	}
};

struct GLMProjection {
public:
	const static bool precomputeGradient = true; // XjY

	const static bool likelihoodHasDenominator = true;

	const static bool hasTwoNumeratorTerms = true;

	const static bool exactCLR = false;

	template <class XType, typename RealType>
	static RealType gradientNumeratorContrib(XType x, RealType predictor, RealType xBeta, RealType y) {
//		using namespace indicator_sugar;
		return predictor * x;
	}

    template <typename RealType>
    RealType logLikeNumeratorContrib(int yi, RealType xBetai) {
		return yi * xBetai;
	}

	template <class XType, typename RealType>
	static RealType gradientNumerator2Contrib(XType x, RealType predictor) {
		return predictor * x * x;
	}
};

template <typename RealType>
struct Survival {
public:
	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			RealType* information,
			RealType predictor,
			RealType numer, RealType numer2, RealType denom,
			RealType weight,
			RealType x, RealType xBeta, RealType y) {
		*information += weight * predictor / denom * it.value();
	}

	template <class IteratorType, class Weights>
	inline void incrementMMGradientAndHessian(
	        RealType& gradient, RealType& hessian,
	        RealType expXBeta, RealType denominator,
	        RealType weight, RealType x, RealType xBeta,
	        RealType y, RealType norm) {

	    if (IteratorType::isIndicator) {
	        gradient += weight * expXBeta / denominator;
	        hessian += weight * expXBeta / denominator;
	        //hessian += weight * expXBeta / denominator * norm;
	    } else {
	        gradient += weight * expXBeta * x / denominator;
	        hessian += weight * expXBeta * x * x / denominator;
	        //hessian += weight * expXBeta * x * x / denominator * norm;
	    }

	    // throw new std::logic_error("Not model-specific");
	}
};

template <typename RealType>
struct Logistic {
public:
	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			RealType* information,
			RealType predictor,
			RealType numer, RealType numer2, RealType denom,
			RealType weight,
			RealType x, RealType xBeta, RealType y) {
		const RealType g = predictor / denom;
		*information += weight *
				 (predictor / denom - g * g)
//
//				predictor / (denom * denom)
				 * it.value();
	}

	template <class IteratorType, class Weights> // TODO Code duplication with LR
	static void incrementGradientAndHessian(
			const IteratorType& it,
			Weights w,
			RealType* gradient, RealType* hessian,
			RealType numer, RealType numer2, RealType denom,
			RealType weight,
			RealType x, RealType xBeta, RealType y) {

	    const RealType g = numer / denom;
	    if (Weights::isWeighted) {
	        *gradient += weight * g;
	    } else {
	        *gradient += g;
	    }
	    if (IteratorType::isIndicator) {
	        if (Weights::isWeighted) {
	            *hessian += weight * g * (static_cast<RealType>(1.0) - g);
	        } else {
	            *hessian += g * (static_cast<RealType>(1.0) - g);
	        }
	    } else {
	        if (Weights::isWeighted) {
	            *hessian += weight * (numer2 / denom - g * g); // Bounded by x_j^2
	        } else {
	            *hessian += (numer2 / denom - g * g); // Bounded by x_j^2
	        }
	    }
	}

	template <class IteratorType, class Weights>
	inline void incrementMMGradientAndHessian(
	        RealType& gradient, RealType& hessian,
	        RealType expXBeta, RealType denominator,
	        RealType weight, RealType x, RealType xBeta,
	        RealType y, RealType norm) {

	    if (IteratorType::isIndicator) {
	        gradient += weight * expXBeta / denominator;
	        hessian += weight * expXBeta / denominator; // * norm;
	    } else {
	        gradient += weight * expXBeta * x / denominator;
	        hessian += weight * expXBeta * x * x / denominator; // * norm;
	    }
	}

	template <class IteratorType, class WeightOperationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

    	const RealType g = numerator / denominator;

        const RealType gradient =
            (WeightOperationType::isWeighted) ? weight * g : g;

        const RealType hessian =
            (IteratorType::isIndicator) ?
                (WeightOperationType::isWeighted) ?
                    weight * g * (static_cast<RealType>(1.0) - g) :
                    g * (static_cast<RealType>(1.0) - g)
                :
                (WeightOperationType::isWeighted) ?
                    weight * (numerator2 / denominator - g * g) :
                    (numerator2 / denominator - g * g);

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }
};

template <typename RealType>
struct SelfControlledCaseSeries : public Storage<RealType>, GroupedData, GLMProjection, FixedPid, Survival<RealType> {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    SelfControlledCaseSeries(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

	const static bool precomputeHessian = false; // XjX

	const static bool likelihoodHasFixedTerms = true;

	RealType logLikeFixedTermsContrib(RealType yi, RealType offseti, RealType logoffseti) {
		return yi * std::log(offseti);
	}

	static RealType getDenomNullValue () { return static_cast<RealType>(0); }

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(yi);
	}

	template <class IteratorType, class Weights>
	static void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			RealType* gradient, RealType* hessian,
			RealType numer, RealType numer2, RealType denom,
			RealType nEvents,
			RealType x, RealType xBeta, RealType y) {

		const RealType t = numer / denom;
		const RealType g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<RealType>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class Weights>
	inline void incrementMMGradientAndHessian(
	        RealType& gradient, RealType& hessian,
	        RealType expXBeta, RealType denominator,
	        RealType weight, RealType x, RealType xBeta,
	        RealType y, RealType norm) {

		if (IteratorType::isIndicator) {
			gradient += weight * expXBeta / denominator;
			hessian += weight * expXBeta / denominator; // * norm;
		} else {
			gradient += weight * expXBeta * x / denominator;
			hessian += weight * expXBeta * x * x / denominator; // * norm;
		}
	}

	template <class IteratorType, class WeightOperationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

        // Same as CLR and CPR, TODO Remove code-duplication

    	const RealType g = numerator / denominator;
    	const RealType gradient = weight * g; // Always use weights (number of events)

        const RealType hessian =
            (IteratorType::isIndicator) ?
                gradient * (static_cast<RealType>(1.0) - g) :
                weight * (numerator2 / denominator - g * g);

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }

	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
        return offs * std::exp(xBeta);
    }

	RealType getOffsExpXBeta(const RealType* offs, RealType xBeta, RealType y, int k) {
		return offs[k] * std::exp(xBeta);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType denom) {
		return ni * std::log(denom);
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	RealType predictEstimate(RealType xBeta){
		return xBeta;
	}

};

template <typename RealType>
struct ConditionalPoissonRegression : public Storage<RealType>, GroupedData, GLMProjection, FixedPid, Survival<RealType> {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    ConditionalPoissonRegression(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

	const static bool precomputeHessian = false; // XjX

	const static bool likelihoodHasFixedTerms = true;

    RealType logLikeFixedTermsContrib(RealType yi, RealType offseti, RealType logoffseti) {
	    RealType logLikeFixedTerm = static_cast<RealType>(0);
		for(int i = 2; i <= (int)yi; i++)
			logLikeFixedTerm -= std::log(static_cast<RealType>(i));
		return logLikeFixedTerm;
	}

	static RealType getDenomNullValue () { return static_cast<RealType>(0); }

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(yi);
	}

	template <class IteratorType, class Weights>
	static void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			RealType* gradient, RealType* hessian,
			RealType numer, RealType numer2, RealType denom,
			RealType nEvents,
			RealType x, RealType xBeta, RealType y) {

		const RealType t = numer / denom;
		const RealType g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<RealType>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

        // Same as CLR, TODO Remove code-duplication

    	const RealType t = numerator / denominator;
    	const RealType gradient = weight * t; // Always use weights (number of events)

        const RealType hessian =
            (IteratorType::isIndicator) ?
                gradient * (static_cast<RealType>(1.0) - t) :
                weight * (numerator2 / denominator - t * t);

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }

	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
        return std::exp(xBeta);
    }

	RealType getOffsExpXBeta(const RealType* offs, RealType xBeta, RealType y, int k) {
		return std::exp(xBeta);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType denom) {
		return ni * std::log(denom);
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	RealType predictEstimate(RealType xBeta){
		return xBeta;
	}

};

template <typename RealType>
struct ConditionalLogisticRegression : public Storage<RealType>, GroupedData, GLMProjection, FixedPid, Survival<RealType> {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    ConditionalLogisticRegression(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

	const static bool precomputeHessian = false; // XjX
	const static bool likelihoodHasFixedTerms = false;

	RealType logLikeFixedTermsContrib(RealType yi, RealType offseti, RealType logoffseti) {
        throw new std::logic_error("Not model-specific");
		return static_cast<RealType>(0);
	}

	static RealType getDenomNullValue () { return static_cast<RealType>(0); }

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(yi);
	}

	template <class IteratorType, class Weights>
	static void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			RealType* gradient, RealType* hessian,
			RealType numer, RealType numer2, RealType denom,
			RealType nEvents,
			RealType x, RealType xBeta, RealType y) {

		const RealType t = numer / denom;
		const RealType g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<RealType>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOerationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

    	const RealType g = numerator / denominator;
    	const RealType gradient = weight * g; // Always use weights (number of events)

        const RealType hessian =
            (IteratorType::isIndicator) ?
                gradient * (static_cast<RealType>(1.0) - g) :
                weight * (numerator2 / denominator - g * g);

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }

	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
        return std::exp(xBeta);
	}

	RealType getOffsExpXBeta(const RealType* offs, RealType xBeta, RealType y, int k) {
		return std::exp(xBeta);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType denom) {
		return ni * std::log(denom);
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	RealType predictEstimate(RealType xBeta){
		return xBeta;
	}

};

template <typename RealType>
struct TiedConditionalLogisticRegression : public Storage<RealType>, GroupedWithTiesData, GLMProjection, FixedPid, Survival<RealType> {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    TiedConditionalLogisticRegression(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

	const static bool precomputeGradient = true; // XjY   // TODO Until tied calculations are only used for ties
	const static bool precomputeHessian = false; // XjX
	const static bool likelihoodHasFixedTerms = false;
	const static bool exactCLR = true;

	RealType logLikeFixedTermsContrib(RealType yi, RealType offseti, RealType logoffseti) {
        throw new std::logic_error("Not model-specific");
		return static_cast<RealType>(0);
	}

	static RealType getDenomNullValue () { return static_cast<RealType>(0); }

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(yi);
	}

	template <class IteratorType, class Weights>
	static void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			RealType* gradient, RealType* hessian,
			RealType numer, RealType numer2, RealType denom,
			RealType nEvents,
			RealType x, RealType xBeta, RealType y) {

		const RealType t = numer / denom;
		const RealType g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<RealType>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

//         throw new std::logic_error("tied clr model not yet support");

    	const RealType g = numerator / denominator;

        const RealType gradient = weight * g;

        const RealType hessian =
            (IteratorType::isIndicator) ?
            	gradient * (static_cast<RealType>(1.0) - g) :
                weight * (numerator2 / denominator - g * g);

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }

	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
        return std::exp(xBeta);
    }

	RealType getOffsExpXBeta(const RealType* offs, RealType xBeta, RealType y, int k) {
		return std::exp(xBeta);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType denom) {
		return ni * std::log(denom);
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	RealType predictEstimate(RealType xBeta){
		return xBeta;
	}

};

template <typename RealType>
struct LogisticRegression : public Storage<RealType>, IndependentData, GLMProjection, Logistic<RealType>, FixedPid,
	NoFixedLikelihoodTerms {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    LogisticRegression(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

	const static bool precomputeHessian = false;

	static RealType getDenomNullValue () { return static_cast<RealType>(1); }

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(1);
	}

	RealType setIndependentDenominator(RealType expXBeta) {
	    return static_cast<RealType>(1) + expXBeta;
	}

	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
        return std::exp(xBeta);
    }

	RealType getOffsExpXBeta(const RealType* offs, RealType xBeta, RealType y, int k) {
		return std::exp(xBeta);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType denom) {
	    return ni * std::log(denom);
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	RealType predictEstimate(RealType xBeta){
	    RealType t = std::exp(xBeta);
		return t / (t + static_cast<RealType>(1));
	}

	using Storage<RealType>::offsExpXBeta;
	using Storage<RealType>::denomPid;
	using Storage<RealType>::hKWeight;

	template <class IteratorType, class Weights, typename Index>
	static void incrementGradientAndHessian2(
	        RealType& gradient, RealType& hessian,
	        const IteratorType& it,
	        const Index index) {

	    const auto i = it.index();

	    const auto numerator1 = it.multiple(offsExpXBeta[i], index); // expXBeta * x
	    const auto denominator = denomPid[i];

	    const auto g = numerator1 / denominator;
	    if (Weights::isWeighted) {
	        gradient += hKWeight[i] * g;
	    } else {
	        gradient += g;
	    }

	    if (IteratorType::isIndicator) {
	        const auto h = g * (RealType(1) - g);
	        if (Weights::isWeighted) {
	            hessian += hKWeight[i] * h;
	        } else {
	            hessian += h;
	        }
	    } else {
	        const auto numerator2 = it.multiple(numerator1, index); // expXBeta * x * x

	        const auto h = (numerator2 / denominator - g * g);
	        if (Weights::isWeighted) {
	            hessian += hKWeight[i] * h;
	        } else {
	            hessian += h;
	        }
	    }
	}
};

template <typename RealType>
struct CoxProportionalHazards : public Storage<RealType>, OrderedData, GLMProjection, SortedPid, NoFixedLikelihoodTerms, Survival<RealType> {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    CoxProportionalHazards(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

	const static bool precomputeHessian = false;

	static RealType getDenomNullValue () { return static_cast<RealType>(0); }

	RealType setIndependentDenominator(RealType expXBeta) {
        return expXBeta;
    }

    bool resetAccumulators(int* pid, int k, int currentPid) { return false; } // No stratification

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(yi);
	}

	template <class IteratorType, class Weights>
	static void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			RealType* gradient, RealType* hessian,
			RealType numer, RealType numer2, RealType denom,
			RealType nEvents,
			RealType x, RealType xBeta, RealType y) {

		const RealType t = numer / denom;
		const RealType g = nEvents * t; // Always use weights (not censured indicator)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<RealType>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

        throw new std::logic_error("cox model not yet support");

    	const RealType g = numerator / denominator;

        const RealType gradient =
            (WeightOperationType::isWeighted) ? weight * g : g;

        const RealType hessian =
            (IteratorType::isIndicator) ?
                (WeightOperationType::isWeighted) ?
                    weight * g * (static_cast<RealType>(1.0) - g) :
                    g * (static_cast<RealType>(1.0) - g)
                :
                (WeightOperationType::isWeighted) ?
                    weight * (numerator2 / denominator - g * g) :
                    (numerator2 / denominator - g * g);

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }

	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
        return std::exp(xBeta);
    }

	RealType getOffsExpXBeta(const RealType* offs, RealType xBeta, RealType y, int k) {
		return std::exp(xBeta);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType accDenom) {
		return ni*std::log(accDenom);
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    return weight == static_cast<RealType>(0) ? static_cast<RealType>(0) :
	        y * weight * (xBeta - std::log(denominator));
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
			const int* groups, int i) {
		return weighti == static_cast<RealType>(0) ? static_cast<RealType>(0) :
		    ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	RealType predictEstimate(RealType xBeta){
		return xBeta;
	}
};

template <typename RealType>
struct StratifiedCoxProportionalHazards : public CoxProportionalHazards<RealType> {
public:
    bool resetAccumulators(int* pid, int k, int currentPid) {
        return pid[k] != currentPid;
    }
};

template <typename RealType>
struct BreslowTiedCoxProportionalHazards : public Storage<RealType>, OrderedWithTiesData, GLMProjection, SortedPid, NoFixedLikelihoodTerms, Survival<RealType> {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    BreslowTiedCoxProportionalHazards(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

	const static bool precomputeHessian = false;

	static RealType getDenomNullValue () { return static_cast<RealType>(0); }

    bool resetAccumulators(int* pid, int k, int currentPid) {
        return pid[k] != currentPid;
    }

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(yi);
	}

	template <class IteratorType, class Weights>
	static void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			RealType* gradient, RealType* hessian,
			RealType numer, RealType numer2, RealType denom,
			RealType nEvents,
			RealType x, RealType xBeta, RealType y) {

		const RealType t = numer / denom;
		const RealType g = nEvents * t; // Always use weights (not censured indicator)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<RealType>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

// 	    std::cout << "TODO" << std::endl;

        throw new std::logic_error("breslow cox model not yet support");

    	const RealType g = numerator / denominator;

        const RealType gradient =
            (WeightOperationType::isWeighted) ? weight * g : g;

        const RealType hessian =
            (IteratorType::isIndicator) ?
                (WeightOperationType::isWeighted) ?
                    weight * g * (static_cast<RealType>(1.0) - g) :
                    g * (static_cast<RealType>(1.0) - g)
                :
                (WeightOperationType::isWeighted) ?
                    weight * (numerator2 / denominator - g * g) :
                    (numerator2 / denominator - g * g);

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }

	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
        return std::exp(xBeta);
    }

	RealType getOffsExpXBeta(const RealType* offs, RealType xBeta, RealType y, int k) {
		return std::exp(xBeta);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType accDenom) {
		return ni * std::log(accDenom);
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    return weight == static_cast<RealType>(0) ? static_cast<RealType>(0) :
	        y * weight * (xBeta - std::log(denominator));
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
			const int* groups, int i) {
		return weighti == 0.0 ? 0.0 :
		    ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	RealType predictEstimate(RealType xBeta){
		return xBeta;
	}
};

template <typename RealType>
struct LeastSquares : public Storage<RealType>, IndependentData, FixedPid, NoFixedLikelihoodTerms {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    LeastSquares(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

    using Storage<RealType>::hXBeta;

	const static bool precomputeGradient = false; // XjY

	const static bool precomputeHessian = true; // XjX

	const static bool likelihoodHasDenominator = false;

	const static bool hasTwoNumeratorTerms = false;

	const static bool exactCLR = false;

	static RealType getDenomNullValue () { return static_cast<RealType>(0); }

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(1);
	}

	RealType logLikeNumeratorContrib(RealType yi, RealType xBetai) {
	    const RealType residual = yi - xBetai;
		return - (residual * residual);
	}

	template <class XType>
	static RealType gradientNumeratorContrib(XType x, RealType predictor, RealType xBeta, RealType y) {
			return static_cast<RealType>(2) * (xBeta - y) * x;
	}

	template <class XType>
	static RealType gradientNumerator2Contrib(XType x, RealType predictor) {
        throw new std::logic_error("Not model-specific");
		return static_cast<RealType>(0);
	}

	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			RealType* information,
			RealType predictor,
			RealType numer, RealType numer2, RealType denom,
			RealType weight,
			RealType x, RealType xBeta, RealType y) {
		*information += weight * it.value();
	}

	template <class IteratorType, class Weights>
	static void incrementGradientAndHessian(
			const IteratorType& it,
			const Weights& w,
			RealType* gradient, RealType* hessian,
			RealType numer, RealType numer2, RealType denom, RealType weight,
			RealType x, RealType xBeta, RealType y
			) {
		// Reduce contribution here
		if (Weights::isWeighted) {
			*gradient += weight * numer;
		} else {
			*gradient += numer;
		}
	}

	template <class IteratorType, class Weights>
	inline void incrementMMGradientAndHessian(
	        RealType& gradient, RealType& hessian,
	        RealType expXBeta, RealType denominator,
	        RealType weight, RealType x, RealType xBeta,
	        RealType y, RealType norm) {

	    throw new std::logic_error("Not model-specific");
	}

	template <class IteratorType, class WeightOperationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

 	    const RealType gradient = WeightOperationType::isWeighted ? weight * numerator : numerator;
        return { lhs.real() + gradient, lhs.imag() };
    }

	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
        throw new std::logic_error("Not model-specific");
		return static_cast<RealType>(0);
    }

	RealType getOffsExpXBeta(const  RealType* offs, RealType xBeta, RealType y, int k) {
        throw new std::logic_error("Not model-specific");
		return static_cast<RealType>(0);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType denom) {
		return std::log(denom);
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    const RealType residual = y - xBeta;
	    return - (residual * residual * weight);
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
			const int* groups, int i) {
	    RealType residual = ji - xBetai;
		return - (residual * residual * weighti);
	}

	RealType predictEstimate(RealType xBeta){
		return xBeta;
	}
};

template <typename RealType>
struct PoissonRegression : public Storage<RealType>, IndependentData, GLMProjection, FixedPid {
public:
    typedef typename Storage<RealType>::RealVector RealVector;
    PoissonRegression(const RealVector& y, const RealVector& offs)
        : Storage<RealType>(y, offs) { }

	const static bool precomputeHessian = false; // XjX

	const static bool likelihoodHasFixedTerms = true;

	static RealType getDenomNullValue () { return static_cast<RealType>(0); }

	RealType observationCount(RealType yi) {
		return static_cast<RealType>(1);
	}

	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			RealType* information,
			RealType predictor,
			RealType numer, RealType numer2, RealType denom,
			RealType weight,
			RealType x, RealType xBeta, RealType y) {
		*information += weight * predictor * it.value();
	}

	template <class IteratorType, class Weights>
	static void incrementGradientAndHessian(
		const IteratorType& it,
		const Weights& w,
		RealType* gradient, RealType* hessian,
		RealType numer, RealType numer2, RealType denom, RealType weight,
		RealType x, RealType xBeta, RealType y
		) {
	    // Reduce contribution here
	    if (IteratorType::isIndicator) {
	        if (Weights::isWeighted) {
	            const RealType value = weight * numer;
	            *gradient += value;
	            *hessian += value;
	        } else {
	            *gradient += numer;
	            *hessian += numer;
	        }
	    } else {
	        if (Weights::isWeighted) {
	            *gradient += weight * numer;
	            *hessian += weight * numer2;
	        } else {
	            *gradient += numer;
	            *hessian += numer2;
	        }
	    }
			// // Reduce contribution here
			// if (IteratorType::isIndicator) {
			// 		*gradient += numer;
			// 		*hessian += numer;
			// } else {
			// 		*gradient += numer;
			// 		*hessian += numer2;
			// }
	}

	template <class IteratorType, class Weights>
	inline void incrementMMGradientAndHessian(
	        RealType& gradient, RealType& hessian,
	        RealType expXBeta, RealType denominator,
	        RealType weight, RealType x, RealType xBeta, RealType y, RealType norm) {

	    throw new std::logic_error("Not model-specific");
	}

	template <class IteratorType, class WeightOperationType>
	static inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

        RealType gradient = numerator;
        RealType hessian = (IteratorType::isIndicator) ? numerator : numerator2;

        if (WeightOperationType::isWeighted) {
            gradient *= weight;
            hessian *= weight;
        }

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }


	RealType getOffsExpXBeta(const RealType offs, const RealType xBeta) {
		return std::exp(xBeta);
	}

	RealType getOffsExpXBeta(const RealType* offs, RealType xBeta, RealType y, int k) {
		return std::exp(xBeta);
	}

	RealType logLikeDenominatorContrib(RealType ni, RealType denom) {
		return ni * denom;
	}

	RealType logPredLikeContrib(RealType y, RealType weight, RealType xBeta, RealType denominator) {
	    return (y *  xBeta - std::exp(xBeta)) * weight;
	}

	RealType logPredLikeContrib(RealType ji, RealType weighti, RealType xBetai, const RealType* denoms,
		const int* groups, int i) {
			return (ji*xBetai - std::exp(xBetai))*weighti;
	}

	RealType predictEstimate(RealType xBeta){
		return std::exp(xBeta);
	}

	RealType logLikeFixedTermsContrib(RealType yi, RealType offseti, RealType logoffseti) {
	    RealType logLikeFixedTerm = 0.0;
		for(int i = 2; i <= (int)yi; i++)
			logLikeFixedTerm += -log(static_cast<RealType>(i));
		return logLikeFixedTerm;
	}

};

struct OneValue { };

template <class T>
inline T operator*(const T& lhs, const OneValue& rhs) { return lhs; }

template <class IteratorType, class RealType, int index>
struct TupleXGetterNew {

    typedef RealType ReturnType;

    template <class TupleType>
    inline ReturnType operator()(TupleType& tuple) const {
        return boost::get<index>(tuple);
    }
};

template <class RealType, int index>
struct TupleXGetterNew<IndicatorIterator<RealType>, RealType, index> {

    typedef OneValue ReturnType;

    template <class TupleType>
    inline ReturnType operator()(TupleType& tuple) const {
        return OneValue();
    }
};

template <class RealType, int index>
struct TupleXGetterNew<InterceptIterator<RealType>, RealType, index> {

    typedef OneValue ReturnType;

    template <class TupleType>
    inline ReturnType operator()(TupleType& tuple) const {
        return OneValue();
    }
};

template <class BaseModel, class IteratorType, class RealType>
struct TestNumeratorKernel {

    template <class NumeratorType, class TupleType>
    NumeratorType operator()(const NumeratorType lhs, const TupleType tuple) {

        const auto expXBeta = boost::get<0>(tuple);
        const auto x = getX(tuple); //boost::get<1>(tuple);

        return {
            lhs.first + BaseModel::gradientNumeratorContrib(x, expXBeta, 0.0, 0.0),
            (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) ?
            lhs.second +  BaseModel::gradientNumerator2Contrib(x, expXBeta) :
            0.0
        };
    }

    private:	 // TODO Code duplication; remove
        template <class TupleType>
        inline auto getX(TupleType tuple) const -> typename TupleXGetterNew<IteratorType, RealType, 5>::ReturnType {
            return TupleXGetterNew<IteratorType, RealType, 1>()(tuple);
        }

};

template <class BaseModel, class IteratorType, class WeightType, class RealType>
struct TestGradientKernel {

    template <class GradientType, class NumeratorType, class TupleType>
    GradientType operator()(const GradientType lhs, const NumeratorType numerator, const TupleType tuple) {
        const auto denominator = boost::get<0>(tuple);
        const auto weight = boost::get<1>(tuple);

        // std::cerr << "N n1: " << numerator.first << " n2: " << numerator.second
        //           << " d: " << denominator << " w: " << weight <<  std::endl;

        return BaseModel::template incrementGradientAndHessian<IteratorType,
                                                               WeightType>(
                                                                       lhs,
                                                                       numerator.first, numerator.second,
                                                                       denominator, weight, 0.0, 0.0
                                                               );
    }
};

} // namespace

#include "ModelSpecifics.hpp"

#endif /* MODELSPECIFICS_H_ */
