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

// #define CYCLOPS_DEBUG_TIMING
// #define CYCLOPS_DEBUG_TIMING_LOW

#ifdef CYCLOPS_DEBUG_TIMING
    #include "Timing.h"
#endif

#include <type_traits>
#include <iterator>
#include <boost/tuple/tuple.hpp>
//#include <boost/iterator/zip_iterator.hpp>
#include <boost/range/iterator_range.hpp>

#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include "AbstractModelSpecifics.h"
#include "Iterators.h"
#include "ParallelLoops.h"

namespace bsccs {

#if 0

// http://liveworkspace.org/code/d52cf97bc56f5526292615659ea110c0
// helper namespace to ensure correct functionality
namespace aux{
namespace adl{
using std::begin;
using std::end;

template<class T>
auto do_begin(T& v) -> decltype(begin(v));

template<class T>
auto do_end(T& v) -> decltype(end(v));

template<class T>
T* do_begin(T* v);

template<class T>
T* do_end(T* v);

} // adl::

template<class... Its>
using zipper_it = boost::zip_iterator<boost::tuple<Its...>>;

template<class... Its>
using zipper_range = boost::iterator_range<zipper_it<Its...>>;

template<class T>
T const& as_const(T const& v){ return v; }

// we don't want temporary containers
// these are helpers to ensure that
template<class Head, class... Tail>
struct any_of
  : std::integral_constant<bool, Head::value || any_of<Tail...>::value>{};

template<class Head>
struct any_of<Head> : std::integral_constant<bool, Head::value>{};

template<class C>
struct not_ : std::integral_constant<bool, !C::value>{};

template<class... Conts>
struct any_temporary : any_of<not_<std::is_reference<Conts>>...>{};
} // aux::

template <class T>
T* begin(T* v) { return v; }

template <class T>
T* end(T* v) { return v; }

template<class... Conts>
auto zip_begin(Conts&&... conts)
  -> aux::zipper_it<decltype(aux::adl::do_begin(conts))...>
{
  static_assert(!aux::any_temporary<Conts...>::value,
      "One or more temporary containers passed to zip_begin");
  using std::begin;
  return {boost::make_tuple(begin(conts)...)};
}

template <class... Conts>
auto zipper(Conts&&... conts) -> aux::zipper_it<Conts...> {
    return { boost::make_tuple(conts...) };
}

template<class... Conts>
auto zip_end(Conts&&... conts)
  -> aux::zipper_it<decltype(aux::adl::do_end(conts))...>
{
  static_assert(!aux::any_temporary<Conts...>::value,
      "One or more temporary containers passed to zip_end");
  using std::end;
  return {boost::make_tuple(end(conts)...)};
}

template<class... Conts>
auto zip_range(Conts&&... conts)
  -> boost::iterator_range<decltype(zip_begin(conts...))>
{
  static_assert(!aux::any_temporary<Conts...>::value,
      "One or more temporary containers passed to zip_range");
  return {zip_begin(conts...), zip_end(conts...)};
}

// for const access
template<class... Conts>
auto zip_cbegin(Conts&&... conts)
  -> decltype(zip_begin(aux::as_const(conts)...))
{
  static_assert(!aux::any_temporary<Conts...>::value,
      "One or more temporary containers passed to zip_cbegin");
  using std::begin;
  return zip_begin(aux::as_const(conts)...);
}

template<class... Conts>
auto zip_cend(Conts&&... conts)
  -> decltype(zip_end(aux::as_const(conts)...))
{
  static_assert(!aux::any_temporary<Conts...>::value,
      "One or more temporary containers passed to zip_cend");
  using std::end;
  return zip_end(aux::as_const(conts)...);
}

template<class... Conts>
auto zip_crange(Conts&&... conts)
  -> decltype(zip_range(aux::as_const(conts)...))
{
  static_assert(!aux::any_temporary<Conts...>::value,
      "One or more temporary containers passed to zip_crange");
  return zip_range(aux::as_const(conts)...);
}

#endif

struct OneValue { };

template <class T>
inline T operator*(const T& lhs, const OneValue& rhs) { return lhs; }

inline std::ostream& operator<<(std::ostream& stream, const OneValue& rhs) {
    stream << "1";
    return stream;
}

// struct ParallelInfo { };
//
// struct SerialOnly { };

class SparseIterator; // forward declaration

template <class BaseModel, typename WeightType>
class ModelSpecifics : public AbstractModelSpecifics, BaseModel {
public:
	ModelSpecifics(const ModelData& input);

	virtual ~ModelSpecifics();

	void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian,  bool useWeights);

	AbstractModelSpecifics* clone() const;

protected:
	void computeNumeratorForGradient(int index);

	void computeFisherInformation(int indexOne, int indexTwo, double *oinfo, bool useWeights);

	void updateXBeta(real realDelta, int index, bool useWeights);

	void computeRemainingStatistics(bool useWeights);

	void computeAccumlatedNumerator(bool useWeights);

	void computeAccumlatedDenominator(bool useWeights);

	void computeFixedTermsInLogLikelihood(bool useCrossValidation);

	void computeFixedTermsInGradientAndHessian(bool useCrossValidation);

	double getLogLikelihood(bool useCrossValidation);

	double getPredictiveLogLikelihood(real* weights);

	void getPredictiveEstimates(real* y, real* weights);

	bool allocateXjY(void);

	bool allocateXjX(void);

	bool allocateNtoKIndices(void);

	bool sortPid(void);

	void setWeights(real* inWeights, bool useCrossValidation);

	void doSortPid(bool useCrossValidation);

	bool initializeAccumulationVectors(void);

	bool hasResetableAccumulators(void);

	void printTiming(void);

private:
	template <class IteratorType, class Weights>
	void computeGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian, Weights w);

	template <class IteratorType>
	void incrementNumeratorForGradientImpl(int index);

	template <class IteratorType>
	void updateXBetaImpl(real delta, int index, bool useWeights);

	template <class OutType, class InType>
	void incrementByGroup(OutType* values, int* groups, int k, InType inc) {
		values[BaseModel::getGroup(groups, k)] += inc; // TODO delegate to BaseModel (different in tied-models)
	}

	template <typename IteratorTypeOne, class Weights>
	void dispatchFisherInformation(int indexOne, int indexTwo, double *oinfo, Weights w);

	template <class IteratorTypeOne, class IteratorTypeTwo, class Weights>
	void computeFisherInformationImpl(int indexOne, int indexTwo, double *oinfo, Weights w);

	template<class IteratorType>
	SparseIterator getSubjectSpecificHessianIterator(int index);

	void computeXjY(bool useCrossValidation);

	void computeXjX(bool useCrossValidation);

	void computeNtoKIndices(bool useCrossValidation);

	std::vector<WeightType> hNWeight;
	std::vector<WeightType> hKWeight;

//	std::vector<int> nPid;
//	std::vector<real> nY;
	std::vector<int> hNtoK;

	struct WeightedOperation {
		const static bool isWeighted = true;
	} weighted;

	struct UnweightedOperation {
		const static bool isWeighted = false;
	} unweighted;

	ParallelInfo info;

//	C11ThreadPool threadPool;

#ifdef CYCLOPS_DEBUG_TIMING
//	std::vector<double> duration;
	std::map<std::string,long long> duration;
#endif

};

template <typename WeightType>
class CompareSurvivalTuples {
	bool useCrossValidation;
	std::vector<WeightType>& weight;
	const std::vector<real>& z;
public:
	CompareSurvivalTuples(bool _ucv,
			std::vector<WeightType>& _weight,
			const std::vector<real>& _z)
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

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
   	throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}
};

template <class IteratorType, class RealType>
struct TupleXGetter {
// 	using XTuple = typename IteratorType::XTuple;
// 	using ReturnType = RealType;
    typedef typename IteratorType::XTuple XTuple;
    typedef RealType ReturnType;

	inline ReturnType operator()(XTuple& tuple) const {
		return boost::get<1>(tuple);
	}
};


template <class RealType>
struct TupleXGetter<InterceptIterator, RealType> {
// 	using XTuple = IndicatorIterator::XTuple;
// 	using ReturnType = OneValue;
    typedef typename InterceptIterator::XTuple XTuple;
    typedef OneValue ReturnType;

	inline ReturnType operator()(XTuple& tuple) const {
		return OneValue();
	}
};

template <class RealType>
struct TupleXGetter<IndicatorIterator, RealType> {
// 	using XTuple = IndicatorIterator::XTuple;
// 	using ReturnType = OneValue;
    typedef typename IndicatorIterator::XTuple XTuple;
    typedef OneValue ReturnType;

// 	inline RealType operator()(XTuple& tuple) const {
// 		return static_cast<RealType>(1.0);
// 	}

	inline ReturnType operator()(XTuple& tuple) const {
		return OneValue();
	}
};

template <class IteratorType, class RealType, int index>
struct TupleXGetterNew {

    typedef RealType ReturnType;

    template <class TupleType>
    inline ReturnType operator()(TupleType& tuple) const {
        return boost::get<index>(tuple);
    }
};

template <class RealType, int index>
struct TupleXGetterNew<IndicatorIterator, RealType, index> {

    typedef OneValue ReturnType;

    template <class TupleType>
    inline ReturnType operator()(TupleType& tuple) const {
        return OneValue();
    }
};

template <class RealType, int index>
struct TupleXGetterNew<InterceptIterator, RealType, index> {

    typedef OneValue ReturnType;

    template <class TupleType>
    inline ReturnType operator()(TupleType& tuple) const {
        return OneValue();
    }
};

// template <class RealType>
// std::pair<RealType, RealType> operator+(
//         const std::pair<RealType, RealType>& lhs,
//         const std::pair<RealType, RealType>& rhs) {
//
//     return { lhs.first + rhs.first, lhs.second + rhs.second };
// }

// template <typename T>
// using Fraction = std::complex<T>; // gcc 4.6.3 does not support template aliases
#define Fraction std::complex
// template <typename T>
// struct Fraction {
//     typedef typename std::complex<T> type;
// };

template <class BaseModel, class RealType>
struct TestPredLikeKernel : private BaseModel {

    template <class Tuple>
    RealType operator()(const RealType lhs, const Tuple tuple) {
        const auto y = boost::get<0>(tuple);
        const auto xBeta = boost::get<1>(tuple);
        const auto denominator = boost::get<2>(tuple);
        const auto weight = boost::get<3>(tuple);

        return lhs + BaseModel::logPredLikeContrib(y, weight, xBeta, denominator);
    }
};

template <class BaseModel, class RealType, class IntType>
struct PredLikeKernel : private BaseModel {

    PredLikeKernel(const RealType* y, const RealType* weights, const RealType* xBeta,
            const RealType* denominator, const IntType* pid) : y(y), weights(weights),
            xBeta(xBeta), denominator(denominator), pid(pid) { }

    RealType operator()(const RealType lhs, const IntType i) {
        return lhs + BaseModel::logPredLikeContrib(y[i], weights[i], xBeta[i], denominator, pid, i);
    }

private:
    const RealType* y;
    const RealType* weights;
    const RealType* xBeta;
    const RealType* denominator;
    const IntType* pid;
};

// template <class BaseModel, class RealType, class IntType, bool weighted>
// struct AccumulateLikeNumeratorKernel : private BaseModel {
//
//     AccumulateLikeNumeratorKernel(const RealType* y, const RealType* xBeta, const RealType* kWeight)
//             : y(y), xBeta(xBeta), kWeight(kWeight) { }
//
//     RealType operator()(const RealType lhs, const IntType i) {
//         RealType rhs = BaseModel::logLikeNumeratorContrib(y[i], xBeta[i]);
//         if (weighted) {
//             rhs *= kWeight[i];
//         }
//         return lhs + rhs;
//     }
//
// protected:
//     const RealType* y;
//     const RealType* xBeta;
//     const RealType* kWeight;
// };


template <class BaseModel, class RealType, bool weighted>
struct TestAccumulateLikeNumeratorKernel : private BaseModel {

	template <class Tuple>
    RealType operator()(const RealType lhs, const Tuple tuple) {
    	const auto y = boost::get<0>(tuple);
    	const auto xBeta = boost::get<1>(tuple);

        RealType rhs = BaseModel::logLikeNumeratorContrib(y, xBeta);
        if (weighted) {
        	const auto weight = boost::get<2>(tuple);
            rhs *= weight;
        }
        return lhs + rhs;
    }
};

template <class BaseModel, class RealType>
struct TestAccumulateLikeDenominatorKernel : private BaseModel {

	template <class Tuple>
    RealType operator()(const RealType lhs, const Tuple tuple) {
		const auto denominator = boost::get<0>(tuple);
		const auto weight = boost::get<1>(tuple);

        return lhs + BaseModel::logLikeDenominatorContrib(weight, denominator);
    }

protected:
    const RealType* nWeight;
    const RealType* denominator;
};


template <class BaseModel, class RealType, class IntType>
struct AccumulateLikeDenominatorKernel : private BaseModel {

    AccumulateLikeDenominatorKernel(const RealType* nWeight, const RealType* denominator)
            : nWeight(nWeight), denominator(denominator) { }

    RealType operator()(const RealType lhs, const IntType i) {
        return lhs + BaseModel::logLikeDenominatorContrib(nWeight[i], denominator[i]);
    }

protected:
    const RealType* nWeight;
    const RealType* denominator;
};

template <class BaseModel, class IteratorType, class WeightOperationType,
class RealType, class IntType>
struct TransformAndAccumulateGradientAndHessianKernelIndependent : private BaseModel {

    template <class TupleType>
    Fraction<RealType> operator()(Fraction<RealType>& lhs, TupleType tuple) {

 		const auto x = getX(tuple);
		const auto expXBeta = boost::get<0>(tuple);
		const auto xBeta = boost::get<1>(tuple);
		const auto y = boost::get<2>(tuple);
		const auto denominator = boost::get<3>(tuple);
		const auto weight = boost::get<4>(tuple);

		RealType numerator = BaseModel::gradientNumeratorContrib(x, expXBeta, xBeta, y);
		RealType numerator2 = (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) ?
				BaseModel::gradientNumerator2Contrib(x, expXBeta) :
				static_cast<RealType>(0);

        return BaseModel::template incrementGradientAndHessian<IteratorType, WeightOperationType, RealType>(
            lhs, numerator, numerator2, denominator, weight, xBeta, y);
    }

private:	 // TODO Code duplication; remove
    template <class TupleType>
	inline auto getX(TupleType& tuple) const -> typename TupleXGetterNew<IteratorType, RealType, 5>::ReturnType {
		return TupleXGetterNew<IteratorType, RealType, 5>()(tuple);
	}
};

// template <class BaseModel, class IteratorType, class WeightOperationType,
// class RealType, class IntType>
// struct TransformAndAccumulateGradientAndHessianKernelDependent : private BaseModel {
//
//     template <class TupleType>
//     Fraction<RealType> operator()(Fraction<RealType>& lhs, TupleType tuple) {
//
//  		const auto x = getX(tuple);
// 		const auto expXBeta = boost::get<0>(tuple);
// 		const auto xBeta = boost::get<1>(tuple);
// 		const auto y = boost::get<2>(tuple);
// 		const auto denominator = boost::get<3>(tuple);
// 		const auto weight = boost::get<4>(tuple);
//
// 		RealType numerator = BaseModel::gradientNumeratorContrib(x, expXBeta, xBeta, y);
// 		RealType numerator2 = (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) ?
// 				BaseModel::gradientNumerator2Contrib(x, expXBeta) :
// 				static_cast<RealType>(0);
//
//
//         return BaseModel::template incrementGradientAndHessian<IteratorType, WeightOperationType, RealType>(
//             lhs,
//             numerator, numerator2,
//             denominator, weight, xBeta, y);
//     }
//
// // protected:
// //
// //     RealType* expXBeta;
// //     RealType* xBeta;
// //     const RealType* y;
// //     RealType* denominator;
// //     RealType* weight;
//
//
// private:	 // TODO Code duplication; remove
//     template <class TupleType>
// 	inline auto getX(TupleType& tuple) const -> typename TupleXGetterNew<IteratorType, RealType, 5>::ReturnType {
// 		return TupleXGetterNew<IteratorType, RealType, 5>()(tuple);
// 	}
// };


// template <class BaseModel, class IteratorType, class WeightOperationType,
// class RealType, class IntType>
// struct TransformAndAccumulateGradientAndHessianKernel : private BaseModel {
//
//     typedef typename IteratorType::XTuple XTuple;
//
//      TransformAndAccumulateGradientAndHessianKernel(
//      	      //RealType* _numerator, RealType* _numerator2,
//      	      RealType* _expXBeta, RealType* _xBeta, const RealType* _y,
//             RealType* _denominator, RealType* _weight//, RealType* _xBeta, const RealType* _y
//             )
//             : //numerator(_numerator), numerator2(_numerator2),
//             expXBeta(_expXBeta), xBeta(_xBeta), y(_y),
//             denominator(_denominator),
//               weight(_weight)  { }
//
//     Fraction<RealType> operator()(Fraction<RealType>& lhs, XTuple tuple) {
//
// 		const auto k = getK(tuple);
// 		const auto x = getX(tuple);
//
// 		RealType numerator = BaseModel::gradientNumeratorContrib(x, expXBeta[k], xBeta[k], y[k]);
// 		RealType numerator2 = (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) ?
// 				BaseModel::gradientNumerator2Contrib(x, expXBeta[k]) :
// 				static_cast<RealType>(0);
//
//         return BaseModel::template incrementGradientAndHessian<IteratorType, WeightOperationType, RealType>(
//             lhs,
//             numerator, numerator2,
// //             numerator[i],
// //             numerator2[i],
//             denominator[k], weight[k], xBeta[k], y[k]);
//     }
//
// protected:
// //    RealType* numerator;
// //    RealType* numerator2;
// 	  RealType* expXBeta;
//     RealType* xBeta;
//     const RealType* y;
//     RealType* denominator;
//     RealType* weight;
//
//
// private:	 // TODO Code duplication; remove
// 	inline auto getX(XTuple& tuple) const -> typename TupleXGetter<IteratorType, RealType>::ReturnType {
// 		return TupleXGetter<IteratorType, RealType>()(tuple);
// 	}
//
// 	inline IntType getK(XTuple& tuple) const {
// 		return boost::get<0>(tuple);
// 	}
// };


template <class BaseModel, class IteratorType, class RealType>
struct TestNumeratorKernel : private BaseModel {

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
struct TestGradientKernel : private BaseModel {

    template <class GradientType, class NumeratorType, class TupleType>
    GradientType operator()(const GradientType lhs, const NumeratorType numerator, const TupleType tuple) {
        const auto denominator = boost::get<0>(tuple);
        const auto weight = boost::get<1>(tuple);

//                 std::cerr << "N n1: " << numerator.first << " n2: " << numerator.second
//                     << " d: " << denominator << " w: " << weight <<  std::endl;

        return BaseModel::template incrementGradientAndHessian<IteratorType,
                            WeightType, RealType>(
                        lhs,
                        numerator.first, numerator.second,
                        denominator, weight, 0.0, 0.0
        );
    }
};

// template <class BaseModel, class IteratorType, class WeightOperationType,
// class RealType, class IntType>
// struct AccumulateGradientAndHessianKernel : private BaseModel {
//
//      AccumulateGradientAndHessianKernel(RealType* _numerator, RealType* _numerator2,
//             RealType* _denominator, RealType* _weight, RealType* _xBeta, const RealType* _y)
//             : numerator(_numerator), numerator2(_numerator2), denominator(_denominator),
//               weight(_weight), xBeta(_xBeta), y(_y) { }
//
//     Fraction<RealType> operator()(Fraction<RealType>& lhs, const IntType& i) {
//
//         std::cerr << "O n1: " << numerator[i] << " n2: " << numerator2[i]
//             << " d: " << denominator[i] << " w: " << weight[i] << std::endl;
//
//         return BaseModel::template incrementGradientAndHessian<IteratorType, WeightOperationType, RealType>(
//             lhs, numerator[i], numerator2[i], denominator[i], weight[i], xBeta[i], y[i]);
//     }
//
// protected:
//     RealType* numerator;
//     RealType* numerator2;
//     RealType* denominator;
//     RealType* weight;
//     RealType* xBeta;
//     const RealType* y;
// };

// template <class BaseModel, class IteratorType, class RealType, class IntType>
// struct ZeroOutNumerator : private BaseModel {
//
// 	ZeroOutNumerator(RealType* _numerator, RealType* _numerator2) :
// 		numerator(_numerator), numerator2(_numerator2) { }
//
// 	void operator()(const IntType& i) {
// 		numerator[i] = static_cast<RealType>(0.0);
// 		if (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) {
// 			numerator2[i] = static_cast<RealType>(0.0);
// 		}
// 	}
//
// private:
// 	RealType* numerator;
// 	RealType* numerator2;
// };


// template <class BaseModel, class IteratorType, class RealType, class IntType>
// struct NumeratorForGradientKernel : private BaseModel {
//
// // 	using XTuple = typename IteratorType::XTuple;
//     typedef typename IteratorType::XTuple XTuple;
//
// 	NumeratorForGradientKernel(RealType* _numerator, RealType* _numerator2,
// 			RealType* _expXBeta, RealType* _xBeta, const RealType* _y, IntType* _pid) : numerator(_numerator),
// 			numerator2(_numerator2), expXBeta(_expXBeta), xBeta(_xBeta), y(_y), pid(_pid) { }
//
// 	void operator()(XTuple tuple) {
//
// 		const auto k = getK(tuple);
// 		const auto x = getX(tuple);
//
// 		numerator[BaseModel::getGroup(pid, k)] += BaseModel::gradientNumeratorContrib(x, expXBeta[k], xBeta[k], y[k]);
// 		if (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) {
// 			numerator2[BaseModel::getGroup(pid, k)] += BaseModel::gradientNumerator2Contrib(x, expXBeta[k]);
// 		}
// 	}
//
// private:
// 	inline auto getX(XTuple& tuple) const -> typename TupleXGetter<IteratorType, RealType>::ReturnType {
// 		return TupleXGetter<IteratorType, RealType>()(tuple);
// 	}
//
// 	inline IntType getK(XTuple& tuple) const {
// 		return boost::get<0>(tuple);
// 	}
//
// 	RealType* numerator;
// 	RealType* numerator2;
// 	RealType* expXBeta;
// 	RealType* xBeta;
// 	const RealType* y;
// 	IntType* pid;
// };


template <class BaseModel, class IteratorType, class RealType>
struct TestUpdateXBetaKernelDependent : private BaseModel {

    TestUpdateXBetaKernelDependent(RealType delta) : delta(delta) { }

    template <class Tuple>
    RealType operator()(RealType lhs, Tuple tuple) {

        const auto x = getX(tuple);
        auto& expXBeta = boost::get<0>(tuple);
        auto& xBeta = boost::get<1>(tuple);
        const auto off = boost::get<3>(tuple);

        xBeta += delta * x; // action

        RealType oldEntry = expXBeta;
        RealType newEntry = expXBeta =
            BaseModel::getOffsExpXBeta(off, xBeta);
//             std::exp(xBeta);

        return lhs + (newEntry - oldEntry);
    }

private:
    RealType delta;

private:	 // TODO Code duplication; remove
    template <class TupleType>
	inline auto getX(TupleType& tuple) const -> typename TupleXGetterNew<IteratorType, RealType, 5>::ReturnType {
		return TupleXGetterNew<IteratorType, RealType, 4>()(tuple);
	}
};

template <class BaseModel, class IteratorType, class RealType>
struct TestUpdateXBetaKernel : private BaseModel {

    TestUpdateXBetaKernel(RealType delta) : delta(delta) { }

    template <class Tuple>
    void operator()(Tuple tuple) {

 		const auto x = getX(tuple);
		auto& expXBeta = boost::get<0>(tuple);
		auto& xBeta = boost::get<1>(tuple);
		auto& denominator = boost::get<2>(tuple);

        xBeta += delta * x;

        if (BaseModel::likelihoodHasDenominator) {
            expXBeta = std::exp(xBeta);
            denominator = BaseModel::getDenomNullValue() + expXBeta;
        }
    }

private:
    RealType delta;

private:	 // TODO Code duplication; remove
    template <class TupleType>
	inline auto getX(TupleType& tuple) const -> typename TupleXGetterNew<IteratorType, RealType, 5>::ReturnType {
		return TupleXGetterNew<IteratorType, RealType, 4>()(tuple);
	}
};

template <class BaseModel, class IteratorType, class RealType, class IntType>
struct UpdateXBetaKernel : private BaseModel {

// 	using XTuple = typename IteratorType::XTuple;
    typedef typename IteratorType::XTuple XTuple;

	UpdateXBetaKernel(RealType _delta,
			RealType* _expXBeta, RealType* _xBeta, const RealType* _y, IntType* _pid,
			RealType* _denominator, const RealType* _offs)
			: delta(_delta), expXBeta(_expXBeta), xBeta(_xBeta), y(_y), pid(_pid),
			  denominator(_denominator), offs(_offs) { }

	void operator()(XTuple tuple) {

		const auto k = getK(tuple);
		const auto x = getX(tuple);

		xBeta[k] += delta * x;

		// Update denominators as well
		if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
			if (true) {	// Old method
				real oldEntry = expXBeta[k];
				real newEntry = expXBeta[k] = BaseModel::getOffsExpXBeta(offs, xBeta[k], y[k], k);
				denominator[BaseModel::getGroup(pid, k)] += (newEntry - oldEntry);
			} else {
			#if 0  // logistic
			    const real t = BaseModel::getOffsExpXBeta(offs, xBeta[k], y[k], k);
			    expXBeta[k] = t;
			    denominator[k] = static_cast<real>(1.0) + t;
			#else
				denominator[k] = expXBeta[k] = BaseModel::getOffsExpXBeta(offs, xBeta[k], y[k], k); // For fast Poisson
            #endif
			}
		}
	}

private:
	// TODO remove code duplication with struct above
	inline auto getX(XTuple& tuple) const -> typename TupleXGetter<IteratorType, RealType>::ReturnType {
		return TupleXGetter<IteratorType, RealType>()(tuple);
	}

	inline IntType getK(XTuple& tuple) const {
		return boost::get<0>(tuple);
	}

	RealType delta;
	RealType* expXBeta;
	RealType* xBeta;
	const RealType* y;
	IntType* pid;
	RealType* denominator;
	const RealType* offs;
};

struct GLMProjection {
public:
	const static bool precomputeGradient = true; // XjY

	const static bool likelihoodHasDenominator = true;

	const static bool hasTwoNumeratorTerms = true;

	template <class XType>
	real gradientNumeratorContrib(XType x, real predictor, real xBeta, real y) {
//		using namespace indicator_sugar;
		return predictor * x;
	}


	real logLikeNumeratorContrib(int yi, real xBetai) {
		return yi * xBetai;
	}

	template <class XType>
	real gradientNumerator2Contrib(XType x, real predictor) {
		return predictor * x * x;
	}
};

template <typename WeightType>
struct Survival {
public: /***/
	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		*information += weight * predictor / denom * it.value();
	}
};

template <typename WeightType>
struct Logistic {
public:
	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		const real g = predictor / denom;
		*information += weight *
				 (predictor / denom - g * g)
//
//				predictor / (denom * denom)
				 * it.value();
	}

	template <class IteratorType, class Weights> // TODO Code duplication with LR
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights w,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {

		const real g = numer / denom;
		if (Weights::isWeighted) {
			*gradient += weight * g;
		} else {
			*gradient += g;
		}
		if (IteratorType::isIndicator) {
			if (Weights::isWeighted) {
				*hessian += weight * g * (static_cast<real>(1.0) - g);
			} else {
				*hessian += g * (static_cast<real>(1.0) - g);
			}
		} else {
			if (Weights::isWeighted) {
				*hessian += weight * (numer2 / denom - g * g); // Bounded by x_j^2
			} else {
				*hessian += (numer2 / denom - g * g); // Bounded by x_j^2
			}
		}
	}

	template <class IteratorType, class WeightOperationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
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

template <typename WeightType>
struct SelfControlledCaseSeries : public GroupedData, GLMProjection, FixedPid, Survival<WeightType> {
public:
	const static bool precomputeHessian = false; // XjX

#define TEST_CONSTANT_SCCS
#ifdef TEST_CONSTANT_SCCS
	const static bool likelihoodHasFixedTerms = true;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
		return yi * std::log(offseti);
	}
#else
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}
#endif

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
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

    real getOffsExpXBeta(const real offs, const real xBeta) {
        return offs * std::exp(xBeta);
    }

	real getOffsExpXBeta(const real* offs, real xBeta, real y, int k) {
		return offs[k] * std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		//do nothing for now
	}

};

template <typename WeightType>
struct ConditionalPoissonRegression : public GroupedData, GLMProjection, FixedPid, Survival<WeightType> {
public:
	const static bool precomputeHessian = false; // XjX

	const static bool likelihoodHasFixedTerms = true;

// 	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
// 		return yi * std::log(offseti);
// 	}

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
		real logLikeFixedTerm = 0.0;
		for(int i = 2; i <= (int)yi; i++)
			logLikeFixedTerm += -log((real)i);
		return logLikeFixedTerm;
	}

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

        // Same as CLR, TODO Remove code-duplication

    	const RealType g = numerator / denominator;
    	const RealType gradient = weight * g; // Always use weights (number of events)

        const RealType hessian =
            (IteratorType::isIndicator) ?
                gradient * (static_cast<RealType>(1.0) - g) :
                weight * (numerator2 / denominator - g * g);

        return { lhs.real() + gradient, lhs.imag() + hessian };
    }

    real getOffsExpXBeta(const real offs, const real xBeta) {
        return std::exp(xBeta);
    }

	real getOffsExpXBeta(const real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		//do nothing for now
	}

};

template <typename WeightType>
struct ConditionalLogisticRegression : public GroupedData, GLMProjection, FixedPid, Survival<WeightType> {
public:
	const static bool precomputeHessian = false; // XjX
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOerationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
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

    real getOffsExpXBeta(const real offs, const real xBeta) {
        return std::exp(xBeta);
    }

	real getOffsExpXBeta(const real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
	    // Do nothing
		//yi = xBeta; // Returns the linear predictor;  ###relative risk
	}

};

template <typename WeightType>
struct TiedConditionalLogisticRegression : public GroupedWithTiesData, GLMProjection, FixedPid, Survival<WeightType> {
public:
	const static bool precomputeGradient = true; // XjY   // TODO Until tied calculations are only used for ties
	const static bool precomputeHessian = false; // XjX
	const static bool likelihoodHasFixedTerms = false;

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (number of events)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
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

    real getOffsExpXBeta(const real offs, const real xBeta) {
        return std::exp(xBeta);
    }

	real getOffsExpXBeta(const real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return ni * std::log(denom);
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
	    // Do nothing
		//yi = xBeta; // Returns the linear predictor;  ###relative risk
	}

};

template <typename WeightType>
struct LogisticRegression : public IndependentData, GLMProjection, Logistic<WeightType>, FixedPid,
	NoFixedLikelihoodTerms {
public:
	const static bool precomputeHessian = false;

// 	const static bool

	static real getDenomNullValue () { return static_cast<real>(1.0); }

	real observationCount(real yi) {
		return static_cast<real>(1);
	}

	real setIndependentDenominator(real expXBeta) {
	    return static_cast<real>(1) + expXBeta;
	}

    real getOffsExpXBeta(const real offs, const real xBeta) {
        return std::exp(xBeta);
    }

	real getOffsExpXBeta(const real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real denom) {
		return std::log(denom);
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    return y * weight * (xBeta - std::log(denominator));
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
			const int* groups, int i) {
		return ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		real t = exp(xBeta);
		yi = t/(t+1);
	}
};

template <typename WeightType>
struct CoxProportionalHazards : public OrderedData, GLMProjection, SortedPid, NoFixedLikelihoodTerms, Survival<WeightType> {
public:
	const static bool precomputeHessian = false;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

    real setIndependentDenominator(real expXBeta) {
        return expXBeta;
    }

    bool resetAccumulators(int* pid, int k, int currentPid) { return false; } // No stratification

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (not censured indicator)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {
//
// 	    std::cout << "TODO" << std::endl;
// 	    std::exit(-1); // cox

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

    real getOffsExpXBeta(const real offs, const real xBeta) {
        return std::exp(xBeta);
    }

	real getOffsExpXBeta(const real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real accDenom) {
		return ni*std::log(accDenom);
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    return weight == 0.0 ? 0.0 :
	        y * weight * (xBeta - std::log(denominator));
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
			const int* groups, int i) {
		return weighti == 0.0 ? 0.0 :
		    ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		// do nothing for now
	}
};

template <typename WeightType>
struct StratifiedCoxProportionalHazards : public CoxProportionalHazards<WeightType> {
public:
    bool resetAccumulators(int* pid, int k, int currentPid) {
        return pid[k] != currentPid;
    }
};

template <typename WeightType>
struct BreslowTiedCoxProportionalHazards : public OrderedWithTiesData, GLMProjection, SortedPid, NoFixedLikelihoodTerms, Survival<WeightType> {
public:
	const static bool precomputeHessian = false;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

    bool resetAccumulators(int* pid, int k, int currentPid) {
        return pid[k] != currentPid;
    }

	real observationCount(real yi) {
		return static_cast<real>(yi);
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			Weights false_signature,
			real* gradient, real* hessian,
			real numer, real numer2, real denom,
			WeightType nEvents,
			real x, real xBeta, real y) {

		const real t = numer / denom;
		const real g = nEvents * t; // Always use weights (not censured indicator)
		*gradient += g;
		if (IteratorType::isIndicator) {
			*hessian += g * (static_cast<real>(1.0) - t);
		} else {
			*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
		}
	}

	template <class IteratorType, class WeightOperationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

// 	    std::cout << "TODO" << std::endl;
// 	    std::exit(-1); // breslow cox

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

    real getOffsExpXBeta(const real offs, const real xBeta) {
        return std::exp(xBeta);
    }

	real getOffsExpXBeta(const real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(WeightType ni, real accDenom) {
		return ni*std::log(accDenom);
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    return weight == 0.0 ? 0.0 :
	        y * weight * (xBeta - std::log(denominator));
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
			const int* groups, int i) {
		return weighti == 0.0 ? 0.0 :
		    ji * weighti * (xBetai - std::log(denoms[getGroup(groups, i)]));
	}

	void predictEstimate(real& yi, real xBeta){
		// do nothing for now
		yi = xBeta;
	}
};

template <typename WeightType>
struct LeastSquares : public IndependentData, FixedPid, NoFixedLikelihoodTerms {
public:
	const static bool precomputeGradient = false; // XjY

	const static bool precomputeHessian = true; // XjX

	const static bool likelihoodHasDenominator = false;

	const static bool hasTwoNumeratorTerms = false;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(1);
	}

	real logLikeNumeratorContrib(real yi, real xBetai) {
		real residual = yi - xBetai;
		return - (residual * residual);
	}

	template <class XType>
	real gradientNumeratorContrib(XType x, real predictor, real xBeta, real y) {
			return static_cast<real>(2) * (xBeta - y) * x;
	}

	template <class XType>
	real gradientNumerator2Contrib(XType x, real predictor) {
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}

// 	struct kernelNumeratorForGradient {
//
// 	    void operator()(GenericIterators::NumeratorForGradientTuple x) {
// 			using boost::get;
//
// 			get<4>(x) += gradientNumeratorContrib (get<0>(x), get<1>(x),
// 			                    get<2>(x), get<3>(x));
// 			get<5>(x) += gradientNumerator2Contrib(get<0>(x), get<1>(x));
// 			// TODO
// 		}
//
// // 	    void operator()(GenericIterators::NumeratorForGradientTupleIndicator x) {
// // 			using boost::get;
// //
// // 			get<3>(x) += gradientNumeratorContrib (1.0, get<0>(x),
// // 			                    get<1>(x), get<2>(x));
// // 			get<4>(x) += gradientNumerator2Contrib(1.0, get<0>(x));
// // 			// TODO
// // 		}
// 	};


	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		*information += weight * it.value();
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
			const IteratorType& it,
			const Weights& w,
			real* gradient, real* hessian,
			real numer, real numer2, real denom, WeightType weight,
			real x, real xBeta, real y
			) {
		// Reduce contribution here
		if (Weights::isWeighted) {
			*gradient += weight * numer;
		} else {
			*gradient += numer;
		}
	}

	template <class IteratorType, class WeightOperationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
	    RealType numerator, RealType numerator2, RealType denominator, RealType weight,
	    RealType xBeta, RealType y) {

 	    const RealType gradient = WeightOperationType::isWeighted ? weight * numerator : numerator;
        return { lhs.real() + gradient, lhs.imag() };
    }

    real getOffsExpXBeta(const real offs, const real xBeta) {
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
    }

	real getOffsExpXBeta(const  real* offs, real xBeta, real y, int k) {
        throw new std::logic_error("Not model-specific");
		return static_cast<real>(0);
	}

	real logLikeDenominatorContrib(int ni, real denom) {
		return std::log(denom);
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    const real residual = y - xBeta;
	    return - (residual * residual * weight);
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
			const int* groups, int i) {
		real residual = ji - xBetai;
		return - (residual * residual * weighti);
	}

	void predictEstimate(real& yi, real xBeta){
		yi = xBeta;
	}
};

template <typename WeightType>
struct PoissonRegression : public IndependentData, GLMProjection, FixedPid {
public:

	const static bool precomputeHessian = false; // XjX

	const static bool likelihoodHasFixedTerms = true;

	static real getDenomNullValue () { return static_cast<real>(0.0); }

	real observationCount(real yi) {
		return static_cast<real>(1);
	}

	template <class IteratorType, class Weights>
	void incrementFisherInformation(
			const IteratorType& it,
			Weights false_signature,
			real* information,
			real predictor,
			real numer, real numer2, real denom,
			WeightType weight,
			real x, real xBeta, real y) {
		*information += weight * predictor * it.value();
	}

	template <class IteratorType, class Weights>
	void incrementGradientAndHessian(
		const IteratorType& it,
		const Weights& w,
		real* gradient, real* hessian,
		real numer, real numer2, real denom, WeightType weight,
		real x, real xBeta, real y
		) {
			// Reduce contribution here
			if (IteratorType::isIndicator) {
				if (Weights::isWeighted) {
					const real value = weight * numer;
					*gradient += value;
					*hessian += value;
				} else {
					*gradient += numer;
					*hessian += numer;
				}
#ifdef DEBUG_POISSON
				std::cerr << (*gradient) << std::endl;
#endif
			} else {
				if (Weights::isWeighted) {
					*gradient += weight * numer;
					*hessian += weight * numer2;
				} else {
					*gradient += numer;
					*hessian += numer2;
				}
#ifdef DEBUG_POISSON
				std::cerr << (*gradient) << std::endl;
#endif
			}
	}

	template <class IteratorType, class WeightOperationType, class RealType>
	inline Fraction<RealType> incrementGradientAndHessian(const Fraction<RealType>& lhs,
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


	real getOffsExpXBeta(const real offs, const real xBeta) {
		return std::exp(xBeta);
	}

	real getOffsExpXBeta(const real* offs, real xBeta, real y, int k) {
		return std::exp(xBeta);
	}

	real logLikeDenominatorContrib(int ni, real denom) {
		return denom;
	}

	real logPredLikeContrib(real y, real weight, real xBeta, real denominator) {
	    return (y *  xBeta - std::exp(xBeta)) * weight;
	}

	real logPredLikeContrib(int ji, real weighti, real xBetai, const real* denoms,
		const int* groups, int i) {
			return (ji*xBetai - exp(xBetai))*weighti;
	}

	void predictEstimate(real& yi, real xBeta){
		yi = exp(xBeta);
	}

	real logLikeFixedTermsContrib(real yi, real offseti, real logoffseti) {
		real logLikeFixedTerm = 0.0;
		for(int i = 2; i <= (int)yi; i++)
			logLikeFixedTerm += -log((real)i);
		return logLikeFixedTerm;
	}

};

} // namespace

#include "ModelSpecifics.hpp"

#endif /* MODELSPECIFICS_H_ */
