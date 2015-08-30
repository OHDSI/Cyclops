/*
 * ModelSpecificsGPU.h
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */


#ifndef MODELSPECIFICSGPU_H_
#define MODELSPECIFICSGPU_H_

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
#include "vexcl/vexcl.hpp"
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


// struct ParallelInfo { };
//
// struct SerialOnly { };

class SparseIterator; // forward declaration

template <class BaseModel, typename WeightType>
class ModelSpecificsGPU : public AbstractModelSpecifics, BaseModel {
public:
    ModelSpecificsGPU(const ModelData& input);

	virtual ~ModelSpecificsGPU();

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


    std::vector< std::complex<double> > gradientandhessian;

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


} // namespace

#include "ModelSpecificsGPU.hpp"

#endif /* MODELSPECIFICSGPU_H_ */

