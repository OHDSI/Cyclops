/*
 * ModelSpecificsGPU.hpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */


#ifndef ModelSpecificsGPU_HPP_
#define ModelSpecificsGPU_HPP_

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>

#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

#include "ModelSpecificsGPU.h"
#include "Iterators.h"

#include "Recursions.hpp"
#include "ParallelLoops.h"
#include "Ranges.h"

#include "R.h"

#include "Rcpp.h"


#include "boost/compute.hpp"
#include "boost/compute/core.hpp"
#include "boost/compute/algorithm/copy.hpp"
#include "boost/compute/container/vector.hpp"
#include "boost/compute/kernel.hpp"
#include "boost/compute/system.hpp"
#include "boost/compute/program.hpp"
#include "boost/compute/types.hpp"
#include "boost/compute/algorithm/transform.hpp"
#include "boost/compute/function.hpp"
//#include "boost/compute/utility/source.hpp"
#include "Timing.h"
#include "CcdInterface.h"


#ifdef CYCLOPS_DEBUG_TIMING
    #include "Timing.h"
	namespace bsccs {
		const std::string DenseIterator::name = "Den";
		const std::string IndicatorIterator::name = "Ind";
		const std::string SparseIterator::name = "Spa";
		const std::string InterceptIterator::name = "Icp";
	}
#endif

//#define OLD_WAY
//#define NEW_WAY1
#define NEW_WAY2

//#define USE_BIGNUM
#define USE_LONG_DOUBLE

namespace bsccs {

#ifdef USE_BIGNUM
	typedef bigNum DDouble;
#else
	#ifdef USE_LONG_DOUBLE
		typedef long double DDouble;
	#else
		typedef double DDouble;
	#endif
#endif

#if defined(DEBUG_COX) || defined(DEBUG_COX_MIN)
    using std::cerr;
    using std::endl;
#endif

#define NEW_LOOPS

namespace helper {
//
//     auto getRangeAll(const int length) ->
//             boost::iterator_range<
//                 decltype(boost::make_counting_iterator(0))
//             > {
//         return  {
//             boost::make_counting_iterator(0),
//             boost::make_counting_iterator(length)
//         };
//     }
//
//     template <class IteratorTag>
//     auto getRangeDenominator(const IntVectorPtr& mat, const int N, IteratorTag) ->  void {
//     	std::cerr << "Not yet implemented." << std::endl;
//     	std::exit(-1);
//     }
//
//     auto getRangeDenominator(const IntVectorPtr& mat, const int N, DenseTag) ->
//             boost::iterator_range<
//                 decltype(boost::make_counting_iterator(0))
//             > {
//         return {
//             boost::make_counting_iterator(0),
//             boost::make_counting_iterator(N)
//         };
//     }
//
//     auto getRangeDenominator(const IntVectorPtr& mat, const int N, SparseTag) ->
//             boost::iterator_range<
//                 decltype(mat->begin())
//             > {
//         return {
//             std::begin(*mat), std::end(*mat)
//         };
//     }
//
//     auto getRangeDenominator(const IntVectorPtr& mat, const int N, IndicatorTag) ->
//             boost::iterator_range<
//                 decltype(mat->begin())
//             > {
//         return {
//             std::begin(*mat), std::end(*mat)
//         };
//     }
//
//     template <class IteratorTag>
//     auto getRangeNumerator(const IntVectorPtr& mat, const int N, IteratorTag) ->  void {
//     	std::cerr << "Not yet implemented." << std::endl;
//     	std::exit(-1);
//     }
//
//     auto getRangeNumerator(const IntVectorPtr& mat, const int N, DenseTag) ->
//             boost::iterator_range<
//                 decltype(boost::make_counting_iterator(0))
//             > {
//         return {
//             boost::make_counting_iterator(0),
//             boost::make_counting_iterator(N)
//         };
//     }
//
//     auto getRangeNumerator(const IntVectorPtr& mat, const int N, SparseTag) ->
//             boost::iterator_range<
//                 decltype(mat->begin())
//             > {
//         return {
//             std::begin(*mat), std::end(*mat)
//         };
//     }
//
//     auto getRangeNumerator(const IntVectorPtr& mat, const int N, IndicatorTag) ->
//             boost::iterator_range<
//                 decltype(mat->begin())
//             > {
//         return {
//             std::begin(*mat), std::end(*mat)
//         };
//     }
//
//     template <class IteratorTag>
//     auto getRangeCOOX(const CompressedDataMatrix& mat, const int index, IteratorTag) -> void {
//     	std::cerr << "Not yet implemented." << std::endl;
//     	std::exit(-1);
//     }
//
//     auto getRangeCOOX(const CompressedDataMatrix& mat, const int index, DenseTag) ->
//             boost::iterator_range<
//                 boost::zip_iterator<
//                     boost::tuple<
//                         decltype(boost::make_counting_iterator(0)), // TODO Not quite right
//                         decltype(boost::make_counting_iterator(0)),
//                         decltype(begin(mat.getDataVector(index)))
//                     >
//                 >
//             > {
//         auto i = boost::make_counting_iterator(0); // TODO Not quite right
//         auto j = boost::make_counting_iterator(0);
//         auto x = begin(mat.getDataVector(index));
//
//         const size_t K = mat.getNumberOfRows();
//
//         return {
//             boost::make_zip_iterator(
//                 boost::make_tuple(i, j, x)),
//             boost::make_zip_iterator(
//                 boost::make_tuple(i + K, j + K, x + K))
//         };
//     }
//
// 	template <class IteratorTag>
//     auto getRangeX(const CompressedDataMatrix& mat, const int index, IteratorTag) -> void {
//     	std::cerr << "Not yet implemented." << std::endl;
//     	std::exit(-1);
//     }


} // namespace helper


#ifdef opencl
    int loopCount = 0;
    double totalTime = 0.0;
    int vectorLength = 9;
    vex::Context ctx(vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::DoublePrecision);
    vex::vector<double> vex_DataVector0(ctx, vectorLength);  // Device vector.
    vex::vector<double> vex_DataVector1(ctx, vectorLength);  // Device vector.
    vex::vector<double> vex_DataVector2(ctx, vectorLength);  // Device vector.
    vex::vector<double> vex_offsExpXBeta(ctx, vectorLength);  // Device vector.
    vex::vector<double> vex_denominator(ctx, vectorLength);  // Device vector.
    vex::vector<int> vex_length(ctx, 2);  // Device vector.
    std::vector<vex::backend::kernel> kernel;
#endif

template <class BaseModel,typename WeightType>
ModelSpecificsGPU<BaseModel,WeightType>::ModelSpecificsGPU(const ModelData& input)
	: AbstractModelSpecifics(input), BaseModel()//,
//  	threadPool(4,4,1000)
// threadPool(0,0,10)
	{
	// TODO Memory allocation here

#ifdef opencl
    std::vector<int> vectorLengthVector = {vectorLength,vectorLength};
    vex::copy(std::begin(vectorLengthVector), std::end(vectorLengthVector), vex_length.begin()); // only need to do once.
     vex::copy(std::begin(modelData.getDataVectorSTL(0)), std::end(modelData.getDataVectorSTL(0)), vex_DataVector0.begin()); // only need to do once.
     vex::copy(std::begin(modelData.getDataVectorSTL(1)), std::end(modelData.getDataVectorSTL(1)), vex_DataVector1.begin()); // only need to do once.
     vex::copy(std::begin(modelData.getDataVectorSTL(2)), std::end(modelData.getDataVectorSTL(2)), vex_DataVector2.begin()); // only need to do once.
#endif


#ifdef CYCLOPS_DEBUG_TIMING
	auto now = bsccs::chrono::system_clock::now();
	auto now_c = bsccs::chrono::system_clock::to_time_t(now);
	std::cout << std::endl << "Start: " << std::ctime(&now_c) << std::endl;
#endif


}

template <class BaseModel, typename WeightType>
AbstractModelSpecifics* ModelSpecificsGPU<BaseModel,WeightType>::clone() const {
	return new ModelSpecificsGPU<BaseModel,WeightType>(modelData);
}

template <class BaseModel, typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::printTiming() {

#ifdef CYCLOPS_DEBUG_TIMING

	std::cout << std::endl;
	for (auto& d : duration) {
		std::cout << d.first << " " << d.second << std::endl;
	}
	std::cout << "NEW LOOPS" << std::endl;

#endif
}

template <class BaseModel,typename WeightType>
ModelSpecificsGPU<BaseModel,WeightType>::~ModelSpecificsGPU() {
	// TODO Memory release here

#ifdef CYCLOPS_DEBUG_TIMING

    printTiming();

	auto now = bsccs::chrono::system_clock::now();
	auto now_c = bsccs::chrono::system_clock::to_time_t(now);
	std::cout << std::endl << "End:   " << std::ctime(&now_c) << std::endl;
#endif

}

template <class BaseModel,typename WeightType>
bool ModelSpecificsGPU<BaseModel,WeightType>::allocateXjY(void) { return BaseModel::precomputeGradient; }

template <class BaseModel,typename WeightType>
bool ModelSpecificsGPU<BaseModel,WeightType>::allocateXjX(void) { return BaseModel::precomputeHessian; }

template <class BaseModel,typename WeightType>
bool ModelSpecificsGPU<BaseModel,WeightType>::sortPid(void) { return BaseModel::sortPid; }

template <class BaseModel,typename WeightType>
bool ModelSpecificsGPU<BaseModel,WeightType>::initializeAccumulationVectors(void) { return BaseModel::cumulativeGradientAndHessian; }

template <class BaseModel,typename WeightType>
bool ModelSpecificsGPU<BaseModel,WeightType>::allocateNtoKIndices(void) { return BaseModel::hasNtoKIndices; }

template <class BaseModel,typename WeightType>
bool ModelSpecificsGPU<BaseModel,WeightType>::hasResetableAccumulators(void) { return BaseModel::hasResetableAccumulators; }

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::setWeights(real* inWeights, bool useCrossValidation) {
	// Set K weights
	if (hKWeight.size() != K) {
		hKWeight.resize(K);
	}
	if (useCrossValidation) {
		for (size_t k = 0; k < K; ++k) {
			hKWeight[k] = inWeights[k];
		}
	} else {
		std::fill(hKWeight.begin(), hKWeight.end(), static_cast<WeightType>(1));
	}

	if (initializeAccumulationVectors()) {
		setPidForAccumulation(inWeights);
	}

	// Set N weights (these are the same for independent data models
	if (hNWeight.size() < N + 1) { // Add +1 for extra (zero-weight stratum)
		hNWeight.resize(N + 1);
	}

	std::fill(hNWeight.begin(), hNWeight.end(), static_cast<WeightType>(0));
	for (size_t k = 0; k < K; ++k) {
		WeightType event = BaseModel::observationCount(hY[k])*hKWeight[k];
		incrementByGroup(hNWeight.data(), hPid, k, event);
	}

#ifdef DEBUG_COX
	cerr << "Done with set weights" << endl;
#endif

}

template<class BaseModel, typename WeightType>
void ModelSpecificsGPU<BaseModel, WeightType>::computeXjY(bool useCrossValidation) {
	for (size_t j = 0; j < J; ++j) {
		hXjY[j] = 0;

		GenericIterator it(modelData, j);

		if (useCrossValidation) {
			for (; it; ++it) {
				const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
				} else {
					hXjY[j] += it.value() * hY[k] * hKWeight[k];
				}
			}
		} else {
			for (; it; ++it) {
				const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
				} else {
					hXjY[j] += it.value() * hY[k];
				}
			}
		}
#ifdef DEBUG_COX
		cerr << "j: " << j << " = " << hXjY[j]<< endl;
#endif
	}
}

template<class BaseModel, typename WeightType>
void ModelSpecificsGPU<BaseModel, WeightType>::computeXjX(bool useCrossValidation) {
	for (size_t j = 0; j < J; ++j) {
		hXjX[j] = 0;
		GenericIterator it(modelData, j);

		if (useCrossValidation) {
			for (; it; ++it) {
				const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
				} else {
					hXjX[j] += it.value() * it.value() * hKWeight[k];
				}
			}
		} else {
			for (; it; ++it) {
				const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
				} else {
					hXjX[j] += it.value() * it.value();
				}
			}
		}
	}
}

template<class BaseModel, typename WeightType>
void ModelSpecificsGPU<BaseModel, WeightType>::computeNtoKIndices(bool useCrossValidation) {

	hNtoK.resize(N+1);
	int n = 0;
	for (size_t k = 0; k < K;) {
		hNtoK[n] = k;
		int currentPid = hPid[k];
		do {
			++k;
		} while (k < K && currentPid == hPid[k]);
		++n;
	}
	hNtoK[n] = K;
}

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::computeFixedTermsInLogLikelihood(bool useCrossValidation) {
	if(BaseModel::likelihoodHasFixedTerms) {
		logLikelihoodFixedTerm = 0.0;
		if(useCrossValidation) {
			for(size_t i = 0; i < K; i++) {
				logLikelihoodFixedTerm += BaseModel::logLikeFixedTermsContrib(hY[i], hOffs[i], hOffs[i]) * hKWeight[i];
			}
		} else {
			for(size_t i = 0; i < K; i++) {
				logLikelihoodFixedTerm += BaseModel::logLikeFixedTermsContrib(hY[i], hOffs[i], hOffs[i]);
			}
		}
	}
}

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::computeFixedTermsInGradientAndHessian(bool useCrossValidation) {
	if (sortPid()) {
		doSortPid(useCrossValidation);
	}
	if (allocateXjY()) {
		computeXjY(useCrossValidation);
	}
	if (allocateXjX()) {
		computeXjX(useCrossValidation);
	}
	if (allocateNtoKIndices()) {
		computeNtoKIndices(useCrossValidation);
	}
}

template <class BaseModel,typename WeightType>
double ModelSpecificsGPU<BaseModel,WeightType>::getLogLikelihood(bool useCrossValidation) {

#ifdef CYCLOPS_DEBUG_TIMING
	auto start = bsccs::chrono::steady_clock::now();
#endif

//     auto rangeNumerator = helper::getRangeAll(K);
//
//     real logLikelihood = useCrossValidation ?
//     		variants::reduce(
//                 rangeNumerator.begin(), rangeNumerator.end(), static_cast<real>(0.0),
//                 AccumulateLikeNumeratorKernel<BaseModel,real,int,true>(begin(hY), begin(hXBeta), begin(hKWeight)),
//                 SerialOnly()
//     		) :
//     		variants::reduce(
//                 rangeNumerator.begin(), rangeNumerator.end(), static_cast<real>(0.0),
//                 AccumulateLikeNumeratorKernel<BaseModel,real,int,false>(begin(hY), begin(hXBeta), begin(hKWeight)),
//                 SerialOnly()
//     		);

    auto rangeNumerator = helper::getRangeAllNumerators(K, hY, hXBeta, hKWeight);

    real logLikelihood = useCrossValidation ?
    		variants::reduce(
                rangeNumerator.begin(), rangeNumerator.end(), static_cast<real>(0.0),
                TestAccumulateLikeNumeratorKernel<BaseModel,real,true>(),
                SerialOnly()
    		) :
    		variants::reduce(
                rangeNumerator.begin(), rangeNumerator.end(), static_cast<real>(0.0),
                TestAccumulateLikeNumeratorKernel<BaseModel,real,false>(),
                SerialOnly()
    		);

//     std::cerr << logLikelihood << " == " << logLikelihood2 << std::endl;

    if (BaseModel::likelihoodHasDenominator) {

//         auto rangeDenominator = helper::getRangeAll(N);
//
//         auto kernelDenominator = (BaseModel::cumulativeGradientAndHessian) ?
//                 AccumulateLikeDenominatorKernel<BaseModel,real,int>(begin(hNWeight), begin(accDenomPid)) :
//                 AccumulateLikeDenominatorKernel<BaseModel,real,int>(begin(hNWeight), begin(denomPid));
//
//         logLikelihood -= variants::reduce(
//                 rangeDenominator.begin(), rangeDenominator.end(),
//                 static_cast<real>(0.0),
//                 kernelDenominator,
//                 SerialOnly()
//         );

		auto rangeDenominator = (BaseModel::cumulativeGradientAndHessian) ?
				helper::getRangeAllDenominators(N, accDenomPid, hNWeight) :
				helper::getRangeAllDenominators(N, denomPid, hNWeight);

		logLikelihood -= variants::reduce(
				rangeDenominator.begin(), rangeDenominator.end(),
				static_cast<real>(0.0),
				TestAccumulateLikeDenominatorKernel<BaseModel,real>(),
				SerialOnly()
		);

//         std::cerr << logLikelihood << " == " << logLikelihood2 << std::endl;
    }

	if (BaseModel::likelihoodHasFixedTerms) {
		logLikelihood += logLikelihoodFixedTerm;
	}

#ifdef CYCLOPS_DEBUG_TIMING
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compLogLike      "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

	return static_cast<double>(logLikelihood);
}

template <class BaseModel,typename WeightType>
double ModelSpecificsGPU<BaseModel,WeightType>::getPredictiveLogLikelihood(real* weights) {

    std::vector<real> saveKWeight;
	if(BaseModel::cumulativeGradientAndHessian)	{

 		saveKWeight = hKWeight; // make copy

// 		std::vector<int> savedPid = hPidInternal; // make copy
// 		std::vector<int> saveAccReset = accReset; // make copy

		setPidForAccumulation(weights);
		computeRemainingStatistics(true); // compute accDenomPid

    }

	// Compile-time switch for models with / with-out PID (hasIndependentRows)
	auto range = helper::getRangeAllPredictiveLikelihood(K, hY, hXBeta,
		(BaseModel::cumulativeGradientAndHessian) ? accDenomPid : denomPid,
		weights, hPid, std::integral_constant<bool, BaseModel::hasIndependentRows>());

	auto kernel = TestPredLikeKernel<BaseModel,real>();

	real logLikelihood = variants::reduce(
			range.begin(), range.end(), static_cast<real>(0.0),
			kernel,
			SerialOnly()
		);

	if (BaseModel::cumulativeGradientAndHessian) {

// 		hPidInternal = savedPid; // make copy; TODO swap
// 		accReset = saveAccReset; // make copy; TODO swap
		setPidForAccumulation(&saveKWeight[0]);
		computeRemainingStatistics(true);
	}

	return static_cast<double>(logLikelihood);
}   // END OF DIFF

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::getPredictiveEstimates(real* y, real* weights){

	// TODO Check with SM: the following code appears to recompute hXBeta at large expense
//	std::vector<real> xBeta(K,0.0);
//	for(int j = 0; j < J; j++){
//		GenericIterator it(modelData, j);
//		for(; it; ++it){
//			const int k = it.index();
//			xBeta[k] += it.value() * hBeta[j] * weights[k];
//		}
//	}
	if (weights) {
		for (size_t k = 0; k < K; ++k) {
			if (weights[k]) {
				BaseModel::predictEstimate(y[k], hXBeta[k]);
			}
		}
	} else {
		for (size_t k = 0; k < K; ++k) {
			BaseModel::predictEstimate(y[k], hXBeta[k]);
		}
	}
	// TODO How to remove code duplication above?
}

// TODO The following function is an example of a double-dispatch, rewrite without need for virtual function
template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::computeGradientAndHessian(int index, double *ogradient,
		double *ohessian, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	// Run-time dispatch, so virtual call should not effect speed
	if (useWeights) {
		switch (modelData.getFormatType(index)) {
			case INDICATOR :
				computeGradientAndHessianImpl<IndicatorIterator>(index, ogradient, ohessian, weighted);
				break;
			case SPARSE :
				computeGradientAndHessianImpl<SparseIterator>(index, ogradient, ohessian, weighted);
				break;
			case DENSE :
				computeGradientAndHessianImpl<DenseIterator>(index, ogradient, ohessian, weighted);
				break;
			case INTERCEPT :
				computeGradientAndHessianImpl<InterceptIterator>(index, ogradient, ohessian, weighted);
				break;
		}
	} else {
		switch (modelData.getFormatType(index)) {
			case INDICATOR :
				computeGradientAndHessianImpl<IndicatorIterator>(index, ogradient, ohessian, unweighted);
				break;
			case SPARSE :
				computeGradientAndHessianImpl<SparseIterator>(index, ogradient, ohessian, unweighted);
				break;
			case DENSE :
				computeGradientAndHessianImpl<DenseIterator>(index, ogradient, ohessian, unweighted);
				break;
			case INTERCEPT :
				computeGradientAndHessianImpl<InterceptIterator>(index, ogradient, ohessian, unweighted);
				break;
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compGradAndHess  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}



template <class BaseModel,typename WeightType> template <class IteratorType, class Weights>
void ModelSpecificsGPU<BaseModel,WeightType>::computeGradientAndHessianImpl(int index, double *ogradient,
		double *ohessian, Weights w) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif
    std::cout << "GPU HERE" << std::endl;
	real gradient = static_cast<real>(0);
	real hessian = static_cast<real>(0);

	if (BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

    	if (sparseIndices[index] == nullptr || sparseIndices[index]->size() > 0) {

		// TODO
		// x. Fill numerators <- 0
		// x. Compute non-zero numerators
		// x. Segmented scan of numerators
		// x. Transformation/reduction of [begin,end)

		IteratorType it(*(sparseIndices)[index], N);


		real accNumerPid  = static_cast<real>(0);
		real accNumerPid2 = static_cast<real>(0);

// 		const real* data = modelData.getDataVector(index);

        // find start relavent accumulator reset point
        auto reset = begin(accReset);
        while( *reset < it.index() ) {
            ++reset;
        }

		for (; it; ) {
			int i = it.index();

			if (*reset <= i) {
			    accNumerPid  = static_cast<real>(0.0);
			    accNumerPid2 = static_cast<real>(0.0);
			    ++reset;
			}


//			const real x = it.value();

//			const real x = (IteratorType::isIndicator) ? 1.0 :
//				(IteratorType::isSparse) ? *data : data[i];
// 			const real x = 1.0;

			const auto numerator1 = numerPid[i];
			const auto numerator2 = numerPid2[i];

//     		const real numerator1 = BaseModel::gradientNumeratorContrib(x, offsExpXBeta[i], hXBeta[i], hY[i]);
//     		const real numerator2 = BaseModel::gradientNumerator2Contrib(x, offsExpXBeta[i]);

     		accNumerPid += numerator1;
     		accNumerPid2 += numerator2;

#ifdef DEBUG_COX
			cerr << "w: " << i << " " << hNWeight[i] << " " << numerator1 << ":" <<
					accNumerPid << ":" << accNumerPid2 << ":" << accDenomPid[i];
#endif
			// Compile-time delegation
			BaseModel::incrementGradientAndHessian(it,
					w, // Signature-only, for iterator-type specialization
					&gradient, &hessian, accNumerPid, accNumerPid2,
					accDenomPid[i], hNWeight[i], it.value(), hXBeta[i], hY[i]);
					// When function is in-lined, compiler will only use necessary arguments
#ifdef DEBUG_COX
			cerr << " -> g:" << gradient << " h:" << hessian << endl;
#endif
			++it;

			if (IteratorType::isSparse) {
// 				++data;
				const int next = it ? it.index() : N;
				for (++i; i < next; ++i) {
#ifdef DEBUG_COX
			cerr << "q: " << i << " " << hNWeight[i] << " " << 0 << ":" <<
					accNumerPid << ":" << accNumerPid2 << ":" << accDenomPid[i];
#endif
                    if (*reset <= i) {
			            accNumerPid  = static_cast<real>(0.0);
        			    accNumerPid2 = static_cast<real>(0.0);
		        	    ++reset;
                   }

					BaseModel::incrementGradientAndHessian(it,
							w, // Signature-only, for iterator-type specialization
							&gradient, &hessian, accNumerPid, accNumerPid2,
							accDenomPid[i], hNWeight[i], static_cast<real>(0), hXBeta[i], hY[i]);
							// When function is in-lined, compiler will only use necessary arguments
#ifdef DEBUG_COX
			cerr << " -> g:" << gradient << " h:" << hessian << endl;
#endif

				}
			}
		}
		}
	} else if (BaseModel::hasIndependentRows) {
	    auto range = helper::independent::getRangeX(modelData, index,
	    offsExpXBeta, hXBeta, hY, denomPid, hNWeight,
	    typename IteratorType::tag());



	    // -DVEXCL_SHOW_KERNELS to show kernels
	    bool iamstupid = false;
	    double gradienttest = 0.0;
	    double hessiantest = 0.0;


#ifdef opencl
	    if (iamstupid){
/*
	       using boost::compute::uint_;
//            // Boost::compute
//
            boost::compute::device device = boost::compute::system::find_device("Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz");//boost::compute::system::default_device(); //
            std::cout << device.name() << std::endl;
            boost::compute::context context{device};
            boost::compute::command_queue queue{context, device, boost::compute::command_queue::enable_profiling};

           // create vector on device
            boost::compute::vector<double> bcDataVector(K, context);  // Device vector.
            boost::compute::vector<double> bcoffsExpXBeta(K, context); // Device vector.
	        boost::compute::vector<double> bcDenominator(K, context);  // Device vector.
            boost::compute::vector<double> gradientAndHessian(1, context);  // Device vector.
//
//

            // copy from host to device
            std::cout << "Copy to device" << std::endl;
            boost::compute::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), bcoffsExpXBeta.begin(),queue);
            boost::compute::copy(std::begin(modelData.getDataVectorSTL(index)), std::end(modelData.getDataVectorSTL(index)), bcDenominator.begin(),queue);
            //boost::compute::copy(std::begin(denomPid), std::end(denomPid), bcDenominator.begin(),queue);
            boost::compute::copy(std::begin(modelData.getDataVectorSTL(index)), std::end(modelData.getDataVectorSTL(index)), bcDataVector.begin(),queue);
//
//          // create vector on host
            std::vector<double> hostVector(K);

            // Load kernel
            std::cout << "Load kernel" << std::endl;

            const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                        __kernel void transformationReductionKernel(  ulong n,
                                global double * prm_1,
                                global double * prm_2,
                                global double * prm_3,
                                global double * gradientandhessian
                                )
                                {
                                    local double2 * sdata = smem;
                                    size_t tid = get_local_id(0);
                                    size_t block_size = get_local_size(0);
                                    double mySum = 0.0;
                                    for(ulong idx = get_global_id(0); idx < n; idx += get_global_size(0))
                                    {
                                        mySum = mySum + prm_1[idx] + prm_2[idx] + prm_3[idx];
                                    }
                                    sdata[tid] = mySum;
                                    barrier(CLK_LOCAL_MEM_FENCE);
                                    if (block_size >= 1024)
                                    {
                                        if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
                                    if (block_size >= 512)
                                    {
                                        if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
                                    if (block_size >= 256)
                                    {
                                        if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
                                    if (block_size >= 128)
                                    {
                                        if (tid < 64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
                                    if (tid < 32)
                                    {
                                        volatile local double2 * smem = sdata;
                                        if (block_size >= 64) { smem[tid] = mySum = mySum + sdata[tid + 32]; }
                                        if (block_size >= 32) { smem[tid] = mySum = mySum + sdata[tid + 16]; }
                                        if (block_size >= 16) { smem[tid] = mySum = mySum + sdata[tid + 8]; }
                                        if (block_size >= 8) { smem[tid] = mySum = mySum + sdata[tid + 4]; }
                                        if (block_size >= 4) { smem[tid] = mySum = mySum + sdata[tid + 2]; }
                                        if (block_size >= 2) { smem[tid] = mySum = mySum + sdata[tid + 1]; }
                                    }
                                    if (tid == 0) {
                                        gradientandhessian[get_group_id(0)] = smem[tid];
                                    }

                                }
	            );

            std::stringstream kernelSource;
            kernelSource << " __kernel void testKernel(uint n, __global double *offsExpXBeta, __global double * dataVector, __global double * gradientandhessian)\n";
            kernelSource << "{\n const uint block_offset = get_group_id(0);\n";
            kernelSource << " size_t block_size = get_local_size(0);\n const uint lid = get_local_id(0); \n";
            kernelSource << "__local float scratch[block_size]; \n";
            kernelSource << "for(uint idx = get_global_id(0); idx < n; idx += get_global_size(0))\n";
            kernelSource << " {sum = sum + dataVector[idx]+offsExpXBeta[idx]; \n";
            kernelSource << " barrier(CLK_LOCAL_MEM_FENCE); \n";
            kernelSource << " if (block_size >= 1024){if (lid < 512) { sum += scratch[lid + 512]; scratch[lid] = sum; }barrier(CLK_LOCAL_MEM_FENCE);}";
            kernelSource << " if (block_size >= 512){if (lid < 256) { sum += scratch[lid + 256]; scratch[lid] = sum; }barrier(CLK_LOCAL_MEM_FENCE);}";
            kernelSource << " if (block_size >= 256){if (lid < 128) { sum += scratch[lid + 128]; scratch[lid] = sum; }barrier(CLK_LOCAL_MEM_FENCE);}";
            kernelSource << " if (block_size >= 128){if (lid < 64) { sum += scratch[lid + 64]; scratch[lid] = sum; }barrier(CLK_LOCAL_MEM_FENCE);}";
            kernelSource << "if (lid < 32){volatile __local float *lmem = scratch;";
            kernelSource << "if (block_size >= 64) { lmem[lid] = sum = sum + lmem[lid+32]; }";
            kernelSource << "if (block_size >= 32) { lmem[lid] = sum = sum + lmem[lid+16]; }";
            kernelSource << "if (block_size >= 16) { lmem[lid] = sum = sum + lmem[lid+8]; }";
            kernelSource << "if (block_size >= 8) { lmem[lid] = sum = sum + lmem[lid+4]; }";
            kernelSource << "if (block_size >= 4) { lmem[lid] = sum = sum + lmem[lid+2]; }";
            kernelSource << "if (block_size >= 2) { lmem[lid] = sum = sum + lmem[lid+1]; }";
            kernelSource << "}\n";
            kernelSource << "if (lid == 0) {gradientandhessian[get_group_id(0)] = scratch[0];}}";

            std::cout << kernelSource.str() << std::endl;

//

            boost::compute::program testProgram = boost::compute::program::create_with_source(source, context);
            testProgram.build();
            boost::compute::kernel testKernel = boost::compute::kernel(testProgram, "transformationReductionKernel");
//

            testKernel.set_arg(0, uint_(K)); // TODO Must update
            testKernel.set_arg(1, bcoffsExpXBeta); // TODO Must update
            testKernel.set_arg(2, bcDenominator); // TODO Must update
            testKernel.set_arg(3, bcDataVector); // TODO Must update gradientAndHessian
            testKernel.set_arg(4, gradientAndHessian);

            //uint_ vpt = VPT_DEF;
            queue.enqueue_1d_range_kernel(testKernel, 0, 20*hostVector.size(), 1);
            boost::compute::copy(bcoffsExpXBeta.begin(), bcoffsExpXBeta.end(), hostVector.begin(), queue);
            for(int i = 0; i<K; i++){
                std::cout << hostVector[i] << ",";
            }
            std::cout << "" << std::endl;

*/

//
//
//            // copy data back to host
//            boost::compute::copy(bcoffsExpXBeta.begin(), bcoffsExpXBeta.end(), host_vector.begin(), queue);
//            std::cout << host_vector[0] << std::endl;
            //Rcpp::stop("stupid");



            // Some things need to be copied each time
	        vex::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), vex_offsExpXBeta.begin());
            vex::copy(std::begin(denomPid), std::end(denomPid), vex_denominator.begin());

//            struct timeval time1, time2;
//	        gettimeofday(&time1, NULL);

            std::vector< std::complex<double> > gradientandhessian(1);
            vex::vector<cl_double2> vex_gradientandhessian(ctx, 1);  // Device vector.


            // define and place the transformation reduction kernel into the queue
            if (loopCount == 0){
                for(uint d = 0; d < ctx.size(); d++) {
                    kernel.emplace_back(ctx.queue(d),
                    VEX_STRINGIZE_SOURCE(
                        double2 transformSumKernel
                        (
                            double2 prm1,
                            double prm2,
                            double prm3,
                            double prm4
                            )
                            {
                                double2 temp;
                                temp.x = (prm2*prm3)/prm4;
                                temp.y = prm2*(temp.x) - (temp.x)*(temp.x);
                                temp.x += prm1.x;
                                temp.y += prm1.y;
                                return(temp);
                            }

                            double2 sumDouble2
                            (
                                double2 prm1,
                                double2 prm2
                                )
                                {
                                    double2 temp;
                                    temp.x = prm1.x + prm2.x;
                                    temp.y = prm1.y + prm2.y;
                                    return temp;
                                }
                                kernel void transformationReductionKernel( global int * n,
                                global double * prm_1,
                                global double * prm_2,
                                global double * prm_3,
                                global double2 * gradientandhessian,
                                local double2 * smem
                                )
                                {
                                    local double2 * sdata = smem;
                                    size_t tid = get_local_id(0);
                                    size_t block_size = get_local_size(0);
                                    double2 mySum;
                                    mySum.x = 0.0;
                                    mySum.y = 0.0;
                                    for(int idx = get_global_id(0); idx < n[0]; idx += get_global_size(0))
                                    {
                                        mySum = transformSumKernel(mySum, prm_1[idx], prm_2[idx],prm_3[idx]);
                                    }
                                    sdata[tid] = mySum;
                                    barrier(CLK_LOCAL_MEM_FENCE);
                                    if (block_size >= 1024)
                                    {
                                        if (tid < 512) { sdata[tid] = mySum = sumDouble2(mySum, sdata[tid + 512]); }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
                                    if (block_size >= 512)
                                    {
                                        if (tid < 256) { sdata[tid] = mySum = sumDouble2(mySum, sdata[tid + 256]); }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
                                    if (block_size >= 256)
                                    {
                                        if (tid < 128) { sdata[tid] = mySum = sumDouble2(mySum, sdata[tid + 128]); }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
                                    if (block_size >= 128)
                                    {
                                        if (tid < 64) { sdata[tid] = mySum = sumDouble2(mySum, sdata[tid + 64]); }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
                                    if (tid < 32)
                                    {
                                        volatile local double2 * smem = sdata;
                                        if (block_size >= 64) { smem[tid] = mySum = sumDouble2(mySum, smem[tid + 32]); }
                                        if (block_size >= 32) { smem[tid] = mySum = sumDouble2(mySum, smem[tid + 16]); }
                                        if (block_size >= 16) { smem[tid] = mySum = sumDouble2(mySum, smem[tid + 8]); }
                                        if (block_size >= 8) { smem[tid] = mySum = sumDouble2(mySum, smem[tid + 4]); }
                                        if (block_size >= 4) { smem[tid] = mySum = sumDouble2(mySum, smem[tid + 2]); }
                                        if (block_size >= 2) { smem[tid] = mySum = sumDouble2(mySum, smem[tid + 1]); }
                                    }
                                    if (tid == 0) {
                                        gradientandhessian[get_group_id(0)] = smem[tid];
                                    }

                                }
                                ),
                                "transformationReductionKernel"
                                );
                }
            }

	        //push arguments onto the queue, run the kernel
	        for(int d = 0; d < ctx.size(); d++) {
	            kernel[d].push_arg(vex_length(d)); //kernel[d].push_arg<int>(vex_denominator.part_size(d));
                if (index == 0){
                    kernel[d].push_arg(vex_DataVector0(d));
                } else if (index == 1){
                    kernel[d].push_arg(vex_DataVector1(d));
                } else{
                    kernel[d].push_arg(vex_DataVector2(d));
                }
	            kernel[d].push_arg(vex_offsExpXBeta(d));
	            kernel[d].push_arg(vex_denominator(d));
	            kernel[d].push_arg(vex_gradientandhessian(d));
	            kernel[d].set_smem([](size_t wgs){ return wgs * 2 * sizeof(real); });
	            kernel[d](ctx.queue(d));
	        }

            vex::copy(vex_gradientandhessian.begin(), vex_gradientandhessian.end(), reinterpret_cast<cl_double2*>(gradientandhessian.data()));

	        gradienttest = gradientandhessian[0].real();
	        hessiantest = gradientandhessian[0].imag();

	        if (loopCount > 0) {
	            // stop timer;
//                gettimeofday(&time2, NULL);
//	            double time = Timer::calculateSeconds(time1, time2);
//	            totalTime += time;
	        }

	        std::cout << "gradienttest = " << gradienttest << std::endl;
	        std::cout << "hessiantest = " << hessiantest << std::endl;


            /*
	        if (loopCount > 2) {
	            std::ostringstream stream;
	            stream << "Time: " << totalTime;
	            Rcpp::stop(stream.str());
	        }
            */
	        ++loopCount;
	    }
        #endif

	    const auto result = variants::reduce(range.begin(), range.end(), Fraction<real>(0,0),
	    TransformAndAccumulateGradientAndHessianKernelIndependent<BaseModel,IteratorType, Weights, real, int>(),
	    SerialOnly()

	    // 		RcppParallel()
	    );


	    // 		const auto result2 = variants::reduce(range.begin(), range.end(), Fraction<real>(0,0),
	    // 		    TransformAndAccumulateGradientAndHessianKernelIndependent<BaseModel,IteratorType, Weights, real, int>(),
	    // // 			SerialOnly()
	    // 			RcppParallel()
	    // 		);


	    // 		std::cerr << result.real() << " " << result.imag()	<< std::endl;
	    // 		std::cerr << result2.real() << " " << result2.imag()	<< std::endl << std::endl;

	    gradient = result.real();
	    hessian = result.imag();


	} else {

// #ifdef OLD_WAY
//
// 		auto range = helper::getRangeDenominator(sparseIndices[index], N, typename IteratorType::tag());
//
// 		auto kernel = AccumulateGradientAndHessianKernel<BaseModel,IteratorType, Weights, real, int>(
// 							begin(numerPid), begin(numerPid2), begin(denomPid),
// 							begin(hNWeight), begin(hXBeta), begin(hY));
//
// 		Fraction<real> result = variants::reduce(range.begin(), range.end(), Fraction<real>(0,0), kernel,
// 		 SerialOnly()
// 	//     info
// 		);
//
// 		gradient = result.real();
// 		hessian = result.imag();
//
// #endif
//
// #ifdef NEW_WAY2

		auto rangeKey = helper::dependent::getRangeKey(modelData, index, hPid,
		        typename IteratorType::tag());

        auto rangeXNumerator = helper::dependent::getRangeX(modelData, index, offsExpXBeta,
                typename IteratorType::tag());

        auto rangeGradient = helper::dependent::getRangeGradient(*sparseIndices[index], N,
                denomPid, hNWeight,
                typename IteratorType::tag());

		const auto result = variants::trial::nested_reduce(
		        rangeKey.begin(), rangeKey.end(),
		        rangeXNumerator.begin(), rangeGradient.begin(),
		        std::pair<real,real>{0,0}, Fraction<real>{0,0},
                TestNumeratorKernel<BaseModel,IteratorType,real>(), // Inner transform-reduce
		       	TestGradientKernel<BaseModel,IteratorType,Weights,real>()); // Outer transform-reduce

		gradient = result.real();
		hessian = result.imag();
// #endif

//       std::cerr << std::endl
//            << result.real() << " " << result.imag() << std::endl
//            << result2.real() << " " << result2.imag() << std::endl
// 		   		 << result3.real() << " " << result3.imag() << std::endl;

//  		::Rf_error("break");

    } // not Cox

	if (BaseModel::precomputeGradient) { // Compile-time switch
		gradient -= hXjY[index];
	}

	if (BaseModel::precomputeHessian) { // Compile-time switch
		hessian += static_cast<real>(2.0) * hXjX[index];
	}

 	*ogradient = static_cast<double>(gradient);
	*ohessian = static_cast<double>(hessian);

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	auto name = "compGradHess" + IteratorType::name + "  ";
	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

 }

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::computeFisherInformation(int indexOne, int indexTwo,
		double *oinfo, bool useWeights) {

	if (useWeights) {
// 		std::cerr << "Weights are not yet implemented in Fisher Information calculations" << std::endl;
// 		exit(-1);
		throw new std::logic_error("Weights are not yet implemented in Fisher Information calculations");
	} else { // no weights
		switch (modelData.getFormatType(indexOne)) {
			case INDICATOR :
				dispatchFisherInformation<IndicatorIterator>(indexOne, indexTwo, oinfo, weighted);
				break;
			case SPARSE :
				dispatchFisherInformation<SparseIterator>(indexOne, indexTwo, oinfo, weighted);
				break;
			case DENSE :
				dispatchFisherInformation<DenseIterator>(indexOne, indexTwo, oinfo, weighted);
				break;
			case INTERCEPT :
				dispatchFisherInformation<InterceptIterator>(indexOne, indexTwo, oinfo, weighted);
				break;
		}
	}
}

template <class BaseModel, typename WeightType> template <typename IteratorTypeOne, class Weights>
void ModelSpecificsGPU<BaseModel,WeightType>::dispatchFisherInformation(int indexOne, int indexTwo, double *oinfo, Weights w) {
	switch (modelData.getFormatType(indexTwo)) {
		case INDICATOR :
			computeFisherInformationImpl<IteratorTypeOne,IndicatorIterator>(indexOne, indexTwo, oinfo, w);
			break;
		case SPARSE :
			computeFisherInformationImpl<IteratorTypeOne,SparseIterator>(indexOne, indexTwo, oinfo, w);
			break;
		case DENSE :
			computeFisherInformationImpl<IteratorTypeOne,DenseIterator>(indexOne, indexTwo, oinfo, w);
			break;
		case INTERCEPT :
			computeFisherInformationImpl<IteratorTypeOne,InterceptIterator>(indexOne, indexTwo, oinfo, w);
			break;
	}
//	std::cerr << "End of dispatch" << std::endl;
}


template<class BaseModel, typename WeightType> template<class IteratorType>
SparseIterator ModelSpecificsGPU<BaseModel, WeightType>::getSubjectSpecificHessianIterator(int index) {

	if (hessianSparseCrossTerms.find(index) == hessianSparseCrossTerms.end()) {
		// Make new
//		std::vector<int>* indices = new std::vector<int>();
        auto indices = make_shared<std::vector<int> >();
//		std::vector<real>* values = new std::vector<real>();
        auto values = make_shared<std::vector<real> >();
//		CompressedDataColumn* column = new CompressedDataColumn(indices, values,
//				SPARSE);
    	CDCPtr column = bsccs::make_shared<CompressedDataColumn>(indices, values, SPARSE);
		hessianSparseCrossTerms.insert(std::make_pair(index,
// 		    CompressedDataColumn(indices, values, SPARSE)));
		    column));

		IteratorType itCross(modelData, index);
		for (; itCross;) {
			real value = 0.0;
			int currentPid = hPid[itCross.index()];  // TODO Need to fix for stratified Cox
			do {
				const int k = itCross.index();
				value += BaseModel::gradientNumeratorContrib(itCross.value(),
						offsExpXBeta[k], hXBeta[k], hY[k]);
				++itCross;
			} while (itCross && currentPid == hPid[itCross.index()]); // TODO Need to fix for stratified Cox
			indices->push_back(currentPid);
			values->push_back(value);
		}
	}
	return SparseIterator(*hessianSparseCrossTerms[index]);

}

template <class BaseModel, typename WeightType> template <class IteratorTypeOne, class IteratorTypeTwo, class Weights>
void ModelSpecificsGPU<BaseModel,WeightType>::computeFisherInformationImpl(int indexOne, int indexTwo, double *oinfo, Weights w) {

	IteratorTypeOne itOne(modelData, indexOne);
	IteratorTypeTwo itTwo(modelData, indexTwo);
	PairProductIterator<IteratorTypeOne,IteratorTypeTwo> it(itOne, itTwo);

	real information = static_cast<real>(0);
	for (; it.valid(); ++it) {
		const int k = it.index();
		// Compile-time delegation

		BaseModel::incrementFisherInformation(it,
				w, // Signature-only, for iterator-type specialization
				&information,
				offsExpXBeta[k],
				0.0, 0.0, // numerPid[k], numerPid2[k], // remove
				denomPid[BaseModel::getGroup(hPid, k)],
				hKWeight[k], it.value(), hXBeta[k], hY[k]); // When function is in-lined, compiler will only use necessary arguments
	}

	if (BaseModel::hasStrataCrossTerms) {

		// Check if index is pre-computed
//#define USE_DENSE
#ifdef USE_DENSE
		if (hessianCrossTerms.find(indexOne) == hessianCrossTerms.end()) {
			// Make new
			std::vector<real> crossOneTerms(N);
			IteratorTypeOne crossOne(modelData, indexOne);
			for (; crossOne; ++crossOne) {
				const int k = crossOne.index();
				incrementByGroup(crossOneTerms.data(), hPid, k,
						BaseModel::gradientNumeratorContrib(crossOne.value(), offsExpXBeta[k], hXBeta[k], hY[k]));
			}
			hessianCrossTerms[indexOne];
//			std::cerr << std::accumulate(crossOneTerms.begin(), crossOneTerms.end(), 0.0) << std::endl;
			hessianCrossTerms[indexOne].swap(crossOneTerms);
		}
		std::vector<real>& crossOneTerms = hessianCrossTerms[indexOne];

		// TODO Remove code duplication
		if (hessianCrossTerms.find(indexTwo) == hessianCrossTerms.end()) {
			std::vector<real> crossTwoTerms(N);
			IteratorTypeTwo crossTwo(modelData, indexTwo);
			for (; crossTwo; ++crossTwo) {
				const int k = crossTwo.index();
				incrementByGroup(crossTwoTerms.data(), hPid, k,
						BaseModel::gradientNumeratorContrib(crossTwo.value(), offsExpXBeta[k], hXBeta[k], hY[k]));
			}
			hessianCrossTerms[indexTwo];
//			std::cerr << std::accumulate(crossTwoTerms.begin(), crossTwoTerms.end(), 0.0) << std::endl;
			hessianCrossTerms[indexTwo].swap(crossTwoTerms);
		}
		std::vector<real>& crossTwoTerms = hessianCrossTerms[indexTwo];

		// TODO Sparse loop
		real cross = 0.0;
		for (int n = 0; n < N; ++n) {
			cross += crossOneTerms[n] * crossTwoTerms[n] / (denomPid[n] * denomPid[n]);
		}
//		std::cerr << cross << std::endl;
		information -= cross;
#else
		SparseIterator sparseCrossOneTerms = getSubjectSpecificHessianIterator<IteratorTypeOne>(indexOne);
		SparseIterator sparseCrossTwoTerms = getSubjectSpecificHessianIterator<IteratorTypeTwo>(indexTwo);
		PairProductIterator<SparseIterator,SparseIterator> itSparseCross(sparseCrossOneTerms, sparseCrossTwoTerms);

		real sparseCross = 0.0;
		for (; itSparseCross.valid(); ++itSparseCross) {
			const int n = itSparseCross.index();
			sparseCross += itSparseCross.value() / (denomPid[n] * denomPid[n]);
		}
		information -= sparseCross;
#endif
	}

	*oinfo = static_cast<double>(information);
}

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::computeNumeratorForGradient(int index) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	if (BaseModel::cumulativeGradientAndHessian) {
//
// 		// Run-time delegation
// 		switch (modelData.getFormatType(index)) {
// 			case INDICATOR :
// 				incrementNumeratorForGradientImpl<IndicatorIterator>(index);
// 				break;
// 			case SPARSE :
// 				incrementNumeratorForGradientImpl<SparseIterator>(index);
// 				break;
// 			case DENSE :
// 				incrementNumeratorForGradientImpl<DenseIterator>(index);
// 				break;
// 			case INTERCEPT :
// 				incrementNumeratorForGradientImpl<InterceptIterator>(index);
// 				break;
// 			default : break;
// 				// throw error
// 		}
		switch (modelData.getFormatType(index)) {
			case INDICATOR : {
				IndicatorIterator it(*(sparseIndices)[index]);
				for (; it; ++it) { // Only affected entries
					numerPid[it.index()] = static_cast<real>(0.0);
				}
				incrementNumeratorForGradientImpl<IndicatorIterator>(index);
				}
				break;
			case SPARSE : {
				SparseIterator it(*(sparseIndices)[index]);
				for (; it; ++it) { // Only affected entries
					numerPid[it.index()] = static_cast<real>(0.0);
					if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
						numerPid2[it.index()] = static_cast<real>(0.0); // TODO Does this invalid the cache line too much?
					}
				}
				incrementNumeratorForGradientImpl<SparseIterator>(index); }
				break;
			case DENSE :
				zeroVector(numerPid.data(), N);
				if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
					zeroVector(numerPid2.data(), N);
				}
				incrementNumeratorForGradientImpl<DenseIterator>(index);
				break;
			case INTERCEPT :
				zeroVector(numerPid.data(), N);
				if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
					zeroVector(numerPid2.data(), N);
				}
				incrementNumeratorForGradientImpl<InterceptIterator>(index);
				break;
			default : break;
				// throw error
				//exit(-1);
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compNumForGrad   "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
#endif

}

template <class BaseModel,typename WeightType> template <class IteratorType>
void ModelSpecificsGPU<BaseModel,WeightType>::incrementNumeratorForGradientImpl(int index) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

// #ifdef NEW_LOOPS

// 	auto zeroRange = helper::getRangeNumerator(sparseIndices[index], N, typename IteratorType::tag());
//
// 	auto zeroKernel = ZeroOutNumerator<BaseModel,IteratorType,real,int>(
// 			begin(numerPid), begin(numerPid2)
// 	);
//
// 	variants::for_each( // TODO Rewrite as variants::fill if rate-limiting
// 		zeroRange.begin(), zeroRange.end(),
// 		zeroKernel,
// // 		info
// // 		threadPool
// 		SerialOnly()
// // 		RcppParallel()
// 	);
//
// 	if (true) {
//
// 	auto computeRange = helper::getRangeX(modelData, index, typename IteratorType::tag());
//
// 	auto computeKernel = NumeratorForGradientKernel<BaseModel,IteratorType,real,int>(
// 					begin(numerPid), begin(numerPid2),
// 					begin(offsExpXBeta), begin(hXBeta),
// 					begin(hY),
// 					begin(hPid));
//
// 	variants::for_each(
// 		computeRange.begin(), computeRange.end(),
// 		computeKernel,
// // 		info
// //   		threadPool
//  		SerialOnly()
// //		RcppParallel() //  TODO Not thread-safe
// 		);
//
// 	} else {
// //	auto computeRange = helper::getRangeXDependent(modelData, index,
// //		numerPid, numerPid2, offsExpXBeta, hXBeta, hY, hPid,
// //// 		typename IteratorType::tag()
// //		DenseTag()
// //		);
// //
// //	auto info = C11Threads(4, 100);
//
//     // Let computeRange -> tuple<i, j, x>
//
// //    auto computeRangeCOO = helper::getRangeCOOX(modelData, index,
// //        typename IteratorType::tag());
// //
// //    variants::transform_segmented_reduce(
// //        computeRangeCOO.begin(), computeRangeCOO.end()
// //
// //    );
//
// 	}

// #else

	IteratorType it(modelData, index);
	for (; it; ++it) {
		const int k = it.index();
		incrementByGroup(numerPid.data(), hPid, k,
				BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k]));
		if (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) {
			incrementByGroup(numerPid2.data(), hPid, k,
					BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k]));
		}

#ifdef DEBUG_COX
//			if (numerPid[BaseModel::getGroup(hPid, k)] > 0 && numerPid[BaseModel::getGroup(hPid, k)] < 1e-40) {
				cerr << "Increment" << endl;
				cerr << "hPid = "
				//<< hPid <<
				", k = " << k << ", index = " << BaseModel::getGroup(hPid, k) << endl;
				cerr << BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k]) <<  " "
				<< it.value() << " " << offsExpXBeta[k] << " " << hXBeta[k] << " " << hY[k] << endl;
//				exit(-1);
//			}
#endif



	}

// #endif // NEW_LOOPS


#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	auto name = "compNumGrad" + IteratorType::name + "   ";
	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::updateXBeta(real realDelta, int index, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	// Run-time dispatch to implementation depending on covariate FormatType
	switch(modelData.getFormatType(index)) {
		case INDICATOR :
			updateXBetaImpl<IndicatorIterator>(realDelta, index, useWeights);
			break;
		case SPARSE :
			updateXBetaImpl<SparseIterator>(realDelta, index, useWeights);
			break;
		case DENSE :
			updateXBetaImpl<DenseIterator>(realDelta, index, useWeights);
			break;
		case INTERCEPT :
			updateXBetaImpl<InterceptIterator>(realDelta, index, useWeights);
			break;
		default : break;
			// throw error
			//exit(-1);
	}

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["updateXBeta      "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel,typename WeightType> template <class IteratorType>
inline void ModelSpecificsGPU<BaseModel,WeightType>::updateXBetaImpl(real realDelta, int index, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

// #ifdef NEW_LOOPS

#if 1
	auto range = helper::getRangeX(modelData, index, typename IteratorType::tag());

	auto kernel = UpdateXBetaKernel<BaseModel,IteratorType,real,int>(
					realDelta, begin(offsExpXBeta), begin(hXBeta),
					begin(hY),
					begin(hPid),
					begin(denomPid),
					begin(hOffs)
					);


	variants::for_each(
		range.begin(), range.end(),
		kernel,
// 		info
//          threadPool
// 		RcppParallel() // TODO Currently *not* thread-safe
          SerialOnly()
		);

#else

    if (BaseModel::hasIndependentRows) {

        auto range = helper::independent::getRangeXBeta(modelData, index,
                offsExpXBeta, hXBeta, denomPid, hOffs,
                typename IteratorType::tag());

        auto kernel = TestUpdateXBetaKernel<BaseModel,IteratorType,real>(realDelta);
        variants::for_each(
            range.begin(), range.end(),
            kernel,
            SerialOnly()
        );

    } else {

        auto rangeXBeta = helper::independent::getRangeXBeta(modelData, index,
            offsExpXBeta, hXBeta, denomPid, /* denom not used here */ hOffs,
            typename IteratorType::tag());

 		auto rangeKey = helper::dependent::getRangeKey(modelData, index, hPid,
		        typename IteratorType::tag());

		auto rangeDenominator = helper::dependent::getRangeDenominator(*sparseIndices[index], N,
		        denomPid, typename IteratorType::tag());

        auto kernel = TestUpdateXBetaKernelDependent<BaseModel,IteratorType,real>(realDelta);

        auto key = rangeKey.begin();
        auto end = rangeKey.end();
        auto inner = rangeXBeta.begin();
        auto outer = rangeDenominator.begin();

        const auto stop = end - 1;

        real result = 0;

        for (; key != stop; ++key, ++inner) {

            result = kernel(result, *inner);

            if (*key != *(key + 1)) {

                *outer = result + *outer;

                result = 0;
                ++outer;
            }
        }

        result = kernel(result, *inner);

        *outer = result + *outer;
    }

#endif

// 	std::cerr << std::endl << realDelta << std::endl;
//
// 	::Rf_error("return");



// #else
// 	IteratorType it(modelData, index);
// 	for (; it; ++it) {
// 		const int k = it.index();
// 		hXBeta[k] += realDelta * it.value(); // TODO Check optimization with indicator and intercept
// 		// Update denominators as well
// 		if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
// 			real oldEntry = offsExpXBeta[k];
// 			real newEntry = offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k);
// 			incrementByGroup(denomPid, hPid, k, (newEntry - oldEntry));
// 		}
// 	}
//
// #endif

	computeAccumlatedDenominator(useWeights);

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	auto name = "updateXBeta" + IteratorType::name + "   ";
	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::computeRemainingStatistics(bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
	auto start = bsccs::chrono::steady_clock::now();
#endif


	if (BaseModel::likelihoodHasDenominator) {
		fillVector(denomPid.data(), N, BaseModel::getDenomNullValue());
		for (size_t k = 0; k < K; ++k) {
			offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k);
			incrementByGroup(denomPid.data(), hPid, k, offsExpXBeta[k]);
		}
		computeAccumlatedDenominator(useWeights); // WAS computeAccumlatedNumerDenom
	}
#ifdef DEBUG_COX
	cerr << "Done with initial denominators" << endl;

	for (int i = 0; i < N; ++i) {
		cerr << denomPid[i] << " " << accDenomPid[i] << " " << numerPid[i] << endl;
	}
#endif

#ifdef CYCLOPS_DEBUG_TIMING
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compRS           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

}

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::computeAccumlatedNumerator(bool useWeights) {

	if (BaseModel::likelihoodHasDenominator && //The two switches should ideally be separated
			BaseModel::cumulativeGradientAndHessian) { // Compile-time switch
		if (accNumerPid.size() != N) {
			accNumerPid.resize(N, static_cast<real>(0));
		}
		if (accNumerPid2.size() != N) {
			accNumerPid2.resize(N, static_cast<real>(0));
		}

		// segmented prefix-scan
		real totalNumer = static_cast<real>(0);
		real totalNumer2 = static_cast<real>(0);

		auto reset = begin(accReset);

		for (size_t i = 0; i < N; ++i) {

			if (static_cast<unsigned int>(*reset) == i) {
				totalNumer = static_cast<real>(0);
				totalNumer2 = static_cast<real>(0);
				++reset;
			}

			totalNumer += numerPid[i];
			totalNumer2 += numerPid2[i];
			accNumerPid[i] = totalNumer;
			accNumerPid2[i] = totalNumer2;
		}
	}
}

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::computeAccumlatedDenominator(bool useWeights) {

	if (BaseModel::likelihoodHasDenominator && //The two switches should ideally be separated
		BaseModel::cumulativeGradientAndHessian) { // Compile-time switch
			if (accDenomPid.size() != N) {
				accDenomPid.resize(N, static_cast<real>(0));
			}
// 			if (accNumerPid.size() != N) {
// 				accNumerPid.resize(N, static_cast<real>(0));
// 			}
// 			if (accNumerPid2.size() != N) {
// 				accNumerPid2.resize(N, static_cast<real>(0));
// 			}

			// segmented prefix-scan
			real totalDenom = static_cast<real>(0);
// 			real totalNumer = static_cast<real>(0);
// 			real totalNumer2 = static_cast<real>(0);

			auto reset = begin(accReset);

			for (size_t i = 0; i < N; ++i) {
// TODO CHECK
				if (static_cast<unsigned int>(*reset) == i) { // TODO Check with SPARSE
					totalDenom = static_cast<real>(0);
// 					totalNumer = static_cast<real>(0);
// 					totalNumer2 = static_cast<real>(0);
					++reset;
				}

				totalDenom += denomPid[i];
// 				totalNumer += numerPid[i];
// 				totalNumer2 += numerPid2[i];
				accDenomPid[i] = totalDenom;
// 				accNumerPid[i] = totalNumer;
// 				accNumerPid2[i] = totalNumer2;
#if defined(DEBUG_COX) || defined(DEBUG_COX_MIN)
				cerr << denomPid[i] << " " << accDenomPid[i] << " (beta)" << endl;
#endif
			}
	}
}

template <class BaseModel,typename WeightType>
void ModelSpecificsGPU<BaseModel,WeightType>::doSortPid(bool useCrossValidation) {
/* For Cox model:
 *
 * We currently assume that hZ[k] are sorted in decreasing order by k.
 *
 */
}

} // namespace

#endif /* ModelSpecificsGPU_HPP_ */
//

