/*
 * ModelSpecifics.hpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef MODELSPECIFICS_HPP_
#define MODELSPECIFICS_HPP_

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <limits>

#include "ModelSpecifics.h"
#include "Iterators.h"

#include "Recursions.hpp"
#include "ParallelLoops.h"
// #include "Ranges.h"

#ifdef USE_RCPP_PARALLEL
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#endif


//#include "R.h"
//#include "Rcpp.h" // TODO Remove

#ifdef CYCLOPS_DEBUG_TIMING
	#include "Timing.h"
#endif
	namespace bsccs {
	    template <typename RealType>
	    const std::string DenseIterator<RealType>::name = "Den";

	    template <typename RealType>
	    const std::string IndicatorIterator<RealType>::name = "Ind";

	    template <typename RealType>
	    const std::string SparseIterator<RealType>::name = "Spa";

	    template <typename RealType>
	    const std::string InterceptIterator<RealType>::name = "Icp";
	}
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

template <class BaseModel,typename RealType>
ModelSpecifics<BaseModel,RealType>::ModelSpecifics(const ModelData<RealType>& input)
	: AbstractModelSpecifics(input), BaseModel(input.getYVectorRef(), input.getTimeVectorRef()),
   modelData(input),
   hX(modelData.getX())
   // hY(input.getYVectorRef()),
   // hOffs(input.getTimeVectorRef())
 //  hPidOriginal(input.getPidVectorRef()), hPid(const_cast<int*>(hPidOriginal.data())),
 //  boundType(MmBoundType::METHOD_2)
//  	threadPool(4,4,1000)
// threadPool(0,0,10)
	{
    	// Do nothing
	// TODO Memory allocation here
	// std::cerr << "ctor ModelSpecifics \n";

#ifdef CYCLOPS_DEBUG_TIMING
	auto now = bsccs::chrono::system_clock::now();
	auto now_c = bsccs::chrono::system_clock::to_time_t(now);
	std::cout << std::endl << "Start: " << std::ctime(&now_c) << std::endl;
#endif


}

template <class BaseModel, typename RealType>
AbstractModelSpecifics* ModelSpecifics<BaseModel,RealType>::clone(ComputeDeviceArguments computeDevice) const {
	return new ModelSpecifics<BaseModel,RealType>(modelData);
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::getGradientObjective(bool useCrossValidation) {

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

		auto& xBeta = getXBeta();

		RealType criterion = 0;
		if (useCrossValidation) {
			for (int i = 0; i < K; i++) {
				criterion += xBeta[i] * hY[i] * hKWeight[i];
			}
		} else {
			for (int i = 0; i < K; i++) {
				criterion += xBeta[i] * hY[i];
            }
		}
#ifdef CYCLOPS_DEBUG_TIMING
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    duration["getGradObj       "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
		return static_cast<double> (criterion);
	}

template <class BaseModel, typename RealType>
void ModelSpecifics<BaseModel,RealType>::printTiming() {

#ifdef CYCLOPS_DEBUG_TIMING

	std::cout << std::endl;
	for (auto& d : duration) {
		std::cout << d.first << " " << d.second << std::endl;
	}
	std::cout << "NEW LOOPS" << std::endl;

#endif
}

template <class BaseModel,typename RealType>
ModelSpecifics<BaseModel,RealType>::~ModelSpecifics() {
	// TODO Memory release here
	// std::cerr << "dtor ModelSpecifics \n";

#ifdef CYCLOPS_DEBUG_TIMING

    printTiming();

	auto now = bsccs::chrono::system_clock::now();
	auto now_c = bsccs::chrono::system_clock::to_time_t(now);
	std::cout << std::endl << "End:   " << std::ctime(&now_c) << std::endl;
#endif

}

template <class BaseModel, typename RealType> template <class IteratorType>
void ModelSpecifics<BaseModel,RealType>::incrementNormsImpl(int index) {

    // TODO should use row-access
	IteratorType it(hX, index);
	for (; it; ++it) {
		const int k = it.index();
		const RealType x = it.value();

		norm[k] += std::abs(x);
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::initializeMmXt() {

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

    hXt = hX.transpose();

#ifdef CYCLOPS_DEBUG_TIMING
auto end = bsccs::chrono::steady_clock::now();
///////////////////////////"
duration["transpose        "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::initializeMM(
        MmBoundType boundType,
        const std::vector<bool>& fixBeta
    ) {

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

    // std::cout << "Computing norms..." << std::endl;
    norm.resize(K);
    zeroVector(&norm[0], K);

    for (int j = 0; j < J; ++j) {
    	if (
    	true
    	// !fixBeta[j]
    	) {
			switch(hX.getFormatType(j)) {
				case INDICATOR :
					incrementNormsImpl<IndicatorIterator<RealType>>(j);
					break;
				case SPARSE :
					incrementNormsImpl<SparseIterator<RealType>>(j);
					break;
				case DENSE :
					incrementNormsImpl<DenseIterator<RealType>>(j);
					break;
				case INTERCEPT :
					incrementNormsImpl<InterceptIterator<RealType>>(j);
					break;
			}
        }
    }

    if (boundType == MmBoundType::METHOD_1) {
        // std::cerr << "boundType: METHOD_1" << std::endl;


    } else if (boundType == MmBoundType::METHOD_2) {
        // std::cerr << "boundType: METHOD_2" << std::endl;

        double total = 0;

        for (int j = 0; j < J; ++j) {
            if (!fixBeta[j]) {
                typedef std::pair<double,double> Range;

                std::vector<Range> range(modelData.getNumberOfPatients(), Range(
                        -std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max()
                ));

                modelData.binaryReductionByStratum(range, j, [](Range& r, double x) {
                    return Range(
                        std::max(x, r.first),
                        std::min(x, r.second)
                    );
                });

                double curvature = 0.0;
                for (Range r : range) {
                    curvature += r.first - r.second;
                }
                curvature /= 4;

                total += curvature;
            }
        }

        if (curvature.size() == 0) {
            curvature.resize(J);
        }

        for (int j = 0; j < J; ++j) {
            curvature[j] = total;
        }
    }

#ifdef CYCLOPS_DEBUG_TIMING
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    duration["norms            "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

// 	struct timeval time1, time2;
// 	gettimeofday(&time1, NULL);
//
//     std::cout << "Constructing Xt..." << std::endl;

//     if (!hXt) {
//         initializeMmXt();
//     }

//     gettimeofday(&time2, NULL);
// 	double duration = //calculateSeconds(time1, time2);
// 		time2.tv_sec - time1.tv_sec +
// 			(double)(time2.tv_usec - time1.tv_usec) / 1000000.0;
//
// 	std::cout << "Done with MM initialization" << std::endl;
// 	std::cout << duration << std::endl;
	// Rcpp::stop("out");

}

template <class BaseModel,typename RealType>
const std::vector<double> ModelSpecifics<BaseModel,RealType>::getXBeta() {
    return std::vector<double>(std::begin(hXBeta), std::end(hXBeta));
}

template <class BaseModel,typename RealType>
const std::vector<double> ModelSpecifics<BaseModel,RealType>::getXBetaSave() {
    return std::vector<double>(std::begin(hXBetaSave), std::end(hXBetaSave));
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::zeroXBeta() {
	std::fill(std::begin(hXBeta), std::end(hXBeta), 0.0);
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::saveXBeta() {
	auto& xBeta = getXBeta();
	if (hXBetaSave.size() < xBeta.size()) {
		hXBetaSave.resize(xBeta.size());
	}
	std::copy(std::begin(xBeta), std::end(xBeta), std::begin(hXBetaSave));
}



template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeXBeta(double* beta, bool useWeights) {

    if (!hXt) {
        initializeMmXt();
    }

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

    switch(hXt->getFormatType(0)) {
    case INDICATOR :
        computeXBetaImpl<IndicatorIterator<RealType>>(beta);
        break;
    case SPARSE :
        computeXBetaImpl<SparseIterator<RealType>>(beta);
        break;
    case DENSE :
        computeXBetaImpl<DenseIterator<RealType>>(beta);
        break;
    case INTERCEPT:
        break;
}


#ifdef CYCLOPS_DEBUG_TIMING
auto end = bsccs::chrono::steady_clock::now();
///////////////////////////"
duration["computeXBeta     "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

}

template <class BaseModel,typename RealType> template <class IteratorType>
void ModelSpecifics<BaseModel,RealType>::computeXBetaImpl(double *beta) {

    for (int k = 0; k < K; ++k) {
        RealType sum = 0.0;
        IteratorType it(*hXt, k);
        for (; it; ++it) {
            const auto j = it.index();
            sum += it.value() * beta[j];
        }
        hXBeta[k] = sum;
        // TODO Add back in for fastest LR
//         const auto exb = std::exp(sum);
//         offsExpXBeta[k] = exb;
//         denomPid[k] = 1.0 + exb;
    }

    // std::cerr << "did weird stuff to denomPid" << std::endl;
}

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::allocateXjY(void) { return BaseModel::precomputeGradient; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::allocateXjX(void) { return BaseModel::precomputeHessian; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::sortPid(void) { return BaseModel::sortPid; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::initializeAccumulationVectors(void) { return BaseModel::cumulativeGradientAndHessian; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::allocateNtoKIndices(void) { return BaseModel::hasNtoKIndices; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::hasResetableAccumulators(void) { return BaseModel::hasResetableAccumulators; }

template <class BaseModel,typename RealType> template <class IteratorType>
void ModelSpecifics<BaseModel,RealType>::axpy(RealType* y, const RealType alpha, const int index) {
	IteratorType it(hX, index);
	for (; it; ++it) {
		const int k = it.index();
		y[k] += alpha * it.value();
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::axpyXBeta(const double beta, const int j) {

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

	if (beta != 0.0) {
		switch (hX.getFormatType(j)) {
		case INDICATOR:
			axpy < IndicatorIterator<RealType> > (hXBeta.data(), beta, j);
			break;
		case INTERCEPT:
		    axpy < InterceptIterator<RealType> > (hXBeta.data(), beta, j);
		    break;
		case DENSE:
			axpy < DenseIterator<RealType> > (hXBeta.data(), beta, j);
			break;
		case SPARSE:
			axpy < SparseIterator<RealType> > (hXBeta.data(), beta, j);
			break;
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["axpy             "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

}


// ESK: Added cWeights (censoring weights) as an input
template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::setWeights(double* inWeights, double *cenWeights, bool useCrossValidation) {
	// Set K weights
	if (hKWeight.size() != K) {
		hKWeight.resize(K);
	}

	if (useCrossValidation) {
		for (size_t k = 0; k < K; ++k) {
			hKWeight[k] = inWeights[k];
		}
	} else {
		std::fill(hKWeight.begin(), hKWeight.end(), static_cast<RealType>(1));
    }

	if (initializeAccumulationVectors()) {
		setPidForAccumulation(inWeights);
	}

	// Set N weights (these are the same for independent data models)
	if (hNWeight.size() < N + 1) { // Add +1 for extra (zero-weight stratum)
		hNWeight.resize(N + 1);
	}

	std::fill(hNWeight.begin(), hNWeight.end(), static_cast<RealType>(0));
    for (size_t k = 0; k < K; ++k) {
        RealType event = BaseModel::observationCount(hY[k]) * hKWeight[k];
        incrementByGroup(hNWeight.data(), hPid, k, event);
    }

    //ESK: Compute hNtoK indices manually and create censoring weights hYWeight is isTwoWayScan
    if (hYWeight.size() != K) {
        hYWeight.resize(K);
    }
	if (hYWeightDouble.size() != K) {
		hYWeightDouble.resize(K);
	}
    if (BaseModel::isTwoWayScan) {
        hNtoK.resize(N + 1);
        int n = 0;
        for (size_t k = 0; k < K;) {
            while (hKWeight[k] == static_cast<RealType>(0)) {
                ++k;
            }
            hNtoK[n] = k;
            int currentPid = hPid[k];
            do {
                ++k;
            } while (k < K && (currentPid == hPid[k] || hKWeight[k] == static_cast<RealType>(0)));
            ++n;
        }
        hNtoK[n] = K;

        for (size_t k = 0; k < K; ++k) {
            hYWeight[k] = cenWeights[k];
            hYWeightDouble[k] = cenWeights[k];
        }
    }

#ifdef DEBUG_COX
	cerr << "Done with set weights" << endl;
#endif

}

template<class BaseModel, typename RealType>
void ModelSpecifics<BaseModel, RealType>::computeXjY(bool useCrossValidation) {
	for (size_t j = 0; j < J; ++j) {
		hXjY[j] = 0;

		GenericIterator<RealType> it(hX, j);

		if (useCrossValidation) {
			for (; it; ++it) {
                const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
					hXjY[j] += it.value() * hY[k] * hKWeight[k]; // TODO Remove
				} else if (BaseModel::isSurvivalModel) {
				    // ESK: Modified for suvival data
                    hXjY[j] += it.value() * BaseModel::observationCount(hY[k]) * hKWeight[k];
				} else {
					hXjY[j] += it.value() * hY[k] * hKWeight[k];
				}
			}
		} else {
			for (; it; ++it) {
                const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
                    hXjY[j] += it.value() * hY[k];
				} else if (BaseModel::isSurvivalModel) {
                    // ESK: Modified for survival data
                    hXjY[j] += it.value() * BaseModel::observationCount(hY[k]);
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

template<class BaseModel, typename RealType>
void ModelSpecifics<BaseModel, RealType>::computeXjX(bool useCrossValidation) {
	for (size_t j = 0; j < J; ++j) {
		hXjX[j] = 0;
		GenericIterator<RealType> it(hX, j);

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

template<class BaseModel, typename RealType>
void ModelSpecifics<BaseModel, RealType>::computeNtoKIndices(bool useCrossValidation) {

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

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeFixedTermsInLogLikelihood(bool useCrossValidation) {
	if(BaseModel::likelihoodHasFixedTerms) {
		logLikelihoodFixedTerm = static_cast<RealType>(0);
	    bool hasOffs = hOffs.size() > 0;
		if (useCrossValidation) {
			for(size_t i = 0; i < K; i++) {
			    auto offs = hasOffs ? hOffs[i] : static_cast<RealType>(0);
				logLikelihoodFixedTerm += BaseModel::logLikeFixedTermsContrib(hY[i], offs, offs) * hKWeight[i];
			}
		} else {
			for(size_t i = 0; i < K; i++) {
			    auto offs = hasOffs ? hOffs[i] : static_cast<RealType>(0);
				logLikelihoodFixedTerm += BaseModel::logLikeFixedTermsContrib(hY[i], offs, offs); // TODO SEGV in Poisson model
			}
		}
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeFixedTermsInGradientAndHessian(bool useCrossValidation) {
#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif
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
#ifdef CYCLOPS_DEBUG_TIMING
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    auto name = "compFixedGH      ";
    duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

template <class BaseModel,typename RealType>
double ModelSpecifics<BaseModel,RealType>::getLogLikelihood(bool useCrossValidation) {

#ifdef CYCLOPS_DEBUG_TIMING
	auto start = bsccs::chrono::steady_clock::now();
#endif

	RealType logLikelihood = static_cast<RealType>(0.0);
	if (useCrossValidation) {
		for (size_t i = 0; i < K; i++) {
			logLikelihood += BaseModel::logLikeNumeratorContrib(hY[i], hXBeta[i]) * hKWeight[i];
		}
	} else {
		for (size_t i = 0; i < K; i++) {
            logLikelihood += BaseModel::logLikeNumeratorContrib(hY[i], hXBeta[i]);
		}
    }

    if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
		if(BaseModel::cumulativeGradientAndHessian) {
			for (size_t i = 0; i < N; i++) {
				// Weights modified in computeNEvents()
				logLikelihood -= BaseModel::logLikeDenominatorContrib(hNWeight[i], accDenomPid[i]);
			}
		} else {  // TODO Unnecessary code duplication
			for (size_t i = 0; i < N; i++) {
				// Weights modified in computeNEvents()
				logLikelihood -= BaseModel::logLikeDenominatorContrib(hNWeight[i], denomPid[i]);
			}
		}
	}

    // RANGE

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

template <class BaseModel,typename RealType>
double ModelSpecifics<BaseModel,RealType>::getPredictiveLogLikelihood(double* weights) {

	std::vector<double> saveKWeight;
	if (BaseModel::cumulativeGradientAndHessian)	{

 	    // saveKWeight = hKWeight; // make copy
	    if (saveKWeight.size() != K) {
	        saveKWeight.resize(K);
	    }
	    for (size_t k = 0; k < K; ++k) {
	        saveKWeight[k] = hKWeight[k]; // make copy to a double vector
	    }

		setPidForAccumulation(weights);
		setWeights(weights, BaseModel::isTwoWayScan ? hYWeightDouble.data() : nullptr, true); // set new weights // TODO Possible error for gfr
		computeRemainingStatistics(true); // compute accDenomPid
	}

	RealType logLikelihood = static_cast<RealType>(0.0);

	if (BaseModel::cumulativeGradientAndHessian) {
	    for (size_t k = 0; k < K; ++k) {
	        logLikelihood += BaseModel::logPredLikeContrib(hY[k], weights[k], hXBeta[k], &accDenomPid[0], hPid, k); // TODO Going to crash with ties
	    }
	} else { // TODO Unnecessary code duplication
	    for (size_t k = 0; k < K; ++k) { // TODO Is index of K correct?
	        logLikelihood += BaseModel::logPredLikeContrib(hY[k], weights[k], hXBeta[k], &denomPid[0], hPid, k);
	    }
	}

	if (BaseModel::cumulativeGradientAndHessian) {
		setPidForAccumulation(&saveKWeight[0]);
		setWeights(saveKWeight.data(), BaseModel::isTwoWayScan ? hYWeightDouble.data() : nullptr, true); // set old weights // TODO Possible error for gfr
		computeRemainingStatistics(true);
	}

	return static_cast<double>(logLikelihood);
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::getPredictiveEstimates(double* y, double* weights){

	if (weights) {
		for (size_t k = 0; k < K; ++k) {
			if (weights[k]) {
				y[k] = BaseModel::predictEstimate(hXBeta[k]);
			}
		}
	} else {
		for (size_t k = 0; k < K; ++k) {
			y[k] = BaseModel::predictEstimate(hXBeta[k]);
		}
	}
	// TODO How to remove code duplication above?
}

// TODO The following function is an example of a double-dispatch, rewrite without need for virtual function
template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeGradientAndHessian(int index, double *ogradient,
		double *ohessian, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	if (hX.getNumberOfNonZeroEntries(index) == 0) {
	    *ogradient = 0.0; *ohessian = 0.0;
	    return;
	}

	// Run-time dispatch, so virtual call should not effect speed
	if (useWeights) {
		switch (hX.getFormatType(index)) {
			case INDICATOR :
				computeGradientAndHessianImpl<IndicatorIterator<RealType>>(index, ogradient, ohessian, weighted);
				break;
			case SPARSE :
				computeGradientAndHessianImpl<SparseIterator<RealType>>(index, ogradient, ohessian, weighted);
				break;
			case DENSE :
				computeGradientAndHessianImpl<DenseIterator<RealType>>(index, ogradient, ohessian, weighted);
				break;
			case INTERCEPT :
				computeGradientAndHessianImpl<InterceptIterator<RealType>>(index, ogradient, ohessian, weighted);
				break;
		}
	} else {
		switch (hX.getFormatType(index)) {
			case INDICATOR :
				computeGradientAndHessianImpl<IndicatorIterator<RealType>>(index, ogradient, ohessian, unweighted);
				break;
			case SPARSE :
				computeGradientAndHessianImpl<SparseIterator<RealType>>(index, ogradient, ohessian, unweighted);
				break;
			case DENSE :
				computeGradientAndHessianImpl<DenseIterator<RealType>>(index, ogradient, ohessian, unweighted);
				break;
			case INTERCEPT :
				computeGradientAndHessianImpl<InterceptIterator<RealType>>(index, ogradient, ohessian, unweighted);
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

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
        duration["CPU GH           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

template <class RealType>
std::pair<RealType, RealType> operator+(
        const std::pair<RealType, RealType>& lhs,
        const std::pair<RealType, RealType>& rhs) {

    return { lhs.first + rhs.first, lhs.second + rhs.second };
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeMMGradientAndHessian(
        std::vector<GradientHessian>& gh,
        const std::vector<bool>& fixBeta,
        bool useWeights) {

    if (norm.size() == 0) {
        initializeMM(boundType, fixBeta);
    }

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
    auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

    for (int index = 0; index < J; ++index) {
        double *ogradient = &(gh[index].first);
        double *ohessian  = &(gh[index].second);

        if (fixBeta[index]) {
            *ogradient = 0.0; *ohessian = 0.0;
        } else {

        // Run-time dispatch, so virtual call should not effect speed
        if (useWeights) {
            switch (hX.getFormatType(index)) {
            case INDICATOR :
                computeMMGradientAndHessianImpl<IndicatorIterator<RealType>>(index, ogradient, ohessian, weighted);
                break;
            case SPARSE :
                computeMMGradientAndHessianImpl<SparseIterator<RealType>>(index, ogradient, ohessian, weighted);
                break;
            case DENSE :
                computeMMGradientAndHessianImpl<DenseIterator<RealType>>(index, ogradient, ohessian, weighted);
                break;
            case INTERCEPT :
                computeMMGradientAndHessianImpl<InterceptIterator<RealType>>(index, ogradient, ohessian, weighted);
                break;
            }
        } else {
            switch (hX.getFormatType(index)) {
            case INDICATOR :
                computeMMGradientAndHessianImpl<IndicatorIterator<RealType>>(index, ogradient, ohessian, unweighted);
                break;
            case SPARSE :
                computeMMGradientAndHessianImpl<SparseIterator<RealType>>(index, ogradient, ohessian, unweighted);
                break;
            case DENSE :
                computeMMGradientAndHessianImpl<DenseIterator<RealType>>(index, ogradient, ohessian, unweighted);
                break;
            case INTERCEPT :
                computeMMGradientAndHessianImpl<InterceptIterator<RealType>>(index, ogradient, ohessian, unweighted);
                break;
            }
        }
        }
    }

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    duration["compMMGradAndHess"] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::computeMMGradientAndHessianImpl(int index, double *ogradient,
                                                                           double *ohessian, Weights w) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
    auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

    RealType gradient = static_cast<RealType>(0);
    RealType hessian = static_cast<RealType>(0);

    IteratorType it(hX, index);
    for (; it; ++it) {
        const int k = it.index();

        BaseModel::template incrementMMGradientAndHessian<IteratorType, Weights>(
                gradient, hessian, offsExpXBeta[k],
                denomPid[BaseModel::getGroup(hPid, k)], hNWeight[BaseModel::getGroup(hPid, k)],
                it.value(), hXBeta[k], hY[k], norm[k]);
    }

    if (BaseModel::precomputeGradient) { // Compile-time switch
        gradient -= hXjY[index];
    }

    if (BaseModel::precomputeHessian) { // Compile-time switch
        hessian += static_cast<RealType>(2.0) * hXjX[index];
    }

    *ogradient = static_cast<double>(gradient);
    *ohessian = static_cast<double>(hessian);

    // std::cerr << index << " " << gradient << " " << hessian << std::endl;

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    auto name = "compGradHessMM" + IteratorType::name + "";
    duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif
}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::computeGradientAndHessianImpl(int index, double *ogradient,
		double *ohessian, Weights w) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	RealType gradient = static_cast<RealType>(0);
	RealType hessian = static_cast<RealType>(0);

    if (BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

        if (sparseIndices[index] == nullptr || sparseIndices[index]->size() > 0) {

    	    IteratorType it(sparseIndices[index].get(), N);

    	    RealType accNumerPid  = static_cast<RealType>(0);
            RealType accNumerPid2  = static_cast<RealType>(0);
            RealType decNumerPid = static_cast<RealType>(0); // ESK: Competing risks contrib to gradient
            RealType decNumerPid2 = static_cast<RealType>(0); // ESK: Competing risks contrib to hessian

            //computeBackwardAccumlatedNumerator(it, Weights::isWeighted); // Perform inside loop rather than outside.

    	    // find start relavent accumulator reset point
    	    auto reset = begin(accReset);
    	    while( *reset < it.index() ) {
    	        ++reset;
    	    }

    	    for (; it; ) {
    	        int i = it.index();

                if (*reset <= i) {
    	            accNumerPid  = static_cast<RealType>(0.0);
    	            accNumerPid2 = static_cast<RealType>(0.0);
    	            ++reset;
    	        }

    	        const auto numerator1 = numerPid[i];
                const auto numerator2 = numerPid2[i];

    	        accNumerPid += numerator1;
    	        accNumerPid2 += numerator2;

                // Compile-time delegation
                BaseModel::incrementGradientAndHessian(it,
                        w, // Signature-only, for iterator-type specialization
                        &gradient, &hessian, accNumerPid, accNumerPid2,
                        accDenomPid[i], hNWeight[i], 0.0, hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments
                ++it;

    	        if (IteratorType::isSparse) {

    	            const int next = it ? it.index() : N;
                    for (++i; i < next; ++i) {

                        if (*reset <= i) {
    	                    accNumerPid  = static_cast<RealType>(0.0);
    	                    accNumerPid2 = static_cast<RealType>(0.0);
    	                    ++reset;
    	                }

    	                BaseModel::incrementGradientAndHessian(it,
                                w, // Signature-only, for iterator-type specialization
    	                        &gradient, &hessian, accNumerPid, accNumerPid2,
    	                        accDenomPid[i], hNWeight[i], static_cast<RealType>(0), hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments
    	            }
    	        }
    	    }
            if (BaseModel::isTwoWayScan) {
    	        // Manually perform backwards accumulation here instead of a separate function.
                auto revIt = it.reverse();

                // segmented prefix-scan
                RealType totalNumer = static_cast<RealType>(0);
                RealType totalNumer2 = static_cast<RealType>(0);

                auto backReset = end(accReset) - 1;

                for ( ; revIt; ) {

                    int i = revIt.index();

                    if (static_cast<signed int>(*backReset) == i) {
                        totalNumer = static_cast<RealType>(0);
                        totalNumer2 = static_cast<RealType>(0);
                        --backReset;
                    }

                    totalNumer += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? numerPid[i] / hYWeight[hNtoK[i]] : 0;
                    totalNumer2 += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? numerPid2[i] / hYWeight[hNtoK[i]]: 0;
                    decNumerPid = (hY[hNtoK[i]] == static_cast<RealType>(1)) ?
                                  hYWeight[hNtoK[i]] * totalNumer : 0;
                    decNumerPid2 = (hY[hNtoK[i]] == static_cast<RealType>(1)) ?
                                   hYWeight[hNtoK[i]] * totalNumer2 : 0;


                 // Increase gradient and hessian by competing risks contribution
                    BaseModel::incrementGradientAndHessian(it,
                                                           w, // Signature-only, for iterator-type specialization
                                                           &gradient, &hessian, decNumerPid, decNumerPid2,
                                                           accDenomPid[i], hNWeight[i], static_cast<RealType>(0), hXBeta[i],
                                                           hY[i]); // When function is in-lined, compiler will only use necess
                 --revIt;

                    if (IteratorType::isSparse) {

                        const int next = revIt ? revIt.index() : -1;
                        for (--i; i > next; --i) {
                            //if (*reset <= i) { // TODO MAS: This is not correct (only need for stratified models)
                            //    accNumerPid  = static_cast<RealType>(0);
                            //    accNumerPid2 = static_cast<RealType>(0);
                            //    ++reset;
                            //}

                           decNumerPid = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer : 0;
                           decNumerPid2 = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer2 : 0;
                           BaseModel::incrementGradientAndHessian(it, w, // Signature-only, for iterator-type specialization
                                                                   &gradient, &hessian, decNumerPid, decNumerPid2,
                                                                   accDenomPid[i], hNWeight[i], static_cast<RealType>(0),
                                                                   hXBeta[i],
                                                                   hY[i]); // When function is in-lined, compiler will only use necess
                        }
                    }
                } // End two-way scan
            } // End backwards iterator
		}

	} else if (BaseModel::hasIndependentRows) {

	    IteratorType it(hX, index);

	    for (; it; ++it) {
	        const int i = it.index();

	        RealType numerator1 = BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[i], hXBeta[i], hY[i]);
	        RealType numerator2 = (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) ?
	                BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[i]) : static_cast<RealType>(0);

	        // Compile-time delegation
	        BaseModel::incrementGradientAndHessian(it,
                    w, // Signature-only, for iterator-type specialization
                    &gradient, &hessian, numerator1, numerator2,
                    denomPid[i], hNWeight[i], it.value(), hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments
	    }

	} else if (BaseModel::exactCLR) {
	    //tbb::mutex mutex0;

#ifdef USE_RCPP_PARALLEL

	    tbb::combinable<RealType> newGrad(static_cast<RealType>(0));
	    tbb::combinable<RealType> newHess(static_cast<RealType>(0));

	    auto func = [&,index](const tbb::blocked_range<int>& range){

	        using std::isinf;

	        for (int i = range.begin(); i < range.end(); ++i) {
	            DenseView<IteratorType, RealType> x(IteratorType(hX, index), hNtoK[i], hNtoK[i+1]);
	            int numSubjects = hNtoK[i+1] - hNtoK[i];
	            int numCases = hNWeight[i];
	            std::vector<RealType> value = computeHowardRecursion<RealType>(offsExpXBeta.begin() + hNtoK[i], x, numSubjects, numCases);
	            if (value[0]==0 || value[1] == 0 || value[2] == 0 || isinf(value[0]) || isinf(value[1]) || isinf(value[2])) {
	                DenseView<IteratorType, RealType> newX(IteratorType(hX, index), hNtoK[i], hNtoK[i+1]);
	                std::vector<DDouble> value = computeHowardRecursion<DDouble>(offsExpXBeta.begin() + hNtoK[i], newX, numSubjects, numCases);//, threadPool);//, hY.begin() + hNtoK[i]);
	                using namespace sugar;
	                //mutex0.lock();
	                newGrad.local() -= (RealType)(-value[1]/value[0]);
	                newHess.local() -= (RealType)((value[1]/value[0]) * (value[1]/value[0]) - value[2]/value[0]);
	                //mutex0.unlock();
	                continue;
	            }
	            //mutex0.lock();
	            newGrad.local() -= (RealType)(-value[1]/value[0]);
	            newHess.local() -= (RealType)((value[1]/value[0]) * (value[1]/value[0]) - value[2]/value[0]);
	            //mutex0.unlock();
	        }
	    };
	    tbb::parallel_for(tbb::blocked_range<int>(0,N),func);
	    gradient += newGrad.combine([](const RealType& x, const RealType& y) {return x+y;});
	    hessian += newHess.combine([](const RealType& x, const RealType& y) {return x+y;});
#else

	    using std::isinf;

	    for (int i=0; i<N; i++) {
	    DenseView<IteratorType, RealType> x(IteratorType(hX, index), hNtoK[i], hNtoK[i+1]);
	    int numSubjects = hNtoK[i+1] - hNtoK[i];
	    int numCases = hNWeight[i];

	    std::vector<RealType> value = computeHowardRecursion<RealType>(offsExpXBeta.begin() + hNtoK[i], x, numSubjects, numCases);//, threadPool);//, hY.begin() + hNtoK[i]);
	    //std::cout<<" values" << i <<": "<<value[0]<<" | "<<value[1]<<" | "<<value[2];
	    if (value[0]==0 || value[1] == 0 || value[2] == 0 || isinf(value[0]) || isinf(value[1]) || isinf(value[2])) {
	    DenseView<IteratorType, RealType> newX(IteratorType(hX, index), hNtoK[i], hNtoK[i+1]);
	    std::vector<DDouble> value = computeHowardRecursion<DDouble>(offsExpXBeta.begin() + hNtoK[i], newX, numSubjects, numCases);//, threadPool);//, hY.begin() + hNtoK[i]);
	    using namespace sugar;
	    gradient -= (RealType)(-value[1]/value[0]);
	    hessian -= (RealType)((value[1]/value[0]) * (value[1]/value[0]) - value[2]/value[0]);
	    continue;
	    }
	    //gradient -= (RealType)(value[3] - value[1]/value[0]);
	    gradient -= (RealType)(-value[1]/value[0]);
	    hessian -= (RealType)((value[1]/value[0]) * (value[1]/value[0]) - value[2]/value[0]);
	    }
#endif // USE_RCPP_PARALLEL

	} else {

// 	    auto rangeKey = helper::dependent::getRangeKey(
// 	        hX, index, hPid,
//             typename IteratorType::tag());
//
// 	    auto rangeXNumerator = helper::dependent::getRangeX(
// 	        hX, index, offsExpXBeta,
//             typename IteratorType::tag());
//
// 	    auto rangeGradient = helper::dependent::getRangeGradient(
// 	        sparseIndices[index].get(), N, // runtime error: reference binding to null pointer of type 'struct vector'
//             denomPid, hNWeight,
//             typename IteratorType::tag());
//
// 	    const auto result = variants::trial::nested_reduce(
// 	        rangeKey.begin(), rangeKey.end(), // key
// 	        rangeXNumerator.begin(), // inner
// 	        rangeGradient.begin(), // outer
// 	        std::pair<RealType,RealType>{0,0}, Fraction<RealType>{0,0},
// 	        TestNumeratorKernel<BaseModel,IteratorType,RealType>(), // Inner transform-reduce
// 	        TestGradientKernel<BaseModel,IteratorType,Weights,RealType>()); // Outer transform-reduce
//
// 	    gradient = result.real();
// 	    hessian = result.imag();

	    IteratorType it(hX, index);
	    const int end = it.size() - 1;

	    RealType numerator1 = static_cast<RealType>(0);
	    RealType numerator2 = static_cast<RealType>(0);
	    int key = hPid[it.index()];

// 	    auto incrementNumerators2 = [this](const typename IteratorType::Index i, const typename IteratorType::ValueType x,
//                                            const RealTypePair lhs) -> RealTypePair {
//
// 	        const auto linearPredictor = offsExpXBeta[i];
// 	        return {
//     	       lhs.first + BaseModel::gradientNumeratorContrib(x, linearPredictor, static_cast<RealType>(0), static_cast<RealType>(0)),
// 	           lhs.second + (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms ?
// 	                BaseModel::gradientNumerator2Contrib(x, linearPredictor) :
// 	                static_cast<RealType>(0))
// 	        };
// 	    };

	    auto incrementNumerators = [this,&it,&numerator1,&numerator2]() {
	        const int i = it.index();

	        numerator1 += BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[i], static_cast<RealType>(0), static_cast<RealType>(0));
	        numerator2 += (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) ?
	        BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[i]) :
	            static_cast<RealType>(0);
	    };

	    for ( ; it.inRange(end); ++it) {
	        incrementNumerators();

	        const int nextKey = hPid[it.nextIndex()];
	        if (key != nextKey) {

	            BaseModel::incrementGradientAndHessian(it, w, // Signature-only, for iterator-type specialization
                                                    &gradient, &hessian, numerator1, numerator2,
                                                    denomPid[key], hNWeight[key], 0, 0, 0); // When function is in-lined, compiler will only use necessary arguments

	            numerator1 = static_cast<RealType>(0);
	            numerator2 = static_cast<RealType>(0);

	            key = nextKey;
	        }
	    }

	    incrementNumerators();

	    BaseModel::incrementGradientAndHessian(it, w, // Signature-only, for iterator-type specialization
                                            &gradient, &hessian, numerator1, numerator2,
                                            denomPid[key], hNWeight[key], 0, 0, 0); // When function is in-lined, compiler will only use necessary arguments
	}

	if (BaseModel::precomputeGradient) { // Compile-time switch
		gradient -= hXjY[index];
	}

	if (BaseModel::precomputeHessian) { // Compile-time switch
		hessian += static_cast<RealType>(2.0) * hXjX[index];
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

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeThirdDerivative(int index, double *othird, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
    auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

    if (hX.getNumberOfNonZeroEntries(index) == 0) {
        *othird = 0.0;
        return;
    }

    // Run-time dispatch, so virtual call should not effect speed
    if (useWeights) {
        switch (hX.getFormatType(index)) {
        case INDICATOR :
            computeThirdDerivativeImpl<IndicatorIterator<RealType>>(index, othird, weighted);
            break;
        case SPARSE :
            computeThirdDerivativeImpl<SparseIterator<RealType>>(index, othird, weighted);
            break;
        case DENSE :
            computeThirdDerivativeImpl<DenseIterator<RealType>>(index, othird, weighted);
            break;
        case INTERCEPT :
            computeThirdDerivativeImpl<InterceptIterator<RealType>>(index, othird, weighted);
            break;
        }
    } else {
        switch (hX.getFormatType(index)) {
        case INDICATOR :
            computeThirdDerivativeImpl<IndicatorIterator<RealType>>(index, othird, unweighted);
            break;
        case SPARSE :
            computeThirdDerivativeImpl<SparseIterator<RealType>>(index, othird, unweighted);
            break;
        case DENSE :
            computeThirdDerivativeImpl<DenseIterator<RealType>>(index, othird, unweighted);
            break;
        case INTERCEPT :
            computeThirdDerivativeImpl<InterceptIterator<RealType>>(index, othird, unweighted);
            break;
        }
    }

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    duration["compThird  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif
}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::computeThirdDerivativeImpl(int index, double *othird, Weights w) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
    auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

    RealType third = static_cast<RealType>(0);

    if (BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

        if (sparseIndices[index] == nullptr || sparseIndices[index]->size() > 0) {

            IteratorType it(sparseIndices[index].get(), N);

            RealType accNumerPid  = static_cast<RealType>(0);
            RealType accNumerPid2  = static_cast<RealType>(0);

            // find start relavent accumulator reset point
            auto reset = begin(accReset);
            while( *reset < it.index() ) {
                ++reset;
            }

            for (; it; ) {
                int i = it.index();

                if (*reset <= i) {
                    accNumerPid  = static_cast<RealType>(0.0);
                    accNumerPid2 = static_cast<RealType>(0.0);
                    ++reset;
                }

                const auto numerator1 = numerPid[i];
                const auto numerator2 = numerPid2[i];

                accNumerPid += numerator1;
                accNumerPid2 += numerator2;

                // Compile-time delegation
                BaseModel::incrementThirdDerivative(it,
                                                    w, // Signature-only, for iterator-type specialization
                                                    &third, accNumerPid, accNumerPid2,
                                                    accDenomPid[i], hNWeight[i], 0.0, hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments
                ++it;

                if (IteratorType::isSparse) {

                    const int next = it ? it.index() : N;
                    for (++i; i < next; ++i) {

                        if (*reset <= i) {
                            accNumerPid  = static_cast<RealType>(0.0);
                            accNumerPid2 = static_cast<RealType>(0.0);
                            ++reset;
                        }

                        BaseModel::incrementThirdDerivative(it,
                                                            w, // Signature-only, for iterator-type specialization
                                                            &third, accNumerPid, accNumerPid2,
                                                            accDenomPid[i], hNWeight[i], static_cast<RealType>(0), hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments
                    }
                }
            }
        } else {
    		throw new std::logic_error("Not yet support");
        }
    }

    *othird = static_cast<double>(third);

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    auto name = "compThird" + IteratorType::name + "  ";
    duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeFisherInformation(int indexOne, int indexTwo,
		double *oinfo, bool useWeights) {

	if (useWeights) {
		throw new std::logic_error("Weights are not yet implemented in Fisher Information calculations");
	} else { // no weights
		switch (hX.getFormatType(indexOne)) {
			case INDICATOR :
				dispatchFisherInformation<IndicatorIterator<RealType>>(indexOne, indexTwo, oinfo, weighted);
				break;
			case SPARSE :
				dispatchFisherInformation<SparseIterator<RealType>>(indexOne, indexTwo, oinfo, weighted);
				break;
			case DENSE :
				dispatchFisherInformation<DenseIterator<RealType>>(indexOne, indexTwo, oinfo, weighted);
				break;
			case INTERCEPT :
				dispatchFisherInformation<InterceptIterator<RealType>>(indexOne, indexTwo, oinfo, weighted);
				break;
		}
	}
}

template <class BaseModel, typename RealType> template <typename IteratorTypeOne, class Weights>
void ModelSpecifics<BaseModel,RealType>::dispatchFisherInformation(int indexOne, int indexTwo, double *oinfo, Weights w) {
	switch (hX.getFormatType(indexTwo)) {
		case INDICATOR :
			computeFisherInformationImpl<IteratorTypeOne,IndicatorIterator<RealType>>(indexOne, indexTwo, oinfo, w);
			break;
		case SPARSE :
			computeFisherInformationImpl<IteratorTypeOne,SparseIterator<RealType>>(indexOne, indexTwo, oinfo, w);
			break;
		case DENSE :
			computeFisherInformationImpl<IteratorTypeOne,DenseIterator<RealType>>(indexOne, indexTwo, oinfo, w);
			break;
		case INTERCEPT :
			computeFisherInformationImpl<IteratorTypeOne,InterceptIterator<RealType>>(indexOne, indexTwo, oinfo, w);
			break;
	}
//	std::cerr << "End of dispatch" << std::endl;
}


template<class BaseModel, typename RealType> template<class IteratorType>
SparseIterator<RealType> ModelSpecifics<BaseModel, RealType>::getSubjectSpecificHessianIterator(int index) {

	if (hessianSparseCrossTerms.find(index) == hessianSparseCrossTerms.end()) {

        auto indices = make_shared<std::vector<int> >();
        auto values = make_shared<std::vector<RealType> >();

    	CDCPtr column = bsccs::make_shared<CompressedDataColumn<RealType>>(indices, values, SPARSE);
		hessianSparseCrossTerms.insert(std::make_pair(index, column));

		IteratorType itCross(hX, index);
		for (; itCross;) {
			RealType value = 0.0;
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
	return SparseIterator<RealType>(*hessianSparseCrossTerms[index]);

}

template <class BaseModel, typename RealType> template <class IteratorTypeOne, class IteratorTypeTwo, class Weights>
void ModelSpecifics<BaseModel,RealType>::computeFisherInformationImpl(int indexOne, int indexTwo, double *oinfo, Weights w) {

	IteratorTypeOne itOne(hX, indexOne);
	IteratorTypeTwo itTwo(hX, indexTwo);
	PairProductIterator<IteratorTypeOne,IteratorTypeTwo,RealType> it(itOne, itTwo);

	RealType information = static_cast<RealType>(0);
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
			hessianCrossTerms[indexTwo].swap(crossTwoTerms);
		}
		std::vector<real>& crossTwoTerms = hessianCrossTerms[indexTwo];

		// TODO Sparse loop
		real cross = 0.0;
		for (int n = 0; n < N; ++n) {
			cross += crossOneTerms[n] * crossTwoTerms[n] / (denomPid[n] * denomPid[n]);
		}
		information -= cross;
#else
		SparseIterator<RealType> sparseCrossOneTerms = getSubjectSpecificHessianIterator<IteratorTypeOne>(indexOne);
		SparseIterator<RealType> sparseCrossTwoTerms = getSubjectSpecificHessianIterator<IteratorTypeTwo>(indexTwo);
		PairProductIterator<SparseIterator<RealType>,SparseIterator<RealType>,RealType> itSparseCross(sparseCrossOneTerms, sparseCrossTwoTerms);

		RealType sparseCross = static_cast<RealType>(0);
		for (; itSparseCross.valid(); ++itSparseCross) {
			const int n = itSparseCross.index();
			sparseCross += itSparseCross.value() / (denomPid[n] * denomPid[n]);
		}
		information -= sparseCross;
#endif
	}

	*oinfo = static_cast<double>(information);
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeNumeratorForGradient(int index, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	if (BaseModel::hasNtoKIndices || BaseModel::cumulativeGradientAndHessian) {

		switch (hX.getFormatType(index)) {
		case INDICATOR : {
				IndicatorIterator<RealType> itI(*(sparseIndices)[index]);
				for (; itI; ++itI) { // Only affected entries
					numerPid[itI.index()] = static_cast<RealType>(0.0);
				}
/*
				zeroVector(numerPid.data(), N);
				if (BaseModel::efron) {
					zeroVector(numerPid3.data(), N);
				}
*/
				if (useWeights) {
				    incrementNumeratorForGradientImpl<IndicatorIterator<RealType>, WeightedOperation>(index);
				} else {
				    incrementNumeratorForGradientImpl<IndicatorIterator<RealType>, UnweightedOperation>(index);
				}
				break;
		}
		case SPARSE : {
				SparseIterator<RealType> itS(*(sparseIndices)[index]);
				for (; itS; ++itS) { // Only affected entries
					numerPid[itS.index()] = static_cast<RealType>(0.0);
					if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
						numerPid2[itS.index()] = static_cast<RealType>(0.0); // TODO Does this invalid the cache line too much?
					}
				}
/*
				zeroVector(numerPid.data(), N);
				if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
					zeroVector(numerPid2.data(), N);
				}
				if (BaseModel::efron) {
					zeroVector(numerPid3.data(), N);
					zeroVector(numerPid4.data(), N);
				}
*/
				if (useWeights) {
				    incrementNumeratorForGradientImpl<SparseIterator<RealType>, WeightedOperation>(index);
				} else {
				    incrementNumeratorForGradientImpl<SparseIterator<RealType>, UnweightedOperation>(index);
				}
				break;
		}
		case DENSE : {
				zeroVector(numerPid.data(), N);
				if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
					zeroVector(numerPid2.data(), N);
				}
/*
				if (BaseModel::efron) {
					zeroVector(numerPid3.data(), N);
					zeroVector(numerPid4.data(), N);
				}
*/
				if (useWeights) {
				    incrementNumeratorForGradientImpl<DenseIterator<RealType>, WeightedOperation>(index);
				} else {
				    incrementNumeratorForGradientImpl<DenseIterator<RealType>, UnweightedOperation>(index);
				}
				break;
		}
		case INTERCEPT : {
				zeroVector(numerPid.data(), N);
				if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
					zeroVector(numerPid2.data(), N);
				}
/*
				if (BaseModel::efron) {
					zeroVector(numerPid3.data(), N);
					zeroVector(numerPid4.data(), N);
				}
*/
				if (useWeights) {
				    incrementNumeratorForGradientImpl<InterceptIterator<RealType>, WeightedOperation>(index);
				} else {
				    incrementNumeratorForGradientImpl<InterceptIterator<RealType>, UnweightedOperation>(index);
				}
				break;
		}
		default : break;
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compNumForGrad   "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
#endif

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	duration["CPU GH           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::incrementNumeratorForGradientImpl(int index) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	IteratorType it(hX, index);
	for (; it; ++it) {
		const int k = it.index();
		incrementByGroup(numerPid.data(), hPid, k,
                   Weights::isWeighted ?
                       hKWeight[k] * BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k]) :
		               BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k])
		    );
		if (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) {
			incrementByGroup(numerPid2.data(), hPid, k,
                    Weights::isWeighted ?
                        hKWeight[k] * BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k]) :
                        BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k])
                );
		}
/*
		if (BaseModel::efron) {
			incrementByGroup(numerPid3.data(), hPid, k,
					Weights::isWeighted ?
						hKWeight[k] * hY[k] * BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k]) :
						hY[k] * BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k])
				);
		}
		if (!IteratorType::isIndicator && BaseModel::efron) {
			incrementByGroup(numerPid4.data(), hPid, k,
                    Weights::isWeighted ?
                        hKWeight[k] * hY[k] * BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k]) :
                        hY[k] * BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k])
                );
		}
*/
	}
#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	auto name = "compNumGrad" + IteratorType::name + "   ";
	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::updateXBeta(double delta, int index, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	RealType realDelta = static_cast<RealType>(delta);

	// Run-time dispatch to implementation depending on covariate FormatType
	switch(hX.getFormatType(index)) {
	    case INDICATOR : {
		    if (useWeights) {
			    updateXBetaImpl<IndicatorIterator<RealType>, WeightedOperation>(realDelta, index);
		    } else {
		        updateXBetaImpl<IndicatorIterator<RealType>, UnweightedOperation>(realDelta, index);
		    }
			break;
	    }
	    case SPARSE : {
		    if (useWeights) {
			    updateXBetaImpl<SparseIterator<RealType>, WeightedOperation>(realDelta, index);
		    } else {
		        updateXBetaImpl<SparseIterator<RealType>, UnweightedOperation>(realDelta, index);
		    }
			break;
	    }
	    case DENSE : {
	        if (useWeights) {
			    updateXBetaImpl<DenseIterator<RealType>, WeightedOperation>(realDelta, index);
	        } else {
	            updateXBetaImpl<DenseIterator<RealType>, UnweightedOperation>(realDelta, index);
	        }
			break;
	    }
	    case INTERCEPT : {
	        if (useWeights) {
			    updateXBetaImpl<InterceptIterator<RealType>, WeightedOperation>(realDelta, index);
	        } else {
	            updateXBetaImpl<InterceptIterator<RealType>, UnweightedOperation>(realDelta, index);
	        }
			break;
	    }
		default : break;
	}

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["updateXBeta      "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	duration["CPU GH           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
inline void ModelSpecifics<BaseModel,RealType>::updateXBetaImpl(RealType realDelta, int index) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	IteratorType it(hX, index);

    if (BaseModel::cumulativeGradientAndHessian) { // cox
        for (; it; ++it) {
            const int k = it.index();
            hXBeta[k] += realDelta * it.value(); // TODO Check optimization with indicator and intercept

            if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
                RealType oldEntry = Weights::isWeighted ?
                    hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k]; // TODO Delegate condition to forming offExpXBeta
                offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k); // Update offsExpXBeta
                RealType newEntry = Weights::isWeighted ?
                    hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k]; // TODO Delegate condition
                incrementByGroup(denomPid.data(), hPid, k, (newEntry - oldEntry)); // Update denominators
            }
        }
    } else {
        for (; it; ++it) {
            const int k = it.index();
            hXBeta[k] += realDelta * it.value(); // TODO Check optimization with indicator and intercept

            if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
                RealType oldEntry = offsExpXBeta[k];
                RealType newEntry = offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k);
                incrementByGroup(denomPid.data(), hPid, k, (newEntry - oldEntry));
/*
                if (BaseModel::efron) {
                    incrementByGroup(denomPid2.data(), hPid, k, hY[k]*(newEntry - oldEntry)); // Update denominators
                }
*/
            }
        }
    }

	computeAccumlatedDenominator(Weights::isWeighted);

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	auto name = "updateXBeta" + IteratorType::name + "   ";
	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel, typename RealType> template <class Weights>
void ModelSpecifics<BaseModel,RealType>::computeRemainingStatisticsImpl() {

    auto& xBeta = getXBeta();

    if (BaseModel::likelihoodHasDenominator) {
        fillVector(denomPid.data(), N, BaseModel::getDenomNullValue());
/*
        if (BaseModel::efron) {
        	fillVector(denomPid2.data(), N, BaseModel::getDenomNullValue());
        }
*/
        if (BaseModel::cumulativeGradientAndHessian) { // cox
            for (size_t k = 0; k < K; ++k) {
                offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k);
                RealType weightoffsExpXBeta =  Weights::isWeighted ?
                    hKWeight[k] * BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k) :
                    BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k);
                incrementByGroup(denomPid.data(), hPid, k, weightoffsExpXBeta); // Update denominators
            }
        } else {
            for (size_t k = 0; k < K; ++k) {
                offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k);
                incrementByGroup(denomPid.data(), hPid, k, offsExpXBeta[k]);
/*
                if (BaseModel::efron) {
                    incrementByGroup(denomPid2.data(), hPid, k, hY[k]*offsExpXBeta[k]); // Update denominators
                }
*/
            }
        }
        computeAccumlatedDenominator(Weights::isWeighted);
    }
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeRemainingStatistics(bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
	auto start = bsccs::chrono::steady_clock::now();
#endif

	if (useWeights) {
	    computeRemainingStatisticsImpl<WeightedOperation>();
	} else {
	    computeRemainingStatisticsImpl<UnweightedOperation>();
	}

#ifdef CYCLOPS_DEBUG_TIMING
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compRS           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeAccumlatedNumerator(bool useWeights) {

	if (BaseModel::likelihoodHasDenominator && //The two switches should ideally be separated
			BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

		if (accNumerPid.size() != N) {
			accNumerPid.resize(N, static_cast<RealType>(0));
		}
		if (accNumerPid2.size() != N) {
			accNumerPid2.resize(N, static_cast<RealType>(0));
		}

		// segmented prefix-scan
		RealType totalNumer = static_cast<RealType>(0);
		RealType totalNumer2 = static_cast<RealType>(0);

		auto reset = begin(accReset);

		for (size_t i = 0; i < N; ++i) {

			if (static_cast<unsigned int>(*reset) == i) {
				totalNumer = static_cast<RealType>(0);
				totalNumer2 = static_cast<RealType>(0);
				++reset;
			}

			totalNumer += numerPid[i];
			totalNumer2 += numerPid2[i];
			accNumerPid[i] = totalNumer;
			accNumerPid2[i] = totalNumer2;
        }
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeAccumlatedDenominator(bool useWeights) {

    if (BaseModel::likelihoodHasDenominator && //The two switches should ideally be separated
        BaseModel::cumulativeGradientAndHessian) { // Compile-time switch


        if (accDenomPid.size() != (N + 1)) {
            accDenomPid.resize(N + 1, static_cast<RealType>(0));
        }

        // segmented prefix-scan
        RealType totalDenom = static_cast<RealType>(0);

        auto reset = begin(accReset);

        for (size_t i = 0; i < N; ++i) {

            if (static_cast<unsigned int>(*reset) == i) { // TODO Check with SPARSE
                totalDenom = static_cast<RealType>(0);
                ++reset;
            }

            totalDenom += denomPid[i];
            accDenomPid[i] = totalDenom;
        }
        //ESK : Incorporate backwards scan here:
        if (BaseModel::isTwoWayScan) {
            // RealType backDenom = static_cast<RealType>(0);
            RealType totalDenom = static_cast<RealType>(0);

        auto reset = begin(accReset);

            //Q: How can we change int to size_t w/o errors
            for (int i = (N - 1); i >= 0; i--) {

                if (static_cast<unsigned int>(*reset) == i) { // TODO Check with SPARSE
                    totalDenom = static_cast<RealType>(0);
                    ++reset;
                }

                totalDenom += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? denomPid[i] / hYWeight[hNtoK[i]] : 0;
                accDenomPid[i] += (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalDenom : 0;
            }
        } // End two-way scan
    }

}

// ESK: This is no longer needed. Incorporated in incrementGradientAndHessian
template <class BaseModel,typename RealType> template <class IteratorType>
void ModelSpecifics<BaseModel,RealType>::computeBackwardAccumlatedNumerator(
        IteratorType it,
        bool useWeights) {

    if (decNumerPid.size() != N) {
        decNumerPid.resize(N, static_cast<RealType>(0));
    }
    if (decNumerPid2.size() != N) {
        decNumerPid2.resize(N, static_cast<RealType>(0));
    }
    auto revIt = it.reverse();

    // segmented prefix-scan
    RealType totalNumer = static_cast<RealType>(0);
    RealType totalNumer2 = static_cast<RealType>(0);

    auto reset = end(accReset) - 1;

    for ( ; revIt; ) {

        int i = revIt.index();

        if (static_cast<signed int>(*reset) == i) {
            totalNumer = static_cast<RealType>(0);
            totalNumer2 = static_cast<RealType>(0);
            --reset;
        }

        totalNumer +=  (hY[hNtoK[i]] > static_cast<RealType>(1)) ? numerPid[i] / hYWeight[hNtoK[i]] : 0;
        totalNumer2 += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? numerPid2[i] / hYWeight[hNtoK[i]] : 0;
        decNumerPid[i] = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer : 0;
        decNumerPid2[i] = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer2 : 0;
        --revIt;

        if (IteratorType::isSparse) {

            const int next = revIt ? revIt.index() : -1;
            for (--i; i > next; --i) { // TODO MAS: This may be incorrect
                //if (*reset <= i) { // TODO MAS: This is not correct (only need for stratifed models)
                //    accNumerPid  = static_cast<RealType>(0);
                //    accNumerPid2 = static_cast<RealType>(0);
                //    ++reset;
                //}

                // ESK: Start implementing sparse
                decNumerPid[i] = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer : 0;
                decNumerPid2[i] = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer2 : 0;
            }
        }
    }
}

// ESK: This is no longer needed. Incorporated in computeAccumlatedDenominator
template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeBackwardAccumlatedDenominator(bool useWeights) {

    if (BaseModel::likelihoodHasDenominator && //The two switches should ideally be separated
            BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

        // segmented prefix-scan
        RealType backDenom = static_cast<RealType>(0);
        RealType totalDenom = static_cast<RealType>(0);

        auto reset = begin(accReset);

        //Q: How can we change int to size_t w/o errors
        for (int i = (N - 1); i >= 0; i--) {

            if (static_cast<unsigned int>(*reset) == i) { // TODO Check with SPARSE
                totalDenom = static_cast<RealType>(0);
                ++reset;
            }

            totalDenom += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? denomPid[i] / hYWeight[hNtoK[i]] : 0;
            accDenomPid[i] += (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalDenom : 0;
        }
    }
}


template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::doSortPid(bool useCrossValidation) {
/* For Cox model:
 *
 * We currently assume that hZ[k] are sorted in decreasing order by k.
 *
 */
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::getOriginalPid() {

    hPidInternal =  hPidOriginal; // Make copy

}
template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::setPidForAccumulation(const double* weights) {
	setPidForAccumulationImpl(weights);
}

template <class BaseModel,typename RealType> template <typename AnyRealType>
void ModelSpecifics<BaseModel,RealType>::setPidForAccumulationImpl(const AnyRealType* weights) {

    hPidInternal =  hPidOriginal; // Make copy
    hPid = hPidInternal.data(); // Point to copy
    hPidSize = hPidInternal.size();
    accReset.clear();

    const int ignore = -1;

    // Find first non-zero weight
    size_t index = 0;
    while(weights != nullptr && weights[index] == 0.0 && index < K) {
        hPid[index] = ignore;
        index++;
    }

    int lastPid = hPid[index];
    AnyRealType lastTime = hOffs[index];
    AnyRealType lastEvent = hY[index];

    int pid = hPid[index] = 0;

    for (size_t k = index + 1; k < K; ++k) {
        if (weights == nullptr || weights[k] != 0.0) {
            int nextPid = hPid[k];

            if (nextPid != lastPid) { // start new strata
                pid++;
                accReset.push_back(pid);
                lastPid = nextPid;
            } else {

                if (lastEvent == 1.0 && lastTime == hOffs[k] && lastEvent == hY[k]) {
                    // In a tie, do not increment denominator
                } else {
                    pid++;
                }
            }
            lastTime = hOffs[k];
            lastEvent = hY[k];

            hPid[k] = pid;
        } else {
            hPid[k] = ignore;
        }
    }
    pid++;
    accReset.push_back(pid);

    // Save number of denominators
    N = pid;

    if (weights != nullptr) {
        for (size_t i = 0; i < K; ++i) {
            if (hPid[i] == ignore) hPid[i] = N; // do NOT accumulate, since loops use: i < N
        }
    }
    setupSparseIndices(N); // ignore pid == N (pointing to removed data strata)
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::setupSparseIndices(const int max) {
    sparseIndices.clear(); // empty if full!

    for (size_t j = 0; j < J; ++j) {
        if (hX.getFormatType(j) == DENSE || hX.getFormatType(j) == INTERCEPT) {
            sparseIndices.push_back(NULL);
        } else {
            std::set<int> unique;
            const size_t n = hX.getNumberOfEntries(j);
            const int* indicators = hX.getCompressedColumnVector(j);
            for (size_t j = 0; j < n; j++) { // Loop through non-zero entries only
                const int k = indicators[j];
                const int i = (k < hPidSize) ? hPid[k] : k;
                if (i < max) {
                    unique.insert(i);
                }
            }
            auto indices = bsccs::make_shared<IndexVector>(unique.begin(), unique.end());
            sparseIndices.push_back(indices);
        }
    }
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::deviceInitialization() {
    // Do nothing
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::initialize(
        int iN,
        int iK,
        int iJ,
        const void*,
        double* iNumerPid,
        double* iNumerPid2,
        double* iDenomPid,
        double* iXjY,
        std::vector<std::vector<int>* >* iSparseIndices,
        const int* iPid_unused,
        double* iOffsExpXBeta,
        double* iXBeta,
        double* iOffs,
        double* iBeta,
        const double* iY_unused) {

    N = iN;
    K = iK;
    J = iJ;
    offsExpXBeta.resize(K);
    hXBeta.resize(K);

    if (allocateXjY()) {
        hXjY.resize(J);
    }

    if (allocateXjX()) {
        hXjX.resize(J);
    }
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = std::chrono::steady_clock::now();
#endif
    if (initializeAccumulationVectors()) {
        setPidForAccumulation(static_cast<double*>(nullptr)); // calls setupSparseIndices() before returning
    } else {
        // TODO Suspect below is not necessary for non-grouped data.
        // If true, then fill with pointers to CompressedDataColumn and do not delete in destructor
        setupSparseIndices(N); // Need to be recomputed when hPid change!
    }
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = std::chrono::steady_clock::now();
	double timerPid = std::chrono::duration<double>(end - start).count();
	std::cout << " OVERHEAD CCD setPid:  " << timerPid << " s \n";
#endif


    size_t alignedLength = getAlignedLength(N + 1);
    // 	numerDenomPidCache.resize(3 * alignedLength, 0);
    // 	numerPid = numerDenomPidCache.data();
    // 	denomPid = numerPid + alignedLength; // Nested in denomPid allocation
    // 	numerPid2 = numerPid + 2 * alignedLength;
    denomPid.resize(alignedLength);
    denomPid2.resize(alignedLength);
    numerPid.resize(alignedLength);
    numerPid2.resize(alignedLength);
    numerPid3.resize(alignedLength);
    numerPid4.resize(alignedLength);

    deviceInitialization();

}

} // namespace

#endif /* MODELSPECIFICS_HPP_ */
