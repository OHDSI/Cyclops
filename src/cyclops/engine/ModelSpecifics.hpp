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
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <limits>
#include <deque>
#include <unordered_map>

#include "ModelSpecifics.h"
#include "Iterators.h"

#include "Recursions.hpp"
#include "ParallelLoops.h"
// #include "Ranges.h"

#ifdef USE_RCPP_PARALLEL
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#endif

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

// exp(x) for x clamped to [-708, 709], within 1 ulp of std::exp: x = k ln(2) + r with |r| <= ln(2) / 2
// (Cody-Waite), a degree-13 Taylor polynomial in r, times 2^k set in the exponent bits. Branch-free,
// so loops over it vectorize.
template <typename RealType>
inline RealType vectorizableExp(const RealType value) {
    const double x = std::min(std::max(static_cast<double>(value), -708.0), 709.0);
    const double invLn2 = 1.4426950408889634, ln2Hi = 6.93147180369123816490e-01, ln2Lo = 1.90821492927058770002e-10;
    const double k = std::rint(x * invLn2);
    const double r = (x - k * ln2Hi) - k * ln2Lo;
    double q = 1.0 / 6227020800.0;
    q = q * r + 1.0 / 479001600.0;
    q = q * r + 1.0 / 39916800.0;
    q = q * r + 1.0 / 3628800.0;
    q = q * r + 1.0 / 362880.0;
    q = q * r + 1.0 / 40320.0;
    q = q * r + 1.0 / 5040.0;
    q = q * r + 1.0 / 720.0;
    q = q * r + 1.0 / 120.0;
    q = q * r + 1.0 / 24.0;
    q = q * r + 1.0 / 6.0;
    q = q * r + 0.5;
    q = q * r + 1.0;
    q = q * r + 1.0;
    const uint64_t bits = static_cast<uint64_t>(static_cast<int64_t>(k) + 1023) << 52;
    double scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return static_cast<RealType>(q * scale);
}

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
	auto copy = new ModelSpecifics<BaseModel,RealType>(modelData);
	copy->setLegacy(legacy); // cross-validation threads must run the same kernels
	return copy;
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::getGradientObjective(bool useCrossValidation) {

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

		if (isCUDA()) {
			getXBeta(); // copies the device xBeta into hXBeta
		}
		const auto& xBeta = hXBeta; // no copy

		double criterion = 0; // in double for any RealType: in float the sum of K terms stalls
		if (useCrossValidation) {
			for (int i = 0; i < K; i++) {
				criterion += static_cast<double>(xBeta[i]) * hY[i] * hKWeight[i]; // in double, as before
			}
		} else {
			for (int i = 0; i < K; i++) {
				criterion += static_cast<double>(xBeta[i]) * hY[i];
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

	trainingRowsOnly = false;
	activePartition.reset(); // setFoldPartition() follows
	if (BaseModel::hasLogisticRows && !legacy && heldOutRowsUnused && useCrossValidation) {
		bool binary = true;
		for (size_t k = 0; k < K && binary; ++k) {
			binary = inWeights[k] == 0.0 || inWeights[k] == 1.0;
		}
		if (binary) {
			trainingRow.resize(K);
			for (size_t k = 0; k < K; ++k) {
				trainingRow[k] = inWeights[k] != 0.0;
			}
			trainingRowsOnly = true;
		}
	}

#ifdef DEBUG_COX
	cerr << "Done with set weights" << endl;
#endif

}

template<class BaseModel, typename RealType>
void ModelSpecifics<BaseModel, RealType>::computeXjY(bool useCrossValidation) {
	if (BaseModel::hasLogisticRows && useCrossValidation && activePartition) {
		for (size_t j = 0; j < J; ++j) {
			switch (hX.getFormatType(j)) {
				case INDICATOR : hXjY[j] = computeXjYOverTrainingEntries<IndicatorIterator<RealType>>(j); break;
				case SPARSE : hXjY[j] = computeXjYOverTrainingEntries<SparseIterator<RealType>>(j); break;
				case DENSE : hXjY[j] = computeXjYOverTrainingEntries<DenseIterator<RealType>>(j); break;
				case INTERCEPT : hXjY[j] = computeXjYOverTrainingEntries<InterceptIterator<RealType>>(j); break;
			}
		}
		return;
	}
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

	ensureDenominators();

	double logLikelihood = 0.0; // in double for any RealType
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

	ensureDenominators();

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
	} else if (BaseModel::hasLogisticRows && !legacy) {
	    // Bernoulli log-likelihood y log p + (1 - y) log(1 - p) = y xBeta - log(1 + exp(xBeta)), in double
	    // and from xBeta, so saturated rows stay finite; rows with weight 0 are skipped. The legacy score,
	    // kept by createControl(fitter = "legacy"), is y log p: the held-out controls contribute nothing.
	    double sum = 0.0;
	    for (size_t k = 0; k < K; ++k) {
	        if (weights[k] != 0.0) {
	            const double xb = static_cast<double>(hXBeta[k]);
	            sum += weights[k] * (hY[k] * xb - (std::max(xb, 0.0) + std::log1p(std::exp(-std::abs(xb)))));
	        }
	    }
	    return sum;
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


template <class RealType>
void makeDenseCrossProduct(const CompressedDataMatrix<RealType>& x,
                                                int index1, int index2) {

}

template <class BaseModel, typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::getSchoenfeldResidualsImpl(int index,
                                                                    std::vector<double>* residuals,
                                                                    std::vector<double>* times,
                                                                    std::vector<int>* strata,
                                                                    double* covariate,
                                                                    double* score,
                                                                    Weights w) {

    const bool hasResiduals = residuals != nullptr;
    const bool hasTimes = times != nullptr;
    const bool hasStrata = strata != nullptr;
    const bool hasScore = score != nullptr && covariate != nullptr;

    if (hasResiduals) {
        residuals->clear();
    }
    if (hasTimes) {
        times->clear();
    }
    if (hasStrata) {
        strata->clear();
    }

    RealType uGradient = static_cast<RealType>(0); // unweighted
    RealType uHessian = static_cast<RealType>(0);
    RealType wGradient = static_cast<RealType>(0); // weighted
    RealType wHessian = static_cast<RealType>(0);
    RealType xHessian = static_cast<RealType>(0);

    RealType uGradient2 = static_cast<RealType>(0);
    RealType wGradient2 = static_cast<RealType>(0);

    // TODO: only written for accummulive models (Cox, Fine/Grey)

	IteratorType it(hX, index);

	RealType resNumerator  = static_cast<RealType>(0);
	// RealType resDenominator = static_cast<RealType>(0);
	RealType scoreNumerator1 = static_cast<RealType>(0);
	RealType scoreNumerator2 = static_cast<RealType>(0);
	RealType scoreDenominator = static_cast<RealType>(0);

	// find start relavent accumulator reset point
	auto reset = begin(accReset);
	// while( *reset < it.index() ) {
	// 	++reset;
	// }

	std::deque<RealType> deferredResidualX;
	std::deque<RealType> deferredScore1;
	std::deque<RealType> deferredScore2;
	std::deque<RealType> deferredScore3;
	std::deque<int> deferredScore4;

	auto processRow = [&](int i, RealType x) {

	    if (*reset <= hPid[i]) {

	        resNumerator = static_cast<RealType>(0);
	        // resDenominator = static_cast<RealType>(0);
	        scoreNumerator1 = static_cast<RealType>(0);
	        scoreNumerator2 = static_cast<RealType>(0);
	        scoreDenominator = static_cast<RealType>(0);

	        ++reset;
	    }

		const auto expXBeta = offsExpXBeta[i]; // std::exp(hXBeta[i]);

		resNumerator += hKWeight[i] * expXBeta * x;
		// resDenominator += expXBeta;

		if (hY[i] == 1) {
		    // std::cerr << " " << x << " for " << resNumerator << " / " << resDenominator << " - " <<  ( x - resNumerator / resDenominator) << "\n";
		    // std::cerr << denomPid[i] << "\n";
			if (hasResiduals) {
			    if (i < (K - 1) && hPid[i] < *reset &&
                    (hOffs[i] == hOffs[i+ 1]) && hY[i + 1] == 1) {
			        deferredResidualX.push_back(x);
			    } else {

			        const auto denom = accDenomPid[hPid[i]];

			        while (deferredResidualX.size() > 0) {
			            const auto residual = deferredResidualX.front() - resNumerator / denom;
			            residuals->push_back(residual);
			            deferredResidualX.pop_front();
			        }

			        const auto residual = x - resNumerator / denom;
				    residuals->push_back(residual);
			    }
			}
			if (hasTimes) {
				times->push_back(hOffs[i]);
			}
			if (hasStrata) {
				strata->push_back(hPidOriginal[i]);
			}
		} else {
		    // std::cerr << "\n";
		}

		if (hasScore) {
			const auto cov = covariate[i];
			const auto numerator1 = hKWeight[i] * expXBeta * x;
			const auto numerator2 = hKWeight[i] * expXBeta * x * x;

			if (hY[i] == 1) {
				uGradient2 += hKWeight[i] * x;
				wGradient2 += hKWeight[i] * x * cov;
			}

			scoreNumerator1 += numerator1;
			scoreNumerator2 += numerator2;

		    if (i < (K - 1) && hPid[i] == hPid[i + 1]) {
                // zero contribution under Breslow
		    } else {

			    const auto denom = accDenomPid[hPid[i]];

	    		const auto t = scoreNumerator1 / denom;
			    const auto gradient = hNWeight[hPid[i]] * t;
			    const auto hessian = hNWeight[hPid[i]] * (scoreNumerator2 / denom - t * t);

			    uGradient += gradient;
			    wGradient += gradient * cov;

			    // std::cerr << "newA" << i << " " << hPid[i] << " " << scoreNumerator1 << "/" << denom << " " << hY[i] << " " << uGradient << " " << hNWeight[hPid[i]] << "\n";

			    uHessian += hessian;
			    wHessian += hessian * cov * cov;
			    xHessian += hessian * cov;
		    }
		}
	};

	// main loop
	for (int i = 0; i < K; ++i) {
	    if (i == it.index()) {
	        processRow(i, it.value());
	        ++it;
	    } else {
	        processRow(i, static_cast<RealType>(0));
	    }
	}

    if (hasScore) {

        score[0] = static_cast<double>(uGradient2 - uGradient);
        score[1] = static_cast<double>(wGradient2 - wGradient);
        score[2] = static_cast<double>(uHessian);
        score[3] = static_cast<double>(xHessian);
        score[4] = static_cast<double>(xHessian);
        score[5] = static_cast<double>(wHessian);
     }
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

    ensureDenominators();

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

                // std::cerr << "oldA" << i << " " << accNumerPid << "/" << accDenomPid[i] << " " << hY[i] << " " << gradient << " " << hNWeight[i] << "\n";

                ++it;

    	        if (IteratorType::isSparse) {

    	            // TODO Should end be N or K?
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

                        // std::cerr << "oldB" << i << " " << accNumerPid << "/" << accDenomPid[i] << " " << hY[i] << " " << gradient << " " << hNWeight[i] << "\n";
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

	} else if (BaseModel::hasLogisticRows && !legacy) { // legacy: the 3.7 code below

	    // p = e / (1 + e) and w = p (1 - p) from e = offsExpXBeta alone (no denomPid gather); e = inf
	    // (xBeta > ~709) gives p = 1, w = 0 instead of NaN. As before, p rounds to exactly 1 and w to 0
	    // once 1 + e == e, so a separated covariate still ends in 0 / 0 (ILLCONDITIONED, NaN estimate),
	    // which is.nan(coef()) separation checks and createNonSeparablePrior() rely on.
	    IteratorType it(hX, index);
	    const int n = it.size();
	    const int* rows = IteratorType::isSparse ? hX.getCompressedColumnVector(index) : nullptr;
	    const RealType* values = IteratorType::isIndicator ? nullptr : hX.getDataVector(index);
	    const RealType* expXBeta = offsExpXBeta.data();
	    const RealType* weight = hNWeight.data();

	    if (Weights::isWeighted && activePartition) {
	        // 0/1 weights with a fold partition: the training entries are two contiguous ranges
	        const TrainingEntries t = getTrainingEntries<IteratorType>(index);
	        for (int r = 0; r < 4; r += 2) {
	            for (int q = t.range[r]; q < t.range[r + 1]; ++q) {
	                const int i = t.rows[q];
	                const RealType e = expXBeta[i];
	                const RealType p = (e <= std::numeric_limits<RealType>::max()) ?
	                    e / (static_cast<RealType>(1) + e) : static_cast<RealType>(1);
	                const RealType w = p * (static_cast<RealType>(1) - p);
	                if (IteratorType::isIndicator) {
	                    gradient += p;
	                    hessian += w;
	                } else {
	                    const RealType x = t.values[IteratorType::isSparse ? q : i];
	                    gradient += p * x;
	                    hessian += w * x * x;
	                }
	            }
	        }
	    } else if (Weights::isWeighted && trainingRowsOnly) {
	        // 0/1 weights: a 1-byte mask selects the training rows instead of multiplying by the weight
	        // (no branch: a held-out row is a random 10-33% of entries). Each sum adds the same terms in
	        // the same order as the weighted loop (0 * term == +-0), so the result is bitwise identical.
	        const uint8_t* train = trainingRow.data();
	        for (int q = 0; q < n; ++q) {
	            const int i = IteratorType::isSparse ? rows[q] : q;
	            const RealType e = expXBeta[i];
	            const RealType p = (e <= std::numeric_limits<RealType>::max()) ?
	                e / (static_cast<RealType>(1) + e) : static_cast<RealType>(1);
	            const RealType w = p * (static_cast<RealType>(1) - p);
	            if (IteratorType::isIndicator) {
	                gradient += train[i] ? p : static_cast<RealType>(0);
	                hessian += train[i] ? w : static_cast<RealType>(0);
	            } else {
	                const RealType x = values[q];
	                gradient += train[i] ? p * x : static_cast<RealType>(0);
	                hessian += train[i] ? w * x * x : static_cast<RealType>(0);
	            }
	        }
	    } else
	    for (int q = 0; q < n; ++q) {
	        const int i = IteratorType::isSparse ? rows[q] : q;
	        const RealType e = expXBeta[i];
	        const RealType p = (e <= std::numeric_limits<RealType>::max()) ?
	            e / (static_cast<RealType>(1) + e) : static_cast<RealType>(1);
	        const RealType w = p * (static_cast<RealType>(1) - p);
	        if (IteratorType::isIndicator) {
	            gradient += Weights::isWeighted ? weight[i] * p : p;
	            hessian += Weights::isWeighted ? weight[i] * w : w;
	        } else {
	            const RealType x = values[q];
	            gradient += Weights::isWeighted ? weight[i] * (p * x) : p * x;
	            hessian += Weights::isWeighted ? weight[i] * (w * x * x) : w * x * x;
	        }
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

	} else if (BaseModel::hasNtoKIndices && IteratorType::isSparse) { // one stratum per segment

	    IteratorType it(hX, index); // Signature-only, for iterator-type specialization
	    const int* rows = hX.getCompressedColumnVector(index);
	    const RealType* values = IteratorType::isIndicator ? nullptr : hX.getDataVector(index);
	    const StratumSegments& segments = stratumSegments[index];

	    for (size_t s = 0; s < segments.strata.size(); ++s) {
	        RealType numerator1 = static_cast<RealType>(0);
	        RealType numerator2 = static_cast<RealType>(0);

	        for (int i = segments.offsets[s]; i < segments.offsets[s + 1]; ++i) {
	            const RealType x = IteratorType::isIndicator ? static_cast<RealType>(1) : values[i];
	            const RealType predictor = offsExpXBeta[rows[i]];
	            numerator1 += BaseModel::gradientNumeratorContrib(x, predictor, static_cast<RealType>(0), static_cast<RealType>(0));
	            if (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) {
	                numerator2 += BaseModel::gradientNumerator2Contrib(x, predictor);
	            }
	        }

	        const int key = segments.strata[s];
	        BaseModel::incrementGradientAndHessian(it, w, // Signature-only, for iterator-type specialization
                                                &gradient, &hessian, numerator1, numerator2,
                                                denomPid[key], hNWeight[key], 0, 0, 0); // When function is in-lined, compiler will only use necessary arguments
	    }

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
void ModelSpecifics<BaseModel,RealType>::computeSchoenfeldResiduals(int indexOne,
                                                                    std::vector<double>* residuals,
                                                                    std::vector<double>* otimes,
                                                                    std::vector<int>* strata,
                                                                    double* covariate,
                                                                    double* score,
                                                                    bool useWeights) {
    if (useWeights) {
        throw new std::logic_error("Weights are not yet implemented in Schoenfeld residual calculations");
    } else { // no weights
        switch (hX.getFormatType(indexOne)) {
        case INDICATOR :
            getSchoenfeldResidualsImpl<IndicatorIterator<RealType>>(indexOne, residuals, otimes, strata, covariate, score, weighted);
            break;
        case SPARSE :
            getSchoenfeldResidualsImpl<SparseIterator<RealType>>(indexOne, residuals, otimes, strata, covariate, score, weighted);
            break;
        case DENSE :
            getSchoenfeldResidualsImpl<DenseIterator<RealType>>(indexOne, residuals, otimes, strata, covariate, score, weighted);
            break;
        case INTERCEPT :
            getSchoenfeldResidualsImpl<InterceptIterator<RealType>>(indexOne, residuals, otimes, strata, covariate, score, weighted);
            break;
        }
    }
}


template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::makeDirty() {
	AbstractModelSpecifics::makeDirty();
	hessianSparseCrossTerms.clear(); // cached at the coefficients of an earlier call
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeFisherInformation(int indexOne, int indexTwo,
		double *oinfo, bool useWeights) {

	ensureDenominators();

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

	const bool scaleByEvents = BaseModel::fisherInformationScalesWithEvents;

	RealType information = static_cast<RealType>(0);
	for (; it.valid(); ++it) {
		const int k = it.index();
		const int group = BaseModel::getGroup(hPid, k);
		// Compile-time delegation

		BaseModel::incrementFisherInformation(it,
				w, // Signature-only, for iterator-type specialization
				&information,
				offsExpXBeta[k],
				0.0, 0.0, // numerPid[k], numerPid2[k], // remove
				denomPid[group],
				scaleByEvents ? hKWeight[k] * hNWeight[group] : hKWeight[k],
				it.value(), hXBeta[k], hY[k]); // When function is in-lined, compiler will only use necessary arguments
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
			sparseCross += (scaleByEvents ? hNWeight[n] : static_cast<RealType>(1)) *
				itSparseCross.value() / (denomPid[n] * denomPid[n]);
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

	// numerPid and numerPid2 are only read by the cumulative (Cox-type) scans; the grouped models
	// accumulate their numerators by stratum inside computeGradientAndHessianImpl()
	if (BaseModel::cumulativeGradientAndHessian) {

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
    } else if (BaseModel::hasNtoKIndices && IteratorType::isSparse) { // one stratum per segment
        const int* rows = hX.getCompressedColumnVector(index);
        const RealType* values = IteratorType::isIndicator ? nullptr : hX.getDataVector(index);
        const StratumSegments& segments = stratumSegments[index];

        // offsExpXBeta = offs * exp(xBeta) in grouped models, so an update multiplies it by exp(delta * x):
        // one exp() for an indicator column, one per distinct value of a coded sparse column
        RealType indicatorFactor = static_cast<RealType>(1);
        const uint16_t* codes = nullptr;
        if (legacy) { // exp() per entry; no value codes are set up
        } else if (IteratorType::isIndicator) {
            indicatorFactor = std::exp(realDelta);
        } else if (!valueCodes[index].codes.empty()) {
            const RealVector& distinct = valueCodes[index].values;
            expFactors.resize(distinct.size());
            for (size_t d = 0; d < distinct.size(); ++d) {
                expFactors[d] = std::exp(realDelta * distinct[d]);
            }
            codes = valueCodes[index].codes.data();
        }
        const bool multiply = !legacy && (IteratorType::isIndicator || codes != nullptr);

        for (size_t s = 0; s < segments.strata.size(); ++s) {
            RealType denominator = denomPid[segments.strata[s]]; // the same additions, in a register

            for (int i = segments.offsets[s]; i < segments.offsets[s + 1]; ++i) {
                const int k = rows[i];
                hXBeta[k] += realDelta * (IteratorType::isIndicator ? static_cast<RealType>(1) : values[i]);

                if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
                    RealType oldEntry = offsExpXBeta[k];
                    RealType newEntry;
                    if (multiply && oldEntry >= std::numeric_limits<RealType>::min() &&
                        oldEntry <= std::numeric_limits<RealType>::max()) { // not 0, subnormal, inf or NaN
                        newEntry = oldEntry * (IteratorType::isIndicator ? indicatorFactor : expFactors[codes[i]]);
                    } else {
                        newEntry = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k);
                    }
                    offsExpXBeta[k] = newEntry;
                    denominator += (newEntry - oldEntry);
                }
            }
            denomPid[segments.strata[s]] = denominator;
        }
    } else if (BaseModel::hasLogisticRows && !legacy) { // legacy: the 3.7 code below
        // Each row is its own stratum with denominator 1 + exp(xBeta): set from exp(xBeta) (an
        // accumulated new - old drifts after large moves). exp(xBeta + delta x) = exp(xBeta) exp(delta x),
        // so an indicator or intercept column takes one exp() per column, and a column with few
        // distinct values one per distinct value, instead of one per entry.
        const RealType minEntry = std::numeric_limits<RealType>::min(); // multiply normal, finite values only
        const RealType maxEntry = std::numeric_limits<RealType>::max();
        RealType factor = static_cast<RealType>(1);
        const uint16_t* codes = nullptr;
        bool multiply = false;
        if (IteratorType::isIndicator) {
            factor = std::exp(realDelta);
            multiply = factor >= minEntry && factor <= maxEntry;
        } else if (valueCodes.size() == static_cast<size_t>(J) && !valueCodes[index].codes.empty()) {
            const RealVector& distinct = valueCodes[index].values;
            expFactors.resize(distinct.size());
            multiply = true;
            for (size_t d = 0; d < distinct.size(); ++d) {
                expFactors[d] = std::exp(realDelta * distinct[d]);
                multiply = multiply && expFactors[d] >= minEntry && expFactors[d] <= maxEntry;
            }
            codes = valueCodes[index].codes.data();
        }
        if (Weights::isWeighted && activePartition) {
            // training entries only; held-out rows keep stale xBeta and exp(xBeta) (see
            // setHeldOutRowsUnused()). Each row gets the same operations as below.
            const TrainingEntries t = getTrainingEntries<IteratorType>(index);
            RealType* xBeta = hXBeta.data();
            RealType* expXBeta = offsExpXBeta.data();
            if (!IteratorType::isIndicator && !multiply) {
                const size_t n = static_cast<size_t>(t.range[1] - t.range[0]) + (t.range[3] - t.range[2]);
                if (expBuffer.size() < n) {
                    expBuffer.resize(n);
                }
                RealType* buffer = expBuffer.data();
                size_t m = 0;
                for (int r = 0; r < 4; r += 2) {
                    for (int q = t.range[r]; q < t.range[r + 1]; ++q) {
                        const int k = t.rows[q];
                        const RealType v = xBeta[k] + realDelta * t.values[IteratorType::isSparse ? q : k];
                        xBeta[k] = v;
                        buffer[m++] = v;
                    }
                }
                for (size_t q = 0; q < n; ++q) {
                    buffer[q] = vectorizableExp(buffer[q]);
                }
                m = 0;
                for (int r = 0; r < 4; r += 2) {
                    for (int q = t.range[r]; q < t.range[r + 1]; ++q) {
                        expXBeta[t.rows[q]] = buffer[m++];
                    }
                }
            } else {
                for (int r = 0; r < 4; r += 2) {
                    for (int q = t.range[r]; q < t.range[r + 1]; ++q) {
                        const int k = t.rows[q];
                        xBeta[k] += IteratorType::isIndicator ? realDelta :
                            realDelta * t.values[IteratorType::isSparse ? q : k];
                        const RealType oldEntry = expXBeta[k];
                        RealType newEntry;
                        if (multiply && oldEntry >= minEntry && oldEntry <= maxEntry) { // not 0, subnormal, inf or NaN
                            newEntry = oldEntry * (IteratorType::isIndicator ? factor :
                                expFactors[t.codes[IteratorType::isSparse ? q : k]]);
                        } else {
                            newEntry = BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k);
                        }
                        expXBeta[k] = newEntry;
                    }
                }
            }
        } else if (!IteratorType::isIndicator && !multiply) {
            // exp(xBeta) for every entry, by vectorizableExp(); a sparse column through a buffer
            const int n = it.size();
            const RealType* values = hX.getDataVector(index);
            RealType* xBeta = hXBeta.data();
            RealType* expXBeta = offsExpXBeta.data();
            if (IteratorType::isSparse) {
                const int* rows = hX.getCompressedColumnVector(index);
                if (expBuffer.size() < static_cast<size_t>(n)) {
                    expBuffer.resize(n);
                }
                RealType* buffer = expBuffer.data();
                for (int q = 0; q < n; ++q) {
                    const int k = rows[q];
                    const RealType t = xBeta[k] + realDelta * values[q];
                    xBeta[k] = t;
                    buffer[q] = t;
                }
                for (int q = 0; q < n; ++q) {
                    buffer[q] = vectorizableExp(buffer[q]);
                }
                for (int q = 0; q < n; ++q) {
                    expXBeta[rows[q]] = buffer[q];
                }
            } else {
                for (int q = 0; q < n; ++q) {
                    const RealType t = xBeta[q] + realDelta * values[q];
                    xBeta[q] = t;
                    expXBeta[q] = vectorizableExp(t);
                }
            }
        } else {
            for (int q = 0; it; ++it, ++q) {
                const int k = it.index();
                hXBeta[k] += realDelta * it.value();
                const RealType oldEntry = offsExpXBeta[k];
                RealType newEntry;
                if (multiply && oldEntry >= minEntry && oldEntry <= maxEntry) { // not 0, subnormal, inf or NaN
                    newEntry = oldEntry * (IteratorType::isIndicator ? factor : expFactors[codes[q]]);
                } else {
                    newEntry = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k);
                }
                offsExpXBeta[k] = newEntry;
            }
        }
        denominatorsStale = true; // denomPid[k] = 1 + newEntry, deferred to ensureDenominators()
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

    if (BaseModel::likelihoodHasDenominator && BaseModel::hasNtoKIndices && !BaseModel::cumulativeGradientAndHessian) {
        // denomPid += new - old cancels when a denominator shrinks by many orders of magnitude, down to
        // zero or negative values. Re-sum all strata once entries may have moved by a factor of
        // exp(6.9) = 1000 since the last exact summation.
        if (columnMaxAbsX.size() != J) {
            columnMaxAbsX.assign(J, -1.0);
        }
        if (columnMaxAbsX[index] < 0.0) {
            double m = 0.0;
            for (IteratorType itMax(hX, index); itMax; ++itMax) {
                m = std::max(m, std::abs(static_cast<double>(itMax.value())));
            }
            columnMaxAbsX[index] = m;
        }
        denomMove += std::abs(static_cast<double>(realDelta)) * columnMaxAbsX[index];
        if (denomMove > 6.9) {
            resumStratumDenominators();
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
            denomMove = 0.0;
        }
        denominatorsStale = false;
        computeAccumlatedDenominator(Weights::isWeighted);
    }
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::resumStratumDenominators() {
    fillVector(denomPid.data(), N, BaseModel::getDenomNullValue());
    for (size_t k = 0; k < K; ++k) {
        incrementByGroup(denomPid.data(), hPid, k, offsExpXBeta[k]);
    }
    denomMove = 0.0;
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

    int nTies = 0;

    for (size_t k = index + 1; k < K; ++k) {
        if (weights == nullptr || weights[k] != 0.0) {
            int nextPid = hPid[k];

            if (nextPid != lastPid) { // start new strata
                pid++;
                accReset.push_back(pid);
                nTies = 0;
                lastPid = nextPid;
            } else {

                if (lastEvent == 1.0 && lastTime == hOffs[k] && lastEvent == hY[k]) {
                    // In a tie, do not increment denominator
                    ++nTies;
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
void ModelSpecifics<BaseModel,RealType>::setupStratumSegments() {
    stratumSegments.assign(J, StratumSegments());

    for (size_t j = 0; j < J; ++j) {
        if (hX.getFormatType(j) == INDICATOR || hX.getFormatType(j) == SPARSE) {
            StratumSegments& segments = stratumSegments[j];
            const int n = static_cast<int>(hX.getNumberOfEntries(j));
            const int* rows = hX.getCompressedColumnVector(j);
            for (int i = 0; i < n; ++i) {
                const int stratum = hPid[rows[i]];
                if (i == 0 || stratum != segments.strata.back()) {
                    segments.offsets.push_back(i);
                    segments.strata.push_back(stratum);
                }
            }
            segments.offsets.push_back(n);
        }
    }
}

template <class BaseModel,typename RealType>
FoldPartitionPtr ModelSpecifics<BaseModel,RealType>::makeFoldPartition(const std::vector<int>& segmentOfRow,
                                                                       int folds) const {
    if (!BaseModel::hasLogisticRows || legacy || segmentOfRow.size() != K || folds < 1) {
        return nullptr;
    }
    // Single precision keeps the 0/1-mask fold fits (bitwise equal to the weighted ones): reordered float
    // sums move its erratic stopping cycles (cvPLP ps_s: 3805 -> 4062 fold cycles, selected variance +0.11%)
    if (sizeof(RealType) < sizeof(double)) {
        return nullptr;
    }
    const int nSegments = folds + 2; // held out in fold 0, ..., fold folds - 1; always in training; never
    for (int s : segmentOfRow) {
        if (s < 0 || s >= nSegments) {
            return nullptr;
        }
    }

    auto partition = std::make_shared<LogisticFoldPartition>();
    LogisticFoldPartition& p = *partition;
    p.data = &hX;
    p.folds = folds;
    p.segmentOfRow = segmentOfRow;
    const int* segment = p.segmentOfRow.data();

    // Rows by segment, for DENSE and INTERCEPT columns
    p.rowSegments.assign(nSegments + 1, 0);
    for (size_t k = 0; k < K; ++k) {
        ++p.rowSegments[segment[k] + 1];
    }
    for (int s = 0; s < nSegments; ++s) {
        p.rowSegments[s + 1] += p.rowSegments[s];
    }
    std::vector<int> next(p.rowSegments.begin(), p.rowSegments.end() - 1);
    p.rowOrder.resize(K);
    for (size_t k = 0; k < K; ++k) {
        p.rowOrder[next[segment[k]]++] = static_cast<int>(k);
    }

    // INDICATOR and SPARSE columns, stable-partitioned by segment
    const bool haveCodes = valueCodes.size() == static_cast<size_t>(J);
    p.rowsStart.assign(J, 0);
    p.valuesStart.assign(J, 0);
    p.codesStart.assign(J, 0);
    size_t nRows = 0, nValues = 0, nCodes = 0;
    for (size_t j = 0; j < J; ++j) {
        const FormatType format = hX.getFormatType(j);
        if (format == INDICATOR || format == SPARSE) {
            const size_t n = hX.getNumberOfEntries(j);
            p.rowsStart[j] = nRows;
            nRows += n;
            if (format == SPARSE) {
                p.valuesStart[j] = nValues;
                nValues += n;
                if (haveCodes && !valueCodes[j].codes.empty()) {
                    p.codesStart[j] = nCodes;
                    nCodes += n;
                }
            }
        }
    }
    p.rows.resize(nRows);
    p.values.resize(nValues);
    p.codes.resize(nCodes);
    p.segments.assign(J * (nSegments + 1), 0);

    for (size_t j = 0; j < J; ++j) {
        const FormatType format = hX.getFormatType(j);
        if (format != INDICATOR && format != SPARSE) {
            continue;
        }
        const int n = static_cast<int>(hX.getNumberOfEntries(j));
        const int* columnRows = hX.getCompressedColumnVector(j);
        int* offsets = &p.segments[j * (nSegments + 1)];
        for (int q = 0; q < n; ++q) {
            ++offsets[segment[columnRows[q]] + 1];
        }
        for (int s = 0; s < nSegments; ++s) {
            offsets[s + 1] += offsets[s];
        }
        std::copy(offsets, offsets + nSegments, next.begin());
        int* rows = p.rows.data() + p.rowsStart[j];
        for (int q = 0; q < n; ++q) {
            rows[next[segment[columnRows[q]]]++] = columnRows[q];
        }
        if (format == SPARSE) {
            std::copy(offsets, offsets + nSegments, next.begin());
            const RealType* x = hX.getDataVector(j);
            RealType* values = p.values.data() + p.valuesStart[j];
            const bool coded = haveCodes && !valueCodes[j].codes.empty();
            const uint16_t* columnCodes = coded ? valueCodes[j].codes.data() : nullptr;
            uint16_t* codes = coded ? p.codes.data() + p.codesStart[j] : nullptr;
            for (int q = 0; q < n; ++q) {
                const int position = next[segment[columnRows[q]]]++;
                values[position] = x[q];
                if (coded) {
                    codes[position] = columnCodes[q];
                }
            }
        }
    }

    p.bytes = sizeof(int) * (p.segmentOfRow.size() + p.rowOrder.size() + p.rowSegments.size() +
        p.segments.size() + p.rows.size()) + sizeof(size_t) * 3 * J + sizeof(RealType) * p.values.size() +
        sizeof(uint16_t) * p.codes.size();
    return partition;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::setFoldPartition(FoldPartitionPtr partition, int fold) {
    activePartition.reset();
    partitionFold = -1;
    if (!BaseModel::hasLogisticRows || !partition || !trainingRowsOnly) {
        return;
    }
    auto p = std::dynamic_pointer_cast<const LogisticFoldPartition>(partition);
    if (!p || p->data != &hX || p->segmentOfRow.size() != K || fold < 0 || fold >= p->folds) {
        return;
    }
    // The training segments must be exactly the weight-1 rows
    for (size_t k = 0; k < K; ++k) {
        const int s = p->segmentOfRow[k];
        const bool training = s != fold && s != p->folds + 1;
        if (training != (trainingRow[k] != 0)) {
            return;
        }
    }
    activePartition = p;
    partitionFold = fold;
}

template <class BaseModel,typename RealType> template <class IteratorType>
typename ModelSpecifics<BaseModel,RealType>::TrainingEntries
ModelSpecifics<BaseModel,RealType>::getTrainingEntries(int index) const {
    const LogisticFoldPartition& p = *activePartition;
    const int folds = p.folds;
    const int fold = partitionFold;
    TrainingEntries t;
    const int* offsets;
    if (IteratorType::isSparse) {
        t.rows = p.rows.data() + p.rowsStart[index];
        t.values = IteratorType::isIndicator ? nullptr : p.values.data() + p.valuesStart[index];
        t.codes = (IteratorType::isIndicator || p.codes.empty()) ? nullptr : p.codes.data() + p.codesStart[index];
        offsets = &p.segments[index * (folds + 3)];
    } else {
        t.rows = p.rowOrder.data();
        t.values = IteratorType::isIndicator ? nullptr : hX.getDataVector(index);
        t.codes = (IteratorType::isIndicator || valueCodes.size() != static_cast<size_t>(J) ||
            valueCodes[index].codes.empty()) ? nullptr : valueCodes[index].codes.data();
        offsets = p.rowSegments.data();
    }
    // segments 0 .. fold - 1 and fold + 1 .. folds (always in training); not fold, not folds + 1 (never)
    t.range[0] = offsets[0];
    t.range[1] = offsets[fold];
    t.range[2] = offsets[fold + 1];
    t.range[3] = offsets[folds + 1];
    return t;
}

template <class BaseModel,typename RealType> template <class IteratorType>
RealType ModelSpecifics<BaseModel,RealType>::computeXjYOverTrainingEntries(int index) const {
    const TrainingEntries t = getTrainingEntries<IteratorType>(index);
    RealType sum = static_cast<RealType>(0);
    for (int r = 0; r < 4; r += 2) {
        for (int q = t.range[r]; q < t.range[r + 1]; ++q) {
            const int k = t.rows[q];
            sum += IteratorType::isIndicator ? hY[k] : t.values[IteratorType::isSparse ? q : k] * hY[k];
        }
    }
    return sum;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::setupValueCodes() {
    valueCodes.assign(J, ValueCodes());

    // Logistic regression also codes dense columns, up to 256 distinct values
    const size_t maxDistinct = BaseModel::hasLogisticRows ? 256 :
        static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;

    for (size_t j = 0; j < J; ++j) {
        if (hX.getFormatType(j) == SPARSE || (BaseModel::hasLogisticRows && hX.getFormatType(j) == DENSE)) {
            const size_t n = hX.getNumberOfNonZeroEntries(j);
            const RealType* x = hX.getDataVector(j);
            // Worthwhile when an exp() per distinct value is much cheaper than one per entry
            const size_t maxValues = std::min(n / 4, maxDistinct);

            std::unordered_map<RealType, uint16_t> index;
            ValueCodes& coded = valueCodes[j];
            coded.codes.resize(n);
            size_t i = 0;
            for (; i < n; ++i) {
                auto found = index.find(x[i]);
                if (found == index.end()) {
                    if (coded.values.size() == maxValues) {
                        break; // too many distinct values
                    }
                    found = index.emplace(x[i], static_cast<uint16_t>(coded.values.size())).first;
                    coded.values.push_back(x[i]);
                }
                coded.codes[i] = found->second;
            }
            if (i < n) {
                coded = ValueCodes();
            }
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
    } // else sparseIndices are not needed: only the cumulative scans and their numerators read them
    if (BaseModel::hasNtoKIndices) {
        setupStratumSegments();
        if (!legacy) {
            setupValueCodes();
        }
    } else if (BaseModel::hasLogisticRows && !legacy) {
        setupValueCodes();
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
