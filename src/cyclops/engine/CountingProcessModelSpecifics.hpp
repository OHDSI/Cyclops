/*
 * CountingProcessModelSpecifics.hpp
 *
 * Cox proportional hazards (Breslow ties) for counting-process data, Surv(start, stop, event):
 * a row is at risk at event time t when start < t <= stop.
 *
 * Subclass of the Cox ModelSpecifics that reuses its gradient/Hessian scan, log-likelihood and
 * weighting code, and changes only how rows map onto risk-set groups.  Groups are the distinct
 * event times of each stratum in scan order (decreasing time).  Each row adds its contribution
 * at its entry group (first event time <= stop) and removes it at its exit group (first event
 * time <= start), so the Cox prefix scans over groups return (start, stop] risk-set sums and
 * each evaluation stays O(K + N) for K rows and N event times.
 */

#ifndef COUNTINGPROCESSMODELSPECIFICS_HPP_
#define COUNTINGPROCESSMODELSPECIFICS_HPP_

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>

#include "ModelSpecifics.h"

namespace bsccs {

template <typename RealType>
class CountingProcessModelSpecifics :
        public ModelSpecifics<BreslowTiedCoxProportionalHazards<RealType>, RealType> {
public:
    typedef ModelSpecifics<BreslowTiedCoxProportionalHazards<RealType>, RealType> Base;
    typedef typename Base::RealVector RealVector;

    CountingProcessModelSpecifics(const ModelData<RealType>& input)
        : Base(input), hStart(input.getStartTimeVectorRef()), hasExits(false) { }

    virtual ~CountingProcessModelSpecifics() { }

    AbstractModelSpecifics* clone(ComputeDeviceArguments computeDevice) const {
        return new CountingProcessModelSpecifics<RealType>(this->modelData);
    }

    void setPidForAccumulation(const double* weights);

    void computeNumeratorForGradient(int index, bool useWeights);

    void updateXBeta(double delta, int index, bool useWeights);

    void computeRemainingStatistics(bool useWeights);

    // The Cox versions below index rows rather than (start, stop] risk sets
    void computeFisherInformation(int indexOne, int indexTwo, double* oinfo, bool useWeights) {
        throw std::logic_error("Fisher information is not yet implemented for cox_counting models");
    }

    void computeSchoenfeldResiduals(int indexOne, std::vector<double>* residuals,
                                    std::vector<double>* times, std::vector<int>* strata,
                                    double* covariate, double* score, bool useWeights) {
        throw std::logic_error("Schoenfeld residuals are not yet implemented for cox_counting models");
    }

    void computeMMGradientAndHessian(std::vector<GradientHessian>& gh,
                                     const std::vector<bool>& fixBeta, bool useWeights) {
        throw std::logic_error("MM updates are not yet implemented for cox_counting models");
    }

private:
    using Base::hX;
    using Base::hY;
    using Base::hOffs;
    using Base::hXBeta;
    using Base::offsExpXBeta;
    using Base::denomPid;
    using Base::accDenomPid;
    using Base::numerPid;
    using Base::numerPid2;
    using Base::hKWeight;
    using Base::N;
    using Base::K;
    using Base::J;
    using Base::hPid;
    using Base::hPidSize;
    using Base::hPidInternal;
    using Base::hPidOriginal;
    using Base::accReset;
    using Base::sparseIndices;

    template <class IteratorType, bool isWeighted>
    void subtractExitNumerators(int index);

    template <class IteratorType, bool isWeighted>
    void updateXBetaAndDenominators(RealType delta, int index);

    void setupRiskSetIndices();

    const RealVector& hStart; // empty: every row is at risk from the time origin
    std::vector<int> hExitPid; // exit group per row; N when the row never leaves
    bool hasExits;
};

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::setPidForAccumulation(const double* weights) {

    // hPidOriginal holds each row's stratum; strata are contiguous
    hPidInternal = hPidOriginal;
    hPid = hPidInternal.data();
    hPidSize = hPidInternal.size();
    accReset.clear();

    const int none = -1;
    std::vector<int> entry(K, none);
    hExitPid.assign(K, none);

    auto included = [weights](size_t k) {
        return weights == nullptr || weights[k] != 0.0;
    };
    auto later = [](RealType groupTime, RealType time) { // groups are in decreasing time
        return groupTime > time;
    };

    std::vector<RealType> groupTime;
    size_t begin = 0;
    while (begin < K) {
        size_t end = begin + 1;
        while (end < K && hPidOriginal[end] == hPidOriginal[begin]) {
            ++end;
        }

        const size_t first = groupTime.size();
        for (size_t k = begin; k < end; ++k) {
            if (included(k) && hY[k] != static_cast<RealType>(0)) {
                groupTime.push_back(hOffs[k]);
            }
        }
        std::sort(groupTime.begin() + first, groupTime.end(), std::greater<RealType>());
        groupTime.erase(std::unique(groupTime.begin() + first, groupTime.end()), groupTime.end());

        if (groupTime.size() > first && first > 0) {
            accReset.push_back(first); // new stratum
        }

        const auto groupBegin = groupTime.begin() + first;
        const auto groupEnd = groupTime.end();
        for (size_t k = begin; k < end; ++k) {
            if (included(k)) {
                const auto in = std::lower_bound(groupBegin, groupEnd, hOffs[k], later);
                const auto out = hStart.empty() ? groupEnd :
                    std::lower_bound(in, groupEnd, hStart[k], later);
                if (in != out) { // at risk at one or more event times
                    entry[k] = in - groupTime.begin();
                    if (out != groupEnd) {
                        hExitPid[k] = out - groupTime.begin();
                    }
                }
            }
        }
        begin = end;
    }

    N = groupTime.size();
    accReset.push_back(N);

    hasExits = false;
    for (size_t k = 0; k < K; ++k) { // rows outside every risk set use the unused group N
        hPid[k] = (entry[k] == none) ? N : entry[k];
        if (hExitPid[k] == none) {
            hExitPid[k] = N;
        } else {
            hasExits = true;
        }
    }

    setupRiskSetIndices();
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::setupRiskSetIndices() {

    // Groups where a column's accumulated numerators change: entries and exits of its rows
    const int unused = static_cast<int>(N);
    sparseIndices.clear();
    for (size_t j = 0; j < J; ++j) {
        if (hX.getFormatType(j) == DENSE || hX.getFormatType(j) == INTERCEPT) {
            sparseIndices.push_back(nullptr);
        } else {
            std::vector<int> groups;
            const size_t n = hX.getNumberOfEntries(j);
            const int* rows = hX.getCompressedColumnVector(j);
            for (size_t i = 0; i < n; ++i) {
                const int k = rows[i];
                if (hPid[k] < unused) {
                    groups.push_back(hPid[k]);
                    if (hExitPid[k] < unused) {
                        groups.push_back(hExitPid[k]);
                    }
                }
            }
            std::sort(groups.begin(), groups.end());
            groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
            sparseIndices.push_back(
                bsccs::make_shared<AbstractModelSpecifics::IndexVector>(std::move(groups)));
        }
    }
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeNumeratorForGradient(int index, bool useWeights) {

    Base::computeNumeratorForGradient(index, useWeights); // adds each row at its entry group

    if (!hasExits) {
        return;
    }

    switch (hX.getFormatType(index)) {
        case INDICATOR :
            if (useWeights) {
                subtractExitNumerators<IndicatorIterator<RealType>, true>(index);
            } else {
                subtractExitNumerators<IndicatorIterator<RealType>, false>(index);
            }
            break;
        case SPARSE :
            if (useWeights) {
                subtractExitNumerators<SparseIterator<RealType>, true>(index);
            } else {
                subtractExitNumerators<SparseIterator<RealType>, false>(index);
            }
            break;
        case DENSE :
            if (useWeights) {
                subtractExitNumerators<DenseIterator<RealType>, true>(index);
            } else {
                subtractExitNumerators<DenseIterator<RealType>, false>(index);
            }
            break;
        case INTERCEPT :
            if (useWeights) {
                subtractExitNumerators<InterceptIterator<RealType>, true>(index);
            } else {
                subtractExitNumerators<InterceptIterator<RealType>, false>(index);
            }
            break;
        default : break;
    }
}

template <typename RealType> template <class IteratorType, bool isWeighted>
void CountingProcessModelSpecifics<RealType>::subtractExitNumerators(int index) {

    const int unused = static_cast<int>(N);
    for (IteratorType it(hX, index); it; ++it) {
        const int k = it.index();
        const int exit = hExitPid[k];
        if (exit < unused) { // same terms as the Cox numerators, so entries and exits cancel
            const RealType x = it.value();
            const RealType numerator = offsExpXBeta[k] * x;
            numerPid[exit] -= isWeighted ? hKWeight[k] * numerator : numerator;
            if (!IteratorType::isIndicator) {
                const RealType numerator2 = numerator * x;
                numerPid2[exit] -= isWeighted ? hKWeight[k] * numerator2 : numerator2;
            }
        }
    }
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::updateXBeta(double delta, int index, bool useWeights) {

    const RealType realDelta = static_cast<RealType>(delta);

    switch (hX.getFormatType(index)) {
        case INDICATOR :
            if (useWeights) {
                updateXBetaAndDenominators<IndicatorIterator<RealType>, true>(realDelta, index);
            } else {
                updateXBetaAndDenominators<IndicatorIterator<RealType>, false>(realDelta, index);
            }
            break;
        case SPARSE :
            if (useWeights) {
                updateXBetaAndDenominators<SparseIterator<RealType>, true>(realDelta, index);
            } else {
                updateXBetaAndDenominators<SparseIterator<RealType>, false>(realDelta, index);
            }
            break;
        case DENSE :
            if (useWeights) {
                updateXBetaAndDenominators<DenseIterator<RealType>, true>(realDelta, index);
            } else {
                updateXBetaAndDenominators<DenseIterator<RealType>, false>(realDelta, index);
            }
            break;
        case INTERCEPT :
            if (useWeights) {
                updateXBetaAndDenominators<InterceptIterator<RealType>, true>(realDelta, index);
            } else {
                updateXBetaAndDenominators<InterceptIterator<RealType>, false>(realDelta, index);
            }
            break;
        default : break;
    }
}

template <typename RealType> template <class IteratorType, bool isWeighted>
void CountingProcessModelSpecifics<RealType>::updateXBetaAndDenominators(RealType delta, int index) {

    const int unused = static_cast<int>(N);
    for (IteratorType it(hX, index); it; ++it) {
        const int k = it.index();
        hXBeta[k] += delta * it.value();

        const RealType oldEntry = isWeighted ? hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k];
        offsExpXBeta[k] = std::exp(hXBeta[k]);
        const RealType newEntry = isWeighted ? hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k];

        denomPid[hPid[k]] += (newEntry - oldEntry);
        if (hExitPid[k] < unused) {
            denomPid[hExitPid[k]] -= (newEntry - oldEntry);
        }
    }

    this->computeAccumlatedDenominator(isWeighted);
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeRemainingStatistics(bool useWeights) {

    Base::computeRemainingStatistics(useWeights); // exp(x'beta) and denominators at entry groups

    if (hasExits) {
        const int unused = static_cast<int>(N);
        for (size_t k = 0; k < K; ++k) {
            if (hExitPid[k] < unused) {
                denomPid[hExitPid[k]] -= useWeights ? hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k];
            }
        }
        this->computeAccumlatedDenominator(useWeights);
    }

    // Censored rows outside every risk set point to the unused group N; the Cox predictive
    // log-likelihood still takes log() of its denominator, so keep it finite
    accDenomPid[N] = static_cast<RealType>(1);
}

} // namespace bsccs

#endif /* COUNTINGPROCESSMODELSPECIFICS_HPP_ */
