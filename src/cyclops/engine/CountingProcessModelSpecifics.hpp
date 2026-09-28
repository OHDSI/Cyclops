/*
 * CountingProcessModelSpecifics.hpp
 *
 * CPU engine for Cox proportional hazards (Breslow ties): "cox" (every row starts at the time
 * origin) and "cox_counting" for Surv(start, stop, event), where a row is at risk at event time t
 * when start < t <= stop.
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
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "ModelSpecifics.h"
#include "WorkerPool.h"

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

    void computeGradientAndHessian(int index, double* ogradient, double* ohessian, bool useWeights);

    void computeThirdDerivative(int index, double* othird, bool useWeights);

    void setThreads(int threads);

    bool computeGradientsAndHessians(const std::vector<int>& indices,
                                     std::vector<GradientHessian>& gh, bool useWeights);

    void updateXBeta(double delta, int index, bool useWeights);

    void computeRemainingStatistics(bool useWeights);

    void computeFisherInformation(int indexOne, int indexTwo, double* oinfo, bool useWeights);

    void computeSchoenfeldResiduals(int indexOne, std::vector<double>* residuals,
                                    std::vector<double>* times, std::vector<int>* strata,
                                    double* covariate, double* score, bool useWeights);

    void computeMMGradientAndHessian(std::vector<GradientHessian>& gh,
                                     const std::vector<bool>& fixBeta, bool useWeights) {
        throw std::logic_error("MM updates are not yet implemented for Cox models");
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
    using Base::hNWeight;
    using Base::N;
    using Base::K;
    using Base::J;
    using Base::hPid;
    using Base::hPidSize;
    using Base::hPidInternal;
    using Base::hPidOriginal;
    using Base::accReset;
    using Base::sparseIndices;
    using Base::hXjY;

    template <class IteratorType, bool isWeighted>
    void subtractExitNumerators(int index);

    template <class IteratorType, bool isWeighted>
    void updateXBetaAndDenominators(RealType delta, int index);

    void setupRiskSetIndices();

    void accumulateDenominators(int begin, int end);

    void accumulateAllDenominators();

    void accumulateHazards(int begin, int end);

    void computeAllHazards();

    struct Workspace { // per-thread numerator changes of one column at each group
        RealVector numer;
        RealVector numer2;
    };

    template <class IteratorType, bool isWeighted>
    void hazardGradientAndHessian(int index, Workspace& workspace, double* ogradient,
                                  double* ohessian) const;

    void dispatchHazardGradientAndHessian(int index, Workspace& workspace, double* ogradient,
                                          double* ohessian, bool useWeights) const;

    void computeScanNumerators(int index, bool useWeights);

    std::vector<RealType> denseColumn(int index) const;

    std::vector<RealType> riskSetSums(const std::vector<RealType>& x) const;

    const RealVector& hStart; // empty: every row is at risk from the time origin
    std::vector<int> hExitPid; // exit group per row; N when the row never leaves
    bool hasExits;

    std::vector<int> groupStratumEnd; // one past the last group of each group's stratum
    std::vector<int> groupStratumStart;

    // Breslow increments d / D and d / D^2 of each group. In double precision these hold their
    // running sums within the stratum, so a segment's total is one difference; in single precision
    // such differences cancel badly, so segments sum the increments directly, as the Cox scan does.
    static const bool hazardSums = !std::is_same<RealType, float>::value;
    RealVector cumHazard;
    RealVector cumHazard2;

    Workspace workspace;

    std::vector<bool> included; // rows with non-zero weight in the current layout
    bool hasLayout = false;

    std::unique_ptr<WorkerPool> pool; // only with more than one thread
    std::vector<Workspace> workspaces; // one per pool thread
};

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::setPidForAccumulation(const double* weights) {

    // The layout depends only on which rows have non-zero weight, and callers often repeat it
    bool same = hasLayout && included.size() == K;
    for (size_t k = 0; same && k < K; ++k) {
        same = included[k] == (weights == nullptr || weights[k] != 0.0);
    }
    if (same) {
        return;
    }
    included.resize(K);
    for (size_t k = 0; k < K; ++k) {
        included[k] = (weights == nullptr || weights[k] != 0.0);
    }
    hasLayout = true;

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

    groupStratumStart.resize(N);
    groupStratumEnd.resize(N);
    int stratumStart = 0;
    for (int stratumEnd : accReset) {
        for (int g = stratumStart; g < stratumEnd; ++g) {
            groupStratumStart[g] = stratumStart;
            groupStratumEnd[g] = stratumEnd;
        }
        stratumStart = stratumEnd;
    }

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

    // Groups where a column's accumulated numerators change: entries and exits of its rows.
    // Columns are independent, so they are built in parallel when there are threads.
    const int unused = static_cast<int>(N);
    sparseIndices.assign(J, nullptr);
    auto build = [this, unused](size_t j, int) {
        if (hX.getFormatType(j) != DENSE && hX.getFormatType(j) != INTERCEPT) {
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
            sparseIndices[j] = bsccs::make_shared<AbstractModelSpecifics::IndexVector>(std::move(groups));
        }
    };
    if (pool) {
        pool->run(J, build);
    } else {
        for (size_t j = 0; j < J; ++j) {
            build(j, 0);
        }
    }
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::setThreads(int threads) {

    if (threads <= 1) {
        pool.reset();
    } else if (!pool || pool->size() != threads) {
        pool.reset(new WorkerPool(threads));
    }
    workspaces.resize(pool ? threads : 0);
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::computeGradientsAndHessians(const std::vector<int>& indices,
        std::vector<GradientHessian>& gh, bool useWeights) {

    if (!pool || cumHazard.size() != N) {
        return false;
    }
    // Each column only reads the model state and writes its own workspace, and the result equals
    // computeGradientAndHessian()
    gh.resize(indices.size());
    pool->run(indices.size(), [&](size_t task, int thread) {
        const int index = indices[task];
        if (hX.getNumberOfNonZeroEntries(index) == 0) {
            gh[task] = GradientHessian(0.0, 0.0);
        } else {
            dispatchHazardGradientAndHessian(index, workspaces[thread], &gh[task].first,
                                             &gh[task].second, useWeights);
        }
    });
    return true;
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeNumeratorForGradient(int index, bool useWeights) {
    // Numerators are accumulated where they are used: in computeGradientAndHessian() and, for the
    // scans, in computeScanNumerators()
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeScanNumerators(int index, bool useWeights) {

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
void CountingProcessModelSpecifics<RealType>::computeGradientAndHessian(int index, double* ogradient,
                                                                        double* ohessian, bool useWeights) {

    if (cumHazard.size() != N || hX.getNumberOfNonZeroEntries(index) == 0) {
        computeScanNumerators(index, useWeights);
        Base::computeGradientAndHessian(index, ogradient, ohessian, useWeights);
        return;
    }
    dispatchHazardGradientAndHessian(index, workspace, ogradient, ohessian, useWeights);
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::dispatchHazardGradientAndHessian(int index,
        Workspace& workspace, double* ogradient, double* ohessian, bool useWeights) const {

    switch (hX.getFormatType(index)) {
        case INDICATOR :
            if (useWeights) {
                hazardGradientAndHessian<IndicatorIterator<RealType>, true>(index, workspace, ogradient, ohessian);
            } else {
                hazardGradientAndHessian<IndicatorIterator<RealType>, false>(index, workspace, ogradient, ohessian);
            }
            break;
        case SPARSE :
            if (useWeights) {
                hazardGradientAndHessian<SparseIterator<RealType>, true>(index, workspace, ogradient, ohessian);
            } else {
                hazardGradientAndHessian<SparseIterator<RealType>, false>(index, workspace, ogradient, ohessian);
            }
            break;
        case DENSE :
            if (useWeights) {
                hazardGradientAndHessian<DenseIterator<RealType>, true>(index, workspace, ogradient, ohessian);
            } else {
                hazardGradientAndHessian<DenseIterator<RealType>, false>(index, workspace, ogradient, ohessian);
            }
            break;
        case INTERCEPT :
            if (useWeights) {
                hazardGradientAndHessian<InterceptIterator<RealType>, true>(index, workspace, ogradient, ohessian);
            } else {
                hazardGradientAndHessian<InterceptIterator<RealType>, false>(index, workspace, ogradient, ohessian);
            }
            break;
        default : break;
    }
}

template <typename RealType> template <class IteratorType, bool isWeighted>
void CountingProcessModelSpecifics<RealType>::hazardGradientAndHessian(int index, Workspace& workspace,
        double* ogradient, double* ohessian) const {

    // A column's risk-set numerators change only at its rows' entry and exit groups (all groups for
    // dense columns). Collect those changes, then each segment [a, b) between changes adds
    // numer * (C[b-1] - C[a-1]) to the gradient and numer2 * (C[b-1] - C[a-1]) -
    // numer^2 * (P[b-1] - P[a-1]) to the Hessian. Only the workspace is written.
    const int unused = static_cast<int>(N);
    const std::vector<int>* points = sparseIndices[index].get(); // null: every group
    RealVector& change = workspace.numer;
    RealVector& change2 = workspace.numer2;
    if (change.size() < N + 1) {
        change.resize(N + 1);
        change2.resize(N + 1);
    }
    const RealType zero = static_cast<RealType>(0);
    if (points != nullptr) {
        for (int g : *points) {
            change[g] = zero;
            change2[g] = zero;
        }
    } else {
        std::fill(change.begin(), change.begin() + N, zero);
        std::fill(change2.begin(), change2.begin() + N, zero);
    }

    for (IteratorType it(hX, index); it; ++it) {
        const int k = it.index();
        const int entry = hPid[k];
        if (entry < unused) {
            const RealType x = it.value();
            const RealType value = (isWeighted ? hKWeight[k] : static_cast<RealType>(1)) * offsExpXBeta[k] * x;
            const int exit = hExitPid[k];
            change[entry] += value;
            if (!IteratorType::isIndicator) {
                change2[entry] += value * x;
            }
            if (exit < unused) {
                change[exit] -= value;
                if (!IteratorType::isIndicator) {
                    change2[exit] -= value * x;
                }
            }
        }
    }

    RealType numer = zero;
    RealType numer2 = zero;
    RealType gradient = zero;
    RealType hessian = zero;
    int stratumEnd = 0;
    const size_t count = (points != nullptr) ? points->size() : N;
    for (size_t p = 0; p < count; ++p) {
        const int a = (points != nullptr) ? (*points)[p] : static_cast<int>(p);
        if (a >= stratumEnd) { // new stratum
            numer = zero;
            numer2 = zero;
            stratumEnd = groupStratumEnd[a];
        }
        numer += change[a];
        numer2 += IteratorType::isIndicator ? change[a] : change2[a];

        const int next = (p + 1 < count) ? ((points != nullptr) ? (*points)[p + 1] : a + 1) : stratumEnd;
        const int b = std::min(next, stratumEnd);
        if (hazardSums) {
            const bool first = (a == groupStratumStart[a]);
            const RealType hazard = cumHazard[b - 1] - (first ? zero : cumHazard[a - 1]);
            const RealType hazard2 = cumHazard2[b - 1] - (first ? zero : cumHazard2[a - 1]);
            gradient += numer * hazard;
            hessian += numer2 * hazard - numer * numer * hazard2;
        } else {
            for (int g = a; g < b; ++g) {
                gradient += numer * cumHazard[g];
                hessian += numer2 * cumHazard[g] - numer * numer * cumHazard2[g];
            }
        }
    }

    *ogradient = static_cast<double>(gradient - hXjY[index]);
    *ohessian = static_cast<double>(hessian);
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeThirdDerivative(int index, double* othird,
                                                                     bool useWeights) {

    computeScanNumerators(index, useWeights); // the Cox third derivative scans numerPid
    Base::computeThirdDerivative(index, othird, useWeights);
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

    // For sparse columns, only groups [lo, hi) of each stratum holding the column's rows change:
    // from their first entry to their last exit (or the end of the stratum)
    const int unused = static_cast<int>(N);
    const bool ranged = IteratorType::isSparse && accDenomPid.size() == N + 1 && cumHazard.size() == N;
    int lo = unused;
    int hi = 0;
    int stratumEnd = 0;

    // Rows of a 0/1 column all change by the same factor, so one exp() serves the whole update;
    // computeRemainingStatistics() recomputes exp(x'beta) from scratch
    const RealType factor = IteratorType::isIndicator ? std::exp(delta) : static_cast<RealType>(1);

    for (IteratorType it(hX, index); it; ++it) {
        const int k = it.index();
        hXBeta[k] += delta * it.value();

        const RealType oldEntry = isWeighted ? hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k];
        offsExpXBeta[k] = IteratorType::isIndicator ? offsExpXBeta[k] * factor : std::exp(hXBeta[k]);
        const RealType newEntry = isWeighted ? hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k];

        const int entry = hPid[k];
        const int exit = hExitPid[k];
        denomPid[entry] += (newEntry - oldEntry);
        if (exit < unused) {
            denomPid[exit] -= (newEntry - oldEntry);
        }

        if (ranged && entry < unused) {
            if (entry >= stratumEnd) { // rows are grouped by stratum
                if (lo < hi) {
                    accumulateDenominators(lo, hi);
                    accumulateHazards(lo, hi);
                }
                lo = entry;
                hi = entry;
                stratumEnd = groupStratumEnd[entry];
            }
            lo = std::min(lo, entry);
            hi = std::max(hi, exit < unused ? exit : stratumEnd);
        }
    }

    if (!ranged) {
        accumulateAllDenominators();
        computeAllHazards();
    } else if (lo < hi) {
        accumulateDenominators(lo, hi);
        accumulateHazards(lo, hi);
    }
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::accumulateDenominators(int begin, int end) {

    // Same running sum as computeAccumlatedDenominator, restarted within one stratum
    RealType total = (begin == groupStratumStart[begin]) ? static_cast<RealType>(0) : accDenomPid[begin - 1];
    for (int g = begin; g < end; ++g) {
        total += denomPid[g];
        accDenomPid[g] = total;
    }
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::accumulateAllDenominators() {

    if (accDenomPid.size() != N + 1) {
        accDenomPid.resize(N + 1, static_cast<RealType>(0));
    }
    for (size_t g = 0; g < N; g = groupStratumEnd[g]) {
        accumulateDenominators(g, groupStratumEnd[g]);
    }
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeAllHazards() {

    if (hNWeight.size() < N) { // event counts not set yet; gradients use the scan until they are
        cumHazard.clear();
        cumHazard2.clear();
        return;
    }
    cumHazard.resize(N);
    cumHazard2.resize(N);
    for (size_t g = 0; g < N; g = groupStratumEnd[g]) {
        accumulateHazards(g, groupStratumEnd[g]);
    }
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::accumulateHazards(int begin, int end) {

    // Increments change only where denominators did, [begin, end); running sums change up to the
    // end of the stratum
    const int last = hazardSums ? groupStratumEnd[begin] : end;
    const bool first = (begin == groupStratumStart[begin]);
    RealType hazard = (first || !hazardSums) ? static_cast<RealType>(0) : cumHazard[begin - 1];
    RealType hazard2 = (first || !hazardSums) ? static_cast<RealType>(0) : cumHazard2[begin - 1];
    for (int g = begin; g < last; ++g) {
        const RealType inverse = static_cast<RealType>(1) / accDenomPid[g];
        const RealType increment = hNWeight[g] * inverse;
        if (hazardSums) {
            hazard += increment;
            hazard2 += increment * inverse;
        } else {
            hazard = increment;
            hazard2 = increment * inverse;
        }
        cumHazard[g] = hazard;
        cumHazard2[g] = hazard2;
    }
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeRemainingStatistics(bool useWeights) {

    // Net change of each group's risk set: entries minus exits
    const int unused = static_cast<int>(N);
    std::fill(denomPid.begin(), denomPid.begin() + N + 1, static_cast<RealType>(0));
    for (size_t k = 0; k < K; ++k) {
        offsExpXBeta[k] = std::exp(hXBeta[k]);
        const RealType value = useWeights ? hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k];
        denomPid[hPid[k]] += value;
        if (hExitPid[k] < unused) {
            denomPid[hExitPid[k]] -= value;
        }
    }
    accumulateAllDenominators();

    // Censored rows outside every risk set point to the unused group N; the Cox predictive
    // log-likelihood still takes log() of its denominator, so keep it finite
    accDenomPid[N] = static_cast<RealType>(1);

    computeAllHazards();
}

template <typename RealType>
std::vector<RealType> CountingProcessModelSpecifics<RealType>::denseColumn(int index) const {

    std::vector<RealType> x(K, static_cast<RealType>(0));
    for (GenericIterator<RealType> it(hX, index); it; ++it) {
        x[it.index()] = it.value();
    }
    return x;
}

template <typename RealType>
std::vector<RealType> CountingProcessModelSpecifics<RealType>::riskSetSums(
        const std::vector<RealType>& x) const {

    // Sum of w * exp(x'beta) * x over the risk set of each event-time group; as in the Cox
    // residuals, case weights come from hKWeight (all 1 without weights)
    std::vector<RealType> sums(N + 1, static_cast<RealType>(0)); // group N is unused
    for (size_t k = 0; k < K; ++k) {
        const RealType value = hKWeight[k] * offsExpXBeta[k] * x[k];
        sums[hPid[k]] += value;
        sums[hExitPid[k]] -= value;
    }

    RealType total = static_cast<RealType>(0);
    auto reset = accReset.begin();
    for (size_t g = 0; g < N; ++g) {
        if (static_cast<size_t>(*reset) == g) { // new stratum
            total = static_cast<RealType>(0);
            ++reset;
        }
        total += sums[g];
        sums[g] = total;
    }
    return sums;
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeFisherInformation(int indexOne, int indexTwo,
                                                                       double* oinfo, bool useWeights) {

    if (useWeights) {
        throw std::logic_error("Weights are not yet implemented in Fisher Information calculations");
    }

    const std::vector<RealType> xOne = denseColumn(indexOne);
    const std::vector<RealType> xTwo = denseColumn(indexTwo);
    std::vector<RealType> xBoth(K);
    for (size_t k = 0; k < K; ++k) {
        xBoth[k] = xOne[k] * xTwo[k];
    }

    const std::vector<RealType> numerOne = riskSetSums(xOne);
    const std::vector<RealType> numerTwo = riskSetSums(xTwo);
    const std::vector<RealType> numerBoth = riskSetSums(xBoth);

    RealType information = static_cast<RealType>(0);
    for (size_t g = 0; g < N; ++g) {
        const RealType denom = accDenomPid[g];
        information += hNWeight[g] * (numerBoth[g] / denom - numerOne[g] * numerTwo[g] / (denom * denom));
    }
    *oinfo = static_cast<double>(information);
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeSchoenfeldResiduals(int index,
        std::vector<double>* residuals, std::vector<double>* times, std::vector<int>* strata,
        double* covariate, double* score, bool useWeights) {

    if (useWeights) {
        throw std::logic_error("Weights are not yet implemented in Schoenfeld residual calculations");
    }

    const std::vector<RealType> x = denseColumn(index);
    std::vector<RealType> x2(K);
    for (size_t k = 0; k < K; ++k) {
        x2[k] = x[k] * x[k];
    }
    const std::vector<RealType> numer = riskSetSums(x);
    const std::vector<RealType> numer2 = riskSetSums(x2);

    if (residuals != nullptr) {
        residuals->clear();
    }
    if (times != nullptr) {
        times->clear();
    }
    if (strata != nullptr) {
        strata->clear();
    }

    const int unused = static_cast<int>(N);
    for (size_t k = 0; k < K; ++k) {
        if (hY[k] == static_cast<RealType>(1) && hPid[k] < unused) { // skip rows outside the fit
            const int g = hPid[k];
            if (residuals != nullptr) {
                residuals->push_back(static_cast<double>(x[k] - numer[g] / accDenomPid[g]));
            }
            if (times != nullptr) {
                times->push_back(static_cast<double>(hOffs[k]));
            }
            if (strata != nullptr) {
                strata->push_back(hPidOriginal[k]);
            }
        }
    }

    if (score != nullptr && covariate != nullptr) {
        // Score and information for (beta, time-interaction) at the current beta, with the
        // covariate (e.g. transformed time) evaluated at each event time
        RealType uObserved = static_cast<RealType>(0);
        RealType wObserved = static_cast<RealType>(0);
        std::vector<RealType> groupCovariate(N, static_cast<RealType>(0));
        for (size_t k = 0; k < K; ++k) {
            if (hY[k] == static_cast<RealType>(1) && hPid[k] < unused) {
                const RealType cov = static_cast<RealType>(covariate[k]);
                uObserved += hKWeight[k] * x[k];
                wObserved += hKWeight[k] * x[k] * cov;
                groupCovariate[hPid[k]] = cov;
            }
        }

        RealType uGradient = static_cast<RealType>(0);
        RealType wGradient = static_cast<RealType>(0);
        RealType uHessian = static_cast<RealType>(0);
        RealType xHessian = static_cast<RealType>(0);
        RealType wHessian = static_cast<RealType>(0);
        for (size_t g = 0; g < N; ++g) {
            const RealType t = numer[g] / accDenomPid[g];
            const RealType gradient = hNWeight[g] * t;
            const RealType hessian = hNWeight[g] * (numer2[g] / accDenomPid[g] - t * t);
            const RealType cov = groupCovariate[g];
            uGradient += gradient;
            wGradient += gradient * cov;
            uHessian += hessian;
            xHessian += hessian * cov;
            wHessian += hessian * cov * cov;
        }

        score[0] = static_cast<double>(uObserved - uGradient);
        score[1] = static_cast<double>(wObserved - wGradient);
        score[2] = static_cast<double>(uHessian);
        score[3] = static_cast<double>(xHessian);
        score[4] = static_cast<double>(xHessian);
        score[5] = static_cast<double>(wHessian);
    }
}

} // namespace bsccs

#endif /* COUNTINGPROCESSMODELSPECIFICS_HPP_ */
