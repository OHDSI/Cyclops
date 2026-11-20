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
 *
 * Fine-Gray rows (ModelData::loadFineGray): a row k with Fine-Gray group g ended with a competing event
 * at its stop time T_k. It is also at risk at every later event time t_i of its stratum (its "tail": the
 * groups [tailBegin_k, tailEnd_k), which start at the stratum's first group), with its term
 * w_k exp(x_k'beta) multiplied by a_k = 1 / G_g(T_k-) and by the group's m_g(i) = G_g(t_i-). Every risk-set
 * sum is then S(i) = (sum over ordinary rows) + sum_g m_g(i) (sum over tails of group g), kept as one
 * running sum per group, so an evaluation is O(K + N (1 + groups)) with no expanded rows. Columns use a
 * scan over all groups (fineGrayGradientAndHessian) instead of the change points of the hazard path.
 */

#ifndef COUNTINGPROCESSMODELSPECIFICS_HPP_
#define COUNTINGPROCESSMODELSPECIFICS_HPP_

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>

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

    bool supportsConcurrentGradients() const { return pool != nullptr; }

    void updateXBeta(double delta, int index, bool useWeights);

    void computeRemainingStatistics(bool useWeights);

    void computeFisherInformation(int indexOne, int indexTwo, double* oinfo, bool useWeights);

    void computeSchoenfeldResiduals(int indexOne, std::vector<double>* residuals,
                                    std::vector<double>* times, std::vector<int>* strata,
                                    double* covariate, double* score, bool useWeights);

    bool scoreContributions(int index, std::vector<double>& u);

    void computeMMGradientAndHessian(std::vector<GradientHessian>& gh,
                                     const std::vector<bool>& fixBeta, bool useWeights) {
        throw std::logic_error("MM updates are not yet implemented for Cox models");
    }

    // Second-order API (AbstractModelSpecifics), double precision only. Row k is at risk in groups
    // [entry, end) of its stratum with u_k = w_k exp(x_k'beta); dC_k is the Breslow hazard over
    // those groups, d_g the weighted event count and D_g the risk-set sum of group g.
    bool supportsSecondOrder() const { return secondOrder; }

    bool secondOrderPrepare(const std::vector<int>& F);

    bool fullGradient(const std::vector<int>& F, std::vector<double>& g);

    bool hessianBlock(const std::vector<int>& F, std::vector<double>& H);

    bool hessianVector(const std::vector<int>& F, const std::vector<double>& v, std::vector<double>& Hv);

    bool setDirection(const std::vector<int>& F, const std::vector<double>& d);

    double trialLogLikelihood(double t);

    bool commitStep(double t);

    double directionMaxAbs() const;

    double newtonCostRatio(const std::vector<int>& F);

    bool separationDirections(const std::vector<int>& columns, std::vector<int>& direction);

    bool hessianDiagonal(const std::vector<int>& F, std::vector<double>& h);

    bool woodburyParts(const std::vector<int>& F, int rank, std::vector<double>& S2, std::vector<double>& B);

    double newtonCgCostRatio(const std::vector<int>& F, bool preconditioned);

private:
    double cycleWork(const std::vector<int>& F) const; // sum_j nnz_j + groups touched_j
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

    void fineGrayGradientAndHessian(int index, double* ogradient, double* ohessian, bool useWeights);

    void computeTailHazards();

    bool participates(int k) const { return hPid[k] < static_cast<int>(N) || hasTail(k); }

    std::vector<RealType> denseColumn(int index) const;

    std::vector<RealType> riskSetSums(const std::vector<RealType>& x) const;

    const RealVector& hStart; // empty: every row is at risk from the time origin

    // Fine-Gray tails (see the header): fineGrayGroup[k] = g > 0 for rows with a non-empty tail in the layout
    bool hasTails = false;
    int fineGrayGroups = 0;
    std::vector<int> fineGrayGroup;
    std::vector<int> tailBegin;
    std::vector<int> tailEnd;
    std::vector<double> tailFactor; // a_k = 1 / G_g(T_k-)
    std::vector<double> tailMultiplier; // m_g(i) = G_g(t_i-), at (g - 1) * N + i
    std::vector<RealType> tailDenom; // net change of each group's tail sum, at (g - 1) * (N + 1) + i
    std::vector<double> tailHazard; // running sums within the stratum of m_g(i) d_i / D_i, at (g - 1) * N + i
    std::vector<double> tailWork; // scratch for fineGrayGradientAndHessian()
    bool hasTail(int k) const { return hasTails && fineGrayGroup[k] > 0; }
    int tailExit(int k) const { // where a tail leaves the running sums: N when it runs to its stratum's end
        return (tailEnd[k] == groupStratumEnd[tailBegin[k]]) ? static_cast<int>(N) : tailEnd[k];
    }
    double multiplier(int g, int i) const { return tailMultiplier[static_cast<size_t>(g - 1) * N + i]; }
    double tailHazardSum(int k) const { // sum over row k's tail of m_g(i) d_i / D_i
        return tailHazard[static_cast<size_t>(fineGrayGroup[k] - 1) * N + tailEnd[k] - 1];
    }
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

    // Second-order state. X_F is held row-wise (CSR over all K rows; positions within a row
    // ascend); rows entering and leaving each group are listed per layout.
    static const bool secondOrder = std::is_same<RealType, double>::value;
    bool secondOrderReady() const;
    void prepareGroupLists();
    int rowEnd(int k) const { // one past the row's last at-risk group
        return hExitPid[k] < static_cast<int>(N) ? hExitPid[k] : groupStratumEnd[hPid[k]];
    }
    double rowHazard(int k) const; // dC_k
    void productXv(const std::vector<double>& v, std::vector<double>& z) const; // z = X_F v

    std::vector<int> soF;
    bool soPrepared = false;
    std::vector<size_t> soRowStart; // K + 1
    std::vector<int> soPosition; // position in F of each entry
    std::vector<double> soValue;

    int layoutVersion = 0; // incremented whenever setPidForAccumulation() rebuilds the groups
    int soGroupVersion = -1;
    std::vector<int> soEnterStart; // N + 1: rows entering group g are soEnterRows[soEnterStart[g] ...)
    std::vector<int> soEnterRows;
    std::vector<int> soExitStart;
    std::vector<int> soExitRows;
    std::vector<int> soTailExitStart; // tails leaving group g (they all enter at their stratum's first group)
    std::vector<int> soTailExitRows;
    std::vector<int> soTailEnterStart;
    std::vector<int> soTailEnterRows;

    std::vector<double> soDirection; // z = X_F d
    bool soHasDirection = false;
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
    ++layoutVersion;

    // hPidOriginal holds each row's stratum; strata are contiguous
    hPidInternal = hPidOriginal;
    hPid = hPidInternal.data();
    hPidSize = hPidInternal.size();
    accReset.clear();

    const int none = -1;
    std::vector<int> entry(K, none);
    hExitPid.assign(K, none);

    const bool fineGray = this->modelData.hasFineGray();
    if (fineGray && !std::is_same<RealType, double>::value) {
        throw std::logic_error("Fine-Gray rows need double precision");
    }
    const std::vector<int>* rowGroup = fineGray ? &this->modelData.getFineGrayGroupRef() : nullptr;
    fineGrayGroup.assign(fineGray ? K : 0, 0);
    tailBegin.assign(fineGray ? K : 0, 0);
    tailEnd.assign(fineGray ? K : 0, 0);

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
                if (fineGray && (*rowGroup)[k] > 0 && in != groupBegin) { // event times after its stop time
                    fineGrayGroup[k] = (*rowGroup)[k];
                    tailBegin[k] = static_cast<int>(first);
                    tailEnd[k] = static_cast<int>(in - groupTime.begin());
                }
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

    hasTails = false;
    fineGrayGroups = fineGray ? this->modelData.getNumberOfFineGrayGroups() : 0;
    if (fineGray) {
        tailMultiplier.assign(static_cast<size_t>(fineGrayGroups) * N, 1.0);
        for (int g = 1; g <= fineGrayGroups; ++g) {
            for (size_t i = 0; i < N; ++i) {
                tailMultiplier[static_cast<size_t>(g - 1) * N + i] =
                    this->modelData.fineGraySurvivalBefore(g, static_cast<double>(groupTime[i]));
            }
        }
        tailFactor.assign(K, 0.0);
        for (size_t k = 0; k < K; ++k) {
            if (fineGrayGroup[k] > 0) {
                const double before = this->modelData.fineGraySurvivalBefore(fineGrayGroup[k], static_cast<double>(hOffs[k]));
                if (!(before > 0.0)) {
                    throw std::range_error("A Fine-Gray row's censoring survival before its stop time is 0");
                }
                tailFactor[k] = 1.0 / before;
                hasTails = true;
            }
        }
        tailDenom.assign(static_cast<size_t>(fineGrayGroups) * (N + 1), static_cast<RealType>(0));
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

    if (!pool || cumHazard.size() != N || hasTails) {
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

    if (hasTails) {
        fineGrayGradientAndHessian(index, ogradient, ohessian, useWeights);
        return;
    }
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
void CountingProcessModelSpecifics<RealType>::fineGrayGradientAndHessian(int index, double* ogradient,
        double* ohessian, bool useWeights) {

    // Changes of the column's risk-set numerators S1 (u x) and S2 (u x^2) at each group, for ordinary rows
    // (block 0) and each group's tails (block g), then one scan over all groups:
    // gradient = sum_i d_i S1_i / D_i - sum y x, Hessian = sum_i d_i (S2_i / D_i - S1_i^2 / D_i^2)
    const int unused = static_cast<int>(N);
    const size_t stride = N + 1;
    const size_t blocks = static_cast<size_t>(fineGrayGroups) + 1;
    tailWork.assign(2 * blocks * stride, 0.0);
    double* change = tailWork.data();
    double* change2 = tailWork.data() + blocks * stride;
    for (GenericIterator<RealType> it(hX, index); it; ++it) {
        const int k = it.index();
        const double x = static_cast<double>(it.value());
        const double value = (useWeights ? static_cast<double>(hKWeight[k]) : 1.0) *
            static_cast<double>(offsExpXBeta[k]) * x;
        if (hPid[k] < unused) {
            change[hPid[k]] += value;
            change2[hPid[k]] += value * x;
            if (hExitPid[k] < unused) {
                change[hExitPid[k]] -= value;
                change2[hExitPid[k]] -= value * x;
            }
        }
        if (hasTail(k)) {
            const size_t base = static_cast<size_t>(fineGrayGroup[k]) * stride;
            const double tail = tailFactor[k] * value;
            change[base + tailBegin[k]] += tail;
            change[base + tailExit(static_cast<int>(k))] -= tail;
            change2[base + tailBegin[k]] += tail * x;
            change2[base + tailExit(static_cast<int>(k))] -= tail * x;
        }
    }
    std::vector<double> run(blocks, 0.0), run2(blocks, 0.0);
    double gradient = 0.0, hessian = 0.0;
    for (size_t i = 0; i < N; ++i) {
        if (static_cast<int>(i) == groupStratumStart[i]) {
            std::fill(run.begin(), run.end(), 0.0);
            std::fill(run2.begin(), run2.end(), 0.0);
        }
        run[0] += change[i];
        run2[0] += change2[i];
        double s1 = run[0], s2 = run2[0];
        for (size_t g = 1; g < blocks; ++g) {
            run[g] += change[g * stride + i];
            run2[g] += change2[g * stride + i];
            const double m = multiplier(static_cast<int>(g), static_cast<int>(i));
            s1 += m * run[g];
            s2 += m * run2[g];
        }
        const double d = static_cast<double>(hNWeight[i]);
        if (d != 0.0) {
            const double D = static_cast<double>(accDenomPid[i]);
            gradient += d * s1 / D;
            hessian += d * (s2 / D - s1 * s1 / (D * D));
        }
    }
    *ogradient = gradient - static_cast<double>(hXjY[index]);
    *ohessian = hessian;
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeThirdDerivative(int index, double* othird,
                                                                     bool useWeights) {

    if (hasTails) {
        throw std::logic_error("Third derivatives are not yet implemented for Fine-Gray rows");
    }
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
    const bool ranged = IteratorType::isSparse && accDenomPid.size() == N + 1 && cumHazard.size() == N && !hasTails;
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
        if (hasTail(k)) {
            const size_t base = static_cast<size_t>(fineGrayGroup[k] - 1) * (N + 1);
            const RealType change = static_cast<RealType>(tailFactor[k]) * (newEntry - oldEntry);
            tailDenom[base + tailBegin[k]] += change;
            tailDenom[base + tailExit(static_cast<int>(k))] -= change;
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
    if (!hasTails) {
        for (size_t g = 0; g < N; g = groupStratumEnd[g]) {
            accumulateDenominators(g, groupStratumEnd[g]);
        }
        return;
    }
    // Ordinary rows plus each group's tails times m_g(i)
    std::vector<double> run(fineGrayGroups + 1, 0.0);
    for (size_t i = 0; i < N; ++i) {
        if (static_cast<int>(i) == groupStratumStart[i]) {
            std::fill(run.begin(), run.end(), 0.0);
        }
        run[0] += static_cast<double>(denomPid[i]);
        double total = run[0];
        for (int g = 1; g <= fineGrayGroups; ++g) {
            run[g] += static_cast<double>(tailDenom[static_cast<size_t>(g - 1) * (N + 1) + i]);
            total += multiplier(g, static_cast<int>(i)) * run[g];
        }
        accDenomPid[i] = static_cast<RealType>(total);
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
    computeTailHazards();
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeTailHazards() {

    if (!hasTails) {
        tailHazard.clear();
        return;
    }
    tailHazard.assign(static_cast<size_t>(fineGrayGroups) * N, 0.0);
    for (int g = 1; g <= fineGrayGroups; ++g) {
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            if (static_cast<int>(i) == groupStratumStart[i]) {
                sum = 0.0;
            }
            sum += multiplier(g, static_cast<int>(i)) * static_cast<double>(hNWeight[i]) / static_cast<double>(accDenomPid[i]);
            tailHazard[static_cast<size_t>(g - 1) * N + i] = sum;
        }
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
    if (hasTails) {
        std::fill(tailDenom.begin(), tailDenom.end(), static_cast<RealType>(0));
        for (size_t k = 0; k < K; ++k) {
            if (hasTail(static_cast<int>(k))) {
                const RealType value = static_cast<RealType>(tailFactor[k]) *
                    (useWeights ? hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k]);
                const size_t base = static_cast<size_t>(fineGrayGroup[k] - 1) * (N + 1);
                tailDenom[base + tailBegin[k]] += value;
                tailDenom[base + tailExit(static_cast<int>(k))] -= value;
            }
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
    std::vector<double> tails; // each group's tails, as in fineGrayGradientAndHessian()
    if (hasTails) {
        tails.assign(static_cast<size_t>(fineGrayGroups) * (N + 1), 0.0);
        for (size_t k = 0; k < K; ++k) {
            if (hasTail(static_cast<int>(k))) {
                const double value = tailFactor[k] * static_cast<double>(hKWeight[k] * offsExpXBeta[k] * x[k]);
                const size_t base = static_cast<size_t>(fineGrayGroup[k] - 1) * (N + 1);
                tails[base + tailBegin[k]] += value;
                tails[base + tailExit(static_cast<int>(k))] -= value;
            }
        }
    }
    std::vector<double> run(hasTails ? fineGrayGroups : 0, 0.0);

    RealType total = static_cast<RealType>(0);
    auto reset = accReset.begin();
    for (size_t g = 0; g < N; ++g) {
        if (static_cast<size_t>(*reset) == g) { // new stratum
            total = static_cast<RealType>(0);
            std::fill(run.begin(), run.end(), 0.0);
            ++reset;
        }
        total += sums[g];
        double withTails = static_cast<double>(total);
        for (int t = 1; hasTails && t <= fineGrayGroups; ++t) {
            run[t - 1] += tails[static_cast<size_t>(t - 1) * (N + 1) + g];
            withTails += multiplier(t, static_cast<int>(g)) * run[t - 1];
        }
        sums[g] = hasTails ? static_cast<RealType>(withTails) : total;
    }
    return sums;
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::computeFisherInformation(int indexOne, int indexTwo,
                                                                       double* oinfo, bool useWeights) {

    // Case weights (hKWeight, all 1 without weights) enter the risk-set sums and hNWeight the event counts
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
bool CountingProcessModelSpecifics<RealType>::scoreContributions(int index, std::vector<double>& u) {

    // Weighted score residuals (Breslow ties), as coxph's robust variance uses them: row k with weight w_k,
    // r_k = exp(x_k'beta), at risk in groups [entry, end) and event indicator delta_k contributes
    //   u_k = w_k [delta_k (x_k - xbar_entry) - r_k sum_{g in [entry, end)} (x_k - xbar_g) d_g / D_g],
    // xbar_g = S1_g / D_g; the sums over a row's groups are differences of running sums within its stratum
    if (accDenomPid.size() < N || hNWeight.size() < N || offsExpXBeta.size() < K) {
        return false;
    }
    const std::vector<RealType> x = denseColumn(index);
    const std::vector<RealType> numer = riskSetSums(x);
    std::vector<double> hazard(N); // running sums within the stratum of d_g / D_g
    std::vector<double> weighted(N); // and of xbar_g d_g / D_g
    // A row's tail sums m_t(g) d_g / D_g (tailHazardSum()) and m_t(g) xbar_g d_g / D_g over its groups
    std::vector<double> tailWeighted(static_cast<size_t>(fineGrayGroups) * N);
    for (size_t g = 0; g < N; ++g) {
        const double denom = static_cast<double>(accDenomPid[g]);
        const double increment = static_cast<double>(hNWeight[g]) / denom;
        const bool first = static_cast<int>(g) == groupStratumStart[g];
        hazard[g] = (first ? 0.0 : hazard[g - 1]) + increment;
        weighted[g] = (first ? 0.0 : weighted[g - 1]) + increment * static_cast<double>(numer[g]) / denom;
        for (int t = 1; t <= fineGrayGroups; ++t) {
            const size_t at = static_cast<size_t>(t - 1) * N + g;
            tailWeighted[at] = (first ? 0.0 : tailWeighted[at - 1]) +
                multiplier(t, static_cast<int>(g)) * increment * static_cast<double>(numer[g]) / denom;
        }
    }
    if (hasTails) {
        computeTailHazards(); // at the current denominators
    }

    const int unused = static_cast<int>(N);
    u.assign(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        const double w = static_cast<double>(hKWeight[k]);
        if (!participates(static_cast<int>(k)) || w == 0.0) { // outside every risk set
            continue;
        }
        const double xk = static_cast<double>(x[k]);
        double value = 0.0;
        if (hPid[k] < unused) {
            const int entry = hPid[k];
            const int last = rowEnd(static_cast<int>(k)) - 1;
            const bool first = entry == groupStratumStart[entry];
            const double dHazard = hazard[last] - (first ? 0.0 : hazard[entry - 1]);
            const double dWeighted = weighted[last] - (first ? 0.0 : weighted[entry - 1]);
            value = -static_cast<double>(offsExpXBeta[k]) * (xk * dHazard - dWeighted);
            if (hY[k] != static_cast<RealType>(0)) {
                value += xk - static_cast<double>(numer[entry]) / static_cast<double>(accDenomPid[entry]);
            }
        }
        if (hasTail(static_cast<int>(k))) { // tails start at the stratum's first group
            const double dWeighted = tailWeighted[static_cast<size_t>(fineGrayGroup[k] - 1) * N + tailEnd[k] - 1];
            value -= static_cast<double>(offsExpXBeta[k]) * tailFactor[k] * (xk * tailHazardSum(static_cast<int>(k)) - dWeighted);
        }
        u[k] = w * value;
    }
    return true;
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

// Neumaier-compensated running sum: the total is sum + compensation
inline void countingAddCompensated(double& sum, double& compensation, double value) {
    const double total = sum + value;
    compensation += (std::abs(sum) >= std::abs(value)) ? (sum - total) + value : (value - total) + sum;
    sum = total;
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::secondOrderReady() const {
    return secondOrder && hasLayout && cumHazard.size() == N && accDenomPid.size() == N + 1 &&
        hKWeight.size() == K && hNWeight.size() >= N && offsExpXBeta.size() >= K &&
        (!hasTails || tailHazard.size() == static_cast<size_t>(fineGrayGroups) * N);
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::secondOrderPrepare(const std::vector<int>& F) {

    if (!secondOrder) {
        return false;
    }
    if (soPrepared && F == soF) {
        return true;
    }
    for (int j : F) {
        if (j < 0 || static_cast<size_t>(j) >= J) {
            return false;
        }
    }

    // Row-wise copy of X_F: count, then fill column by column so positions ascend within a row
    soRowStart.assign(K + 1, 0);
    for (size_t p = 0; p < F.size(); ++p) {
        for (GenericIterator<RealType> it(hX, F[p]); it; ++it) {
            ++soRowStart[it.index() + 1];
        }
    }
    for (size_t k = 0; k < K; ++k) {
        soRowStart[k + 1] += soRowStart[k];
    }
    soPosition.resize(soRowStart[K]);
    soValue.resize(soRowStart[K]);
    std::vector<size_t> next(soRowStart.begin(), soRowStart.end() - 1);
    for (size_t p = 0; p < F.size(); ++p) {
        for (GenericIterator<RealType> it(hX, F[p]); it; ++it) {
            const size_t e = next[it.index()]++;
            soPosition[e] = static_cast<int>(p);
            soValue[e] = static_cast<double>(it.value());
        }
    }

    soF = F;
    soPrepared = true;
    soHasDirection = false;
    return true;
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::prepareGroupLists() {

    if (soGroupVersion == layoutVersion && soEnterStart.size() == N + 1) {
        return;
    }
    const int unused = static_cast<int>(N);
    soEnterStart.assign(N + 1, 0);
    soExitStart.assign(N + 1, 0);
    for (size_t k = 0; k < K; ++k) {
        if (hPid[k] < unused) {
            ++soEnterStart[hPid[k] + 1];
            if (hExitPid[k] < unused) {
                ++soExitStart[hExitPid[k] + 1];
            }
        }
    }
    for (size_t g = 0; g < N; ++g) {
        soEnterStart[g + 1] += soEnterStart[g];
        soExitStart[g + 1] += soExitStart[g];
    }
    soEnterRows.resize(soEnterStart[N]);
    soExitRows.resize(soExitStart[N]);
    std::vector<int> nextEnter(soEnterStart.begin(), soEnterStart.end() - 1);
    std::vector<int> nextExit(soExitStart.begin(), soExitStart.end() - 1);
    for (size_t k = 0; k < K; ++k) {
        if (hPid[k] < unused) {
            soEnterRows[nextEnter[hPid[k]]++] = static_cast<int>(k);
            if (hExitPid[k] < unused) {
                soExitRows[nextExit[hExitPid[k]]++] = static_cast<int>(k);
            }
        }
    }
    // Tails enter at their stratum's first group and leave at tailEnd
    soTailEnterStart.assign(N + 1, 0);
    soTailExitStart.assign(N + 1, 0);
    for (size_t k = 0; k < K; ++k) {
        if (hasTail(static_cast<int>(k))) {
            ++soTailEnterStart[tailBegin[k] + 1];
            if (tailExit(static_cast<int>(k)) < static_cast<int>(N)) {
                ++soTailExitStart[tailExit(static_cast<int>(k)) + 1];
            }
        }
    }
    for (size_t g = 0; g < N; ++g) {
        soTailEnterStart[g + 1] += soTailEnterStart[g];
        soTailExitStart[g + 1] += soTailExitStart[g];
    }
    soTailEnterRows.resize(soTailEnterStart[N]);
    soTailExitRows.resize(soTailExitStart[N]);
    std::vector<int> nextTailEnter(soTailEnterStart.begin(), soTailEnterStart.end() - 1);
    std::vector<int> nextTailExit(soTailExitStart.begin(), soTailExitStart.end() - 1);
    for (size_t k = 0; k < K; ++k) {
        if (hasTail(static_cast<int>(k))) {
            soTailEnterRows[nextTailEnter[tailBegin[k]]++] = static_cast<int>(k);
            if (tailExit(static_cast<int>(k)) < static_cast<int>(N)) {
                soTailExitRows[nextTailExit[tailExit(static_cast<int>(k))]++] = static_cast<int>(k);
            }
        }
    }
    soGroupVersion = layoutVersion;
}

template <typename RealType>
double CountingProcessModelSpecifics<RealType>::rowHazard(int k) const {
    double hazard = 0.0;
    const int entry = hPid[k];
    if (entry < static_cast<int>(N)) {
        const double before = (entry == groupStratumStart[entry]) ? 0.0 : static_cast<double>(cumHazard[entry - 1]);
        hazard = static_cast<double>(cumHazard[rowEnd(k) - 1]) - before;
    }
    if (hasTail(k)) {
        hazard += tailFactor[k] * tailHazardSum(k);
    }
    return hazard;
}

template <typename RealType>
void CountingProcessModelSpecifics<RealType>::productXv(const std::vector<double>& v,
                                                        std::vector<double>& z) const {
    z.assign(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        double sum = 0.0;
        for (size_t e = soRowStart[k]; e < soRowStart[k + 1]; ++e) {
            sum += soValue[e] * v[soPosition[e]];
        }
        z[k] = sum;
    }
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::fullGradient(const std::vector<int>& F, std::vector<double>& g) {

    if (!secondOrderReady() || !secondOrderPrepare(F)) {
        return false;
    }
    const int unused = static_cast<int>(N);
    g.assign(F.size(), 0.0);
    for (size_t k = 0; k < K; ++k) {
        if (participates(static_cast<int>(k))) {
            const double r = static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]) *
                rowHazard(static_cast<int>(k));
            for (size_t e = soRowStart[k]; e < soRowStart[k + 1]; ++e) {
                g[soPosition[e]] += soValue[e] * r;
            }
        }
    }
    for (size_t p = 0; p < F.size(); ++p) {
        g[p] -= static_cast<double>(hXjY[F[p]]);
    }
    return true;
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::hessianBlock(const std::vector<int>& F, std::vector<double>& H) {

    if (!secondOrderReady() || !secondOrderPrepare(F)) {
        return false;
    }
    prepareGroupLists();
    const int unused = static_cast<int>(N);
    const size_t n = F.size();
    H.assign(n * n, 0.0);
    if (n == 0) {
        return true;
    }

    // S2 = X_F' diag(u o dC) X_F, lower triangle. Row p of H is owned by one task: it walks column F[p] and, in
    // each of its rows k, the entries at positions <= p, so every entry sums over k in increasing order whatever
    // the threads
    std::vector<double> r(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        if (participates(static_cast<int>(k))) {
            r[k] = static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]) * rowHazard(static_cast<int>(k));
        }
    }
    auto rowS2 = [&](size_t p, int) {
        double* row = H.data() + p;
        for (GenericIterator<RealType> it(hX, F[p]); it; ++it) {
            const int k = it.index();
            if (!participates(k)) {
                continue;
            }
            const double rp = r[k] * static_cast<double>(it.value());
            for (size_t b = soRowStart[k]; b < soRowStart[k + 1] && static_cast<size_t>(soPosition[b]) <= p; ++b) {
                row[soPosition[b] * n] += rp * soValue[b];
            }
        }
    };
    if (pool) {
        pool->run(n, rowS2);
    } else { // the same sums in the same order, from each row's outer product (faster on one thread)
        for (size_t k = 0; k < K; ++k) {
            if (participates(static_cast<int>(k))) {
                for (size_t a = soRowStart[k]; a < soRowStart[k + 1]; ++a) {
                    const double ra = r[k] * soValue[a];
                    double* column = H.data() + soPosition[a];
                    for (size_t b = soRowStart[k]; b <= a; ++b) {
                        column[soPosition[b] * n] += ra * soValue[b];
                    }
                }
            }
        }
    }

    // A'A: row g of A is sqrt(d_g) / D_g times the risk-set sum S1_g of u x_F, a compensated running sum over
    // entries and exits. Chunks of groups go to fixed-width block-column products of the lower triangle, so the
    // result does not depend on the threads.
    const size_t chunk = std::max<size_t>(1, std::min<size_t>(N, std::min<size_t>(1024, (size_t(1) << 20) / n))); // <= 8 MB
    const size_t width = 64; // block columns of the lower triangle, one task each
    const size_t blocks = (n + width - 1) / width;
    Eigen::MatrixXd A(n, chunk);
    Eigen::Map<Eigen::MatrixXd> Hm(H.data(), n, n);
    std::vector<double> run(n, 0.0);
    std::vector<double> compensation(n, 0.0);
    size_t filled = 0;
    auto subtract = [&](size_t columns) {
        auto block = [&](size_t b, int) {
            const size_t c0 = b * width;
            const size_t w = std::min(width, n - c0);
            Hm.block(c0, c0, n - c0, w).noalias() -=
                A.block(c0, 0, n - c0, columns) * A.block(c0, 0, w, columns).transpose();
        };
        if (pool) {
            pool->run(blocks, block);
        } else {
            for (size_t b = 0; b < blocks; ++b) {
                block(b, 0);
            }
        }
    };
    auto scatter = [&](int k, double sign) {
        const double u = sign * static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]);
        for (size_t e = soRowStart[k]; e < soRowStart[k + 1]; ++e) {
            countingAddCompensated(run[soPosition[e]], compensation[soPosition[e]], u * soValue[e]);
        }
    };
    // Each Fine-Gray group's tails: S1_g = (ordinary rows) + sum_t m_t(g) (tails of group t)
    std::vector<double> tailRun(static_cast<size_t>(hasTails ? fineGrayGroups : 0) * n, 0.0);
    std::vector<double> tailCompensation(tailRun.size(), 0.0);
    auto scatterTail = [&](int k, double sign) {
        const double u = sign * tailFactor[k] * static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]);
        const size_t base = static_cast<size_t>(fineGrayGroup[k] - 1) * n;
        for (size_t e = soRowStart[k]; e < soRowStart[k + 1]; ++e) {
            countingAddCompensated(tailRun[base + soPosition[e]], tailCompensation[base + soPosition[e]], u * soValue[e]);
        }
    };
    for (size_t g = 0; g < N; ++g) {
        if (static_cast<int>(g) == groupStratumStart[g]) {
            std::fill(run.begin(), run.end(), 0.0);
            std::fill(compensation.begin(), compensation.end(), 0.0);
            std::fill(tailRun.begin(), tailRun.end(), 0.0);
            std::fill(tailCompensation.begin(), tailCompensation.end(), 0.0);
        }
        for (int i = soEnterStart[g]; i < soEnterStart[g + 1]; ++i) {
            scatter(soEnterRows[i], 1.0);
        }
        for (int i = soExitStart[g]; i < soExitStart[g + 1]; ++i) {
            scatter(soExitRows[i], -1.0);
        }
        if (hasTails) {
            for (int i = soTailEnterStart[g]; i < soTailEnterStart[g + 1]; ++i) {
                scatterTail(soTailEnterRows[i], 1.0);
            }
            for (int i = soTailExitStart[g]; i < soTailExitStart[g + 1]; ++i) {
                scatterTail(soTailExitRows[i], -1.0);
            }
        }
        const double d = static_cast<double>(hNWeight[g]);
        if (d != 0.0) {
            const double scale = std::sqrt(d) / static_cast<double>(accDenomPid[g]);
            for (size_t p = 0; p < n; ++p) {
                double s1 = run[p] + compensation[p];
                for (int t = 1; hasTails && t <= fineGrayGroups; ++t) {
                    const size_t at = static_cast<size_t>(t - 1) * n + p;
                    s1 += multiplier(t, static_cast<int>(g)) * (tailRun[at] + tailCompensation[at]);
                }
                A(p, filled) = s1 * scale;
            }
            if (++filled == chunk) {
                subtract(filled);
                filled = 0;
            }
        }
    }
    if (filled > 0) {
        subtract(filled);
    }

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = j + 1; i < n; ++i) {
            H[j + i * n] = H[i + j * n];
        }
    }
    return true;
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::hessianVector(const std::vector<int>& F,
        const std::vector<double>& v, std::vector<double>& Hv) {

    if (!secondOrderReady() || !secondOrderPrepare(F) || v.size() != F.size()) {
        return false;
    }
    const int unused = static_cast<int>(N);
    std::vector<double> z;
    productXv(v, z);

    // M_g = risk-set sum of u z; Q = running sum of d_g M_g / D_g^2 within each stratum (and, for the tails of
    // Fine-Gray group t, of m_t(g) d_g M_g / D_g^2)
    std::vector<double> M(N + 1, 0.0);
    const size_t groups = hasTails ? static_cast<size_t>(fineGrayGroups) : 0;
    std::vector<double> tailM(groups * (N + 1), 0.0);
    for (size_t k = 0; k < K; ++k) {
        if (hPid[k] < unused) {
            const double t = static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]) * z[k];
            M[hPid[k]] += t;
            M[hExitPid[k]] -= t; // group N is unused
        }
        if (hasTail(static_cast<int>(k))) {
            const double t = tailFactor[k] * static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]) * z[k];
            const size_t base = static_cast<size_t>(fineGrayGroup[k] - 1) * (N + 1);
            tailM[base + tailBegin[k]] += t;
            tailM[base + tailExit(static_cast<int>(k))] -= t;
        }
    }
    std::vector<double> Q(N);
    std::vector<double> tailQ(groups * N);
    std::vector<double> tailRun(groups, 0.0), tailRunCompensation(groups, 0.0), tq(groups, 0.0), tqCompensation(groups, 0.0);
    double run = 0.0, runCompensation = 0.0, q = 0.0, qCompensation = 0.0;
    for (size_t g = 0; g < N; ++g) {
        if (static_cast<int>(g) == groupStratumStart[g]) {
            run = runCompensation = q = qCompensation = 0.0;
            std::fill(tailRun.begin(), tailRun.end(), 0.0);
            std::fill(tailRunCompensation.begin(), tailRunCompensation.end(), 0.0);
            std::fill(tq.begin(), tq.end(), 0.0);
            std::fill(tqCompensation.begin(), tqCompensation.end(), 0.0);
        }
        countingAddCompensated(run, runCompensation, M[g]);
        double sum = run + runCompensation;
        for (size_t t = 0; t < groups; ++t) {
            countingAddCompensated(tailRun[t], tailRunCompensation[t], tailM[t * (N + 1) + g]);
            sum += multiplier(static_cast<int>(t) + 1, static_cast<int>(g)) * (tailRun[t] + tailRunCompensation[t]);
        }
        const double D = static_cast<double>(accDenomPid[g]);
        const double increment = static_cast<double>(hNWeight[g]) * sum / (D * D);
        countingAddCompensated(q, qCompensation, increment);
        Q[g] = q + qCompensation;
        for (size_t t = 0; t < groups; ++t) {
            countingAddCompensated(tq[t], tqCompensation[t], multiplier(static_cast<int>(t) + 1, static_cast<int>(g)) * increment);
            tailQ[t * N + g] = tq[t] + tqCompensation[t];
        }
    }

    Hv.assign(F.size(), 0.0);
    for (size_t k = 0; k < K; ++k) {
        if (participates(static_cast<int>(k))) {
            const int entry = hPid[k];
            double segment = (entry < unused) ? Q[rowEnd(static_cast<int>(k)) - 1] -
                ((entry == groupStratumStart[entry]) ? 0.0 : Q[entry - 1]) : 0.0;
            if (hasTail(static_cast<int>(k))) {
                segment += tailFactor[k] * tailQ[static_cast<size_t>(fineGrayGroup[k] - 1) * N + tailEnd[k] - 1];
            }
            const double r = static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]) *
                (z[k] * rowHazard(static_cast<int>(k)) - segment);
            for (size_t e = soRowStart[k]; e < soRowStart[k + 1]; ++e) {
                Hv[soPosition[e]] += soValue[e] * r;
            }
        }
    }
    return true;
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::setDirection(const std::vector<int>& F, const std::vector<double>& d) {

    if (!secondOrder || !secondOrderPrepare(F) || d.size() != F.size()) {
        return false;
    }
    productXv(d, soDirection);
    soHasDirection = true;
    return true;
}

template <typename RealType>
double CountingProcessModelSpecifics<RealType>::trialLogLikelihood(double t) {

    if (!secondOrderReady() || !soHasDirection) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int unused = static_cast<int>(N);

    double numerator = 0.0, numeratorCompensation = 0.0;
    for (size_t k = 0; k < K; ++k) {
        if (hY[k] == static_cast<RealType>(1)) {
            countingAddCompensated(numerator, numeratorCompensation,
                static_cast<double>(hKWeight[k]) * (static_cast<double>(hXBeta[k]) + t * soDirection[k]));
        }
    }

    // log D_g(t), each stratum shifted by its largest x'beta + t z
    std::vector<double> shift(N, -std::numeric_limits<double>::infinity());
    auto stratum = [this, unused](size_t k) { // the first group of row k's stratum
        return hPid[k] < unused ? groupStratumStart[hPid[k]] : tailBegin[k];
    };
    for (size_t k = 0; k < K; ++k) {
        if (participates(static_cast<int>(k))) {
            const int s = stratum(k);
            shift[s] = std::max(shift[s], static_cast<double>(hXBeta[k]) + t * soDirection[k]);
        }
    }
    std::vector<double> E(N + 1, 0.0);
    const size_t groups = hasTails ? static_cast<size_t>(fineGrayGroups) : 0;
    std::vector<double> tailE(groups * (N + 1), 0.0);
    for (size_t k = 0; k < K; ++k) {
        if (participates(static_cast<int>(k))) {
            const double eta = static_cast<double>(hXBeta[k]) + t * soDirection[k];
            const double value = static_cast<double>(hKWeight[k]) * std::exp(eta - shift[stratum(k)]);
            if (hPid[k] < unused) {
                E[hPid[k]] += value;
                E[hExitPid[k]] -= value; // group N is unused
            }
            if (hasTail(static_cast<int>(k))) {
                const size_t base = static_cast<size_t>(fineGrayGroup[k] - 1) * (N + 1);
                tailE[base + tailBegin[k]] += tailFactor[k] * value;
                tailE[base + tailExit(static_cast<int>(k))] -= tailFactor[k] * value;
            }
        }
    }
    double denominator = 0.0, denominatorCompensation = 0.0;
    double run = 0.0, runCompensation = 0.0;
    std::vector<double> tailRun(groups, 0.0), tailCompensation(groups, 0.0);
    for (size_t g = 0; g < N; ++g) {
        if (static_cast<int>(g) == groupStratumStart[g]) {
            run = runCompensation = 0.0;
            std::fill(tailRun.begin(), tailRun.end(), 0.0);
            std::fill(tailCompensation.begin(), tailCompensation.end(), 0.0);
        }
        countingAddCompensated(run, runCompensation, E[g]);
        double sum = run + runCompensation;
        for (size_t s = 0; s < groups; ++s) {
            countingAddCompensated(tailRun[s], tailCompensation[s], tailE[s * (N + 1) + g]);
            sum += multiplier(static_cast<int>(s) + 1, static_cast<int>(g)) * (tailRun[s] + tailCompensation[s]);
        }
        const double d = static_cast<double>(hNWeight[g]);
        if (d != 0.0) {
            countingAddCompensated(denominator, denominatorCompensation,
                d * (shift[groupStratumStart[g]] + std::log(sum)));
        }
    }
    return (numerator + numeratorCompensation) - (denominator + denominatorCompensation);
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::commitStep(double t) {

    if (!secondOrderReady() || !soHasDirection) {
        return false;
    }
    for (size_t k = 0; k < K; ++k) {
        hXBeta[k] += static_cast<RealType>(t * soDirection[k]);
    }
    computeRemainingStatistics(true); // with weights 1 when the fit has none
    this->makeDirty();
    return true;
}

template <typename RealType>
double CountingProcessModelSpecifics<RealType>::directionMaxAbs() const {

    if (!soHasDirection) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int unused = static_cast<int>(N);
    double largest = 0.0;
    for (size_t k = 0; k < K; ++k) {
        if (hPid[k] < unused || hasTail(static_cast<int>(k))) { // rows outside every risk set do not enter the likelihood's sums
            largest = std::max(largest, std::abs(soDirection[k]));
        }
    }
    return largest;
}

template <typename RealType>
double CountingProcessModelSpecifics<RealType>::newtonCostRatio(const std::vector<int>& F) {

    if (!secondOrderReady() || !secondOrderPrepare(F)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Per-unit costs in ns (arm64, Newton plan Phase 0a/2): a CCD visit per entry and per risk-set group the
    // column touches; for one dense iteration, row outer products, the blocked A'A, the Cholesky factor and the
    // O(nnz + K + G) passes of the gradient, two trials and the commit
    const double aCycle = 4.0, aRows = 0.75, aGroups = 0.03, aCholesky = 0.035, aLinear = 1.0;
    const double cycle = cycleWork(F);
    double rows = 0.0;
    for (size_t k = 0; k < K; ++k) {
        const double m = static_cast<double>(soRowStart[k + 1] - soRowStart[k]);
        rows += 0.5 * m * m;
    }
    const double f = static_cast<double>(F.size());
    const double linear = static_cast<double>(soRowStart[K]) + static_cast<double>(K) + static_cast<double>(N);
    const double newton = aRows * rows + aGroups * static_cast<double>(N) * f * f + aCholesky * f * f * f / 3.0 +
        4.0 * aLinear * linear;
    return (cycle > 0.0) ? newton / (aCycle * cycle) : std::numeric_limits<double>::quiet_NaN();
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::separationDirections(const std::vector<int>& columns,
                                                                   std::vector<int>& direction) {

    // Each event's x is at least the risk-set minimum and at most its maximum, so the observed sum of x over
    // events equals sum_g d_g min_{R_g} x exactly when every event sits at its risk set's minimum: the
    // likelihood then does not increase in the coefficient (MLE at -Inf); likewise for the maximum
    if (!hasLayout || hNWeight.size() < N || hXjY.size() < J) {
        return false;
    }
    const int unused = static_cast<int>(N);
    std::vector<int> atRisk(N + 1, 0); // rows at risk in each group: entries minus exits, summed within strata
    for (size_t k = 0; k < K; ++k) {
        if (hPid[k] < unused) {
            ++atRisk[hPid[k]];
            --atRisk[hExitPid[k]]; // group N is unused
        }
        if (hasTail(static_cast<int>(k))) { // a Fine-Gray tail is at risk too (with a positive multiplier)
            ++atRisk[tailBegin[k]];
            --atRisk[tailExit(static_cast<int>(k))];
        }
    }
    for (size_t g = 0, count = 0; g < N; ++g) {
        if (static_cast<int>(g) == groupStratumStart[g]) {
            count = 0;
        }
        count += atRisk[g];
        atRisk[g] = static_cast<int>(count);
    }
    direction.assign(columns.size(), 0);
    struct Change { int group; double value; bool enters; };
    std::vector<Change> changes;
    std::vector<int> delta(N + 1, 0); // single-valued columns: entries minus exits of non-zero rows by group
    std::vector<int> touched;
    for (size_t c = 0; c < columns.size(); ++c) {
        const int j = columns[c];
        const double observed = static_cast<double>(hXjY[j]);

        // A column whose non-zero entries share one value v (indicators, mostly) has risk-set extremes that follow
        // from two counts: v where it is at risk, 0 where a row without it is. The same sums, without a sort or a
        // multiset (separation is checked on every fit, and every bootstrap replicate)
        double v = 0.0;
        bool single = true;
        for (GenericIterator<RealType> it(hX, j); it && single; ++it) {
            const double x = static_cast<double>(it.value());
            if (x != 0.0) {
                single = (v == 0.0 || x == v);
                v = x;
            }
        }
        if (single) {
            touched.clear();
            for (GenericIterator<RealType> it(hX, j); it; ++it) {
                const int k = it.index();
                if (hPid[k] < unused && it.value() != static_cast<RealType>(0)) {
                    ++delta[hPid[k]];
                    touched.push_back(hPid[k]);
                    if (hExitPid[k] < unused) {
                        --delta[hExitPid[k]];
                        touched.push_back(hExitPid[k]);
                    }
                }
                if (hasTail(k) && it.value() != static_cast<RealType>(0)) {
                    ++delta[tailBegin[k]];
                    --delta[tailExit(k)];
                    touched.push_back(tailBegin[k]);
                    touched.push_back(tailExit(k));
                }
            }
            double lowest = 0.0, highest = 0.0, scale = 0.0;
            int count = 0;
            for (size_t g = 0; g < N; ++g) {
                if (static_cast<int>(g) == groupStratumStart[g]) {
                    count = 0;
                }
                count += delta[g];
                const double d = static_cast<double>(hNWeight[g]);
                if (d == 0.0 || atRisk[g] == 0 || count == 0) {
                    continue; // no non-zero row at risk: x = 0 throughout
                }
                const bool zeros = count < atRisk[g];
                const double low = zeros ? std::min(v, 0.0) : v;
                const double high = zeros ? std::max(v, 0.0) : v;
                lowest += d * low;
                highest += d * high;
                scale += d * std::max(std::abs(low), std::abs(high));
            }
            for (int g : touched) {
                delta[g] = 0;
            }
            const double tolerance = 1e-12 * std::max(scale, 1.0);
            if (highest - lowest > tolerance) {
                if (std::abs(observed - lowest) <= tolerance) {
                    direction[c] = -1;
                } else if (std::abs(observed - highest) <= tolerance) {
                    direction[c] = 1;
                }
            }
            continue;
        }

        changes.clear();
        for (GenericIterator<RealType> it(hX, j); it; ++it) {
            const int k = it.index();
            const double x = static_cast<double>(it.value());
            if (hPid[k] < unused && x != 0.0) {
                changes.push_back({hPid[k], x, true});
                if (hExitPid[k] < unused) {
                    changes.push_back({hExitPid[k], x, false});
                }
            }
            if (hasTail(k) && x != 0.0) {
                changes.push_back({tailBegin[k], x, true});
                if (tailExit(k) < unused) {
                    changes.push_back({tailExit(k), x, false});
                }
            }
        }
        std::stable_sort(changes.begin(), changes.end(), [](const Change& a, const Change& b) { return a.group < b.group; });
        std::multiset<double> values; // non-zero x at risk
        double lowest = 0.0, highest = 0.0, scale = 0.0;
        size_t next = 0;
        for (size_t g = 0; g < N; ++g) {
            if (static_cast<int>(g) == groupStratumStart[g]) {
                values.clear();
            }
            for (; next < changes.size() && changes[next].group == static_cast<int>(g); ++next) {
                if (changes[next].enters) {
                    values.insert(changes[next].value);
                } else {
                    values.erase(values.find(changes[next].value));
                }
            }
            const double d = static_cast<double>(hNWeight[g]);
            if (d == 0.0 || atRisk[g] == 0) {
                continue;
            }
            const bool zeros = static_cast<int>(values.size()) < atRisk[g]; // rows at risk with x = 0
            double low = values.empty() ? 0.0 : *values.begin();
            double high = values.empty() ? 0.0 : *values.rbegin();
            if (zeros) {
                low = std::min(low, 0.0);
                high = std::max(high, 0.0);
            }
            lowest += d * low;
            highest += d * high;
            scale += d * std::max(std::abs(low), std::abs(high));
        }
        const double tolerance = 1e-12 * std::max(scale, 1.0);
        if (highest - lowest > tolerance) { // the column varies within some risk set
            if (std::abs(observed - lowest) <= tolerance) {
                direction[c] = -1;
            } else if (std::abs(observed - highest) <= tolerance) {
                direction[c] = 1;
            }
        }
    }
    return true;
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::hessianDiagonal(const std::vector<int>& F, std::vector<double>& h) {

    if (!secondOrderReady()) {
        return false;
    }
    h.assign(F.size(), 0.0);
    for (size_t p = 0; p < F.size(); ++p) {
        double gradient = 0.0;
        if (hasTails) {
            fineGrayGradientAndHessian(F[p], &gradient, &h[p], true);
        } else if (hX.getNumberOfNonZeroEntries(F[p]) > 0) {
            dispatchHazardGradientAndHessian(F[p], workspace, &gradient, &h[p], true); // weights are 1 without any
        }
    }
    return true;
}

template <typename RealType>
bool CountingProcessModelSpecifics<RealType>::woodburyParts(const std::vector<int>& F, int rank,
        std::vector<double>& S2, std::vector<double>& B) {

    if (!secondOrderReady() || !secondOrderPrepare(F) || rank < 1) {
        return false;
    }
    prepareGroupLists();
    const int unused = static_cast<int>(N);
    const size_t n = F.size();
    S2.assign(n * n, 0.0);
    for (size_t k = 0; k < K; ++k) { // as hessianBlock()
        if (participates(static_cast<int>(k))) {
            const double r = static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]) *
                rowHazard(static_cast<int>(k));
            for (size_t a = soRowStart[k]; a < soRowStart[k + 1]; ++a) {
                const double ra = r * soValue[a];
                double* column = S2.data() + soPosition[a];
                for (size_t b = soRowStart[k]; b <= a; ++b) {
                    column[soPosition[b] * n] += ra * soValue[b];
                }
            }
        }
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = j + 1; i < n; ++i) {
            S2[j + i * n] = S2[i + j * n];
        }
    }

    // bins of contiguous groups with about equal event weight
    const int bins = static_cast<int>(std::min<size_t>(rank, N));
    double total = 0.0;
    for (size_t g = 0; g < N; ++g) {
        total += static_cast<double>(hNWeight[g]);
    }
    std::vector<int> bin(N), count(bins, 0);
    double accumulated = 0.0;
    for (size_t g = 0; g < N; ++g) {
        bin[g] = std::min(bins - 1, static_cast<int>(std::floor(accumulated / total * bins)));
        ++count[bin[g]];
        accumulated += static_cast<double>(hNWeight[g]);
    }
    B.assign(static_cast<size_t>(bins) * n, 0.0);
    std::vector<double> run(n, 0.0), compensation(n, 0.0);
    auto scatter = [&](int k, double sign) {
        const double u = sign * static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]);
        for (size_t e = soRowStart[k]; e < soRowStart[k + 1]; ++e) {
            countingAddCompensated(run[soPosition[e]], compensation[soPosition[e]], u * soValue[e]);
        }
    };
    std::vector<double> tailRun(static_cast<size_t>(hasTails ? fineGrayGroups : 0) * n, 0.0);
    std::vector<double> tailCompensation(tailRun.size(), 0.0);
    auto scatterTail = [&](int k, double sign) {
        const double u = sign * tailFactor[k] * static_cast<double>(hKWeight[k]) * static_cast<double>(offsExpXBeta[k]);
        const size_t base = static_cast<size_t>(fineGrayGroup[k] - 1) * n;
        for (size_t e = soRowStart[k]; e < soRowStart[k + 1]; ++e) {
            countingAddCompensated(tailRun[base + soPosition[e]], tailCompensation[base + soPosition[e]], u * soValue[e]);
        }
    };
    for (size_t g = 0; g < N; ++g) { // the rows of A, as in hessianBlock(), summed into their bins
        if (static_cast<int>(g) == groupStratumStart[g]) {
            std::fill(run.begin(), run.end(), 0.0);
            std::fill(compensation.begin(), compensation.end(), 0.0);
            std::fill(tailRun.begin(), tailRun.end(), 0.0);
            std::fill(tailCompensation.begin(), tailCompensation.end(), 0.0);
        }
        for (int i = soEnterStart[g]; i < soEnterStart[g + 1]; ++i) {
            scatter(soEnterRows[i], 1.0);
        }
        for (int i = soExitStart[g]; i < soExitStart[g + 1]; ++i) {
            scatter(soExitRows[i], -1.0);
        }
        if (hasTails) {
            for (int i = soTailEnterStart[g]; i < soTailEnterStart[g + 1]; ++i) {
                scatterTail(soTailEnterRows[i], 1.0);
            }
            for (int i = soTailExitStart[g]; i < soTailExitStart[g + 1]; ++i) {
                scatterTail(soTailExitRows[i], -1.0);
            }
        }
        const double d = static_cast<double>(hNWeight[g]);
        if (d != 0.0) {
            const double scale = std::sqrt(d) / static_cast<double>(accDenomPid[g]) / std::sqrt(static_cast<double>(count[bin[g]]));
            for (size_t p = 0; p < n; ++p) {
                double s1 = run[p] + compensation[p];
                for (int t = 1; hasTails && t <= fineGrayGroups; ++t) {
                    const size_t at = static_cast<size_t>(t - 1) * n + p;
                    s1 += multiplier(t, static_cast<int>(g)) * (tailRun[at] + tailCompensation[at]);
                }
                B[bin[g] + p * bins] += s1 * scale;
            }
        }
    }
    return true;
}

template <typename RealType>
double CountingProcessModelSpecifics<RealType>::cycleWork(const std::vector<int>& F) const {
    double cycle = 0.0;
    for (int j : F) {
        const bool dense = hX.getFormatType(j) == DENSE || hX.getFormatType(j) == INTERCEPT;
        const double groups = dense || !sparseIndices[j] ? static_cast<double>(N) :
            static_cast<double>(sparseIndices[j]->size());
        cycle += static_cast<double>(hX.getNumberOfNonZeroEntries(j)) + groups;
    }
    return cycle;
}

template <typename RealType>
double CountingProcessModelSpecifics<RealType>::newtonCgCostRatio(const std::vector<int>& F, bool preconditioned) {

    if (!secondOrderReady() || !secondOrderPrepare(F)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Unpreconditioned: about 45 Hv per iteration (E5). With the rank-32 Woodbury preconditioner (Phase 4 study):
    // about 12 Hv per iteration, each with two triangular solves, and its setup (S2's row outer products, a
    // Cholesky factor, 32 solves) spread over about 6 iterations
    const double aCycle = 4.0, aLinear = 1.0, aRows = 0.75, aCholesky = 0.035, aSolve = 0.1;
    const double hv = 2.0 * static_cast<double>(soRowStart[K]) + static_cast<double>(K) + static_cast<double>(N);
    const double cycle = cycleWork(F);
    const double f = static_cast<double>(F.size());
    double work = 0.0;
    if (preconditioned) {
        double rows = 0.0;
        for (size_t k = 0; k < K; ++k) {
            const double m = static_cast<double>(soRowStart[k + 1] - soRowStart[k]);
            rows += 0.5 * m * m;
        }
        const double setup = aRows * rows + aCholesky * f * f * f / 3.0 + aSolve * 32.0 * 2.0 * f * f;
        work = 16.0 * aLinear * hv + 12.0 * aSolve * 2.0 * f * f + setup / 6.0;
    } else {
        work = 49.0 * aLinear * hv;
    }
    return (cycle > 0.0) ? work / (aCycle * cycle) : std::numeric_limits<double>::quiet_NaN();
}

} // namespace bsccs

#endif /* COUNTINGPROCESSMODELSPECIFICS_HPP_ */
