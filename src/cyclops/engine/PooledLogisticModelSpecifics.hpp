/*
 * PooledLogisticModelSpecifics.hpp
 *
 * CPU engine for pooled logistic regression ("plr"): logistic regression over the person-interval
 * rows of each subject, without expanding the covariates to those rows. A row of X is a subject;
 * the rows of each subject (PooledRows.h), explicit or implicit, carry an interval into a small
 * dense time basis T. A column j has a time-basis column b(j) (time main effects, and time x
 * covariate interactions, whose X column holds the covariate's values) or none, and enters row r
 * of subject i as x_ij T[t_r, b(j)] (or x_ij). So the linear predictor of row r is
 *
 *     eta_r = offset_r + s_i0 + sum_b T[t_r, b] s_i(b + 1),
 *
 * where s_i0 sums x_ij beta_j over the columns without a time-basis column and s_i(b + 1) over
 * those with time-basis column b. Only s (subjects x (B + 1)) is stored; an update of column j
 * costs O(nnz_j), and its gradient and Hessian O(rows of the subjects in column j x B).
 *
 * Subclass of the logistic ModelSpecifics, which keeps the weights of the subjects (case weights and
 * cross-validation folds); every per-row computation is overridden here, and the optimizations of
 * the logistic engine that read its per-row state (lazy intercept, fold partitions, gradient-drift
 * certificates, second-order steps) are switched off.
 */

#ifndef POOLEDLOGISTICMODELSPECIFICS_HPP_
#define POOLEDLOGISTICMODELSPECIFICS_HPP_

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "ModelSpecifics.h"

namespace bsccs {

template <typename RealType>
class PooledLogisticModelSpecifics :
        public ModelSpecifics<LogisticRegression<RealType>, RealType> {
public:
    typedef ModelSpecifics<LogisticRegression<RealType>, RealType> Base;

    PooledLogisticModelSpecifics(const ModelData<RealType>& input)
        : Base(input), pooled(input.getPooledRows()) { }

    virtual ~PooledLogisticModelSpecifics() { }

    AbstractModelSpecifics* clone(ComputeDeviceArguments computeDevice) const {
        auto copy = new PooledLogisticModelSpecifics<RealType>(this->modelData);
        copy->setLegacy(this->legacy);
        return copy;
    }

    void initialize(int iN, int iK, int iJ, const void* iXi, double* iNumerPid, double* iNumerPid2,
                    double* iDenomPid, double* iXjY, std::vector<std::vector<int>* >* iSparseIndices,
                    const int* iPid, double* iOffsExpXBeta, double* iXBeta, double* iOffs, double* iBeta,
                    const double* iY) {
        Base::initialize(iN, iK, iJ, iXi, iNumerPid, iNumerPid2, iDenomPid, iXjY, iSparseIndices, iPid,
                         iOffsExpXBeta, iXBeta, iOffs, iBeta, iY);
        if (!pooled.loaded || pooled.rowStart.size() != K + 1) {
            throw std::logic_error("Pooled logistic regression needs the rows of each subject");
        }
        B = pooled.basisColumns;
        S = B + 1;
        basisOf.assign(J, -1);
        for (const auto& column : pooled.timeColumns) {
            basisOf[this->modelData.getColumnIndex(column.first)] = column.second;
        }
        state.assign(K * S, 0.0);
    }

    void computeGradientAndHessian(int index, double* ogradient, double* ohessian, bool useWeights) {
        const int b = basisOf[index];
        double gradient = 0.0, hessian = 0.0;
        for (GenericIterator<RealType> it(hX, index); it; ++it) {
            const size_t i = it.index();
            const double sw = subjectWeight(i);
            if (sw == 0.0) continue;
            const double x = static_cast<double>(it.value());
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const int t = pooled.intervalOf(i, r);
                const double z = (b < 0) ? x : x * basisValue(t, b);
                if (z == 0.0) continue;
                const double w = sw * rowWeight(r);
                const double p = probability(eta(i, r, t));
                gradient += w * z * (p - (pooled.eventOf(i, r, hY[i]) ? 1.0 : 0.0));
                hessian += w * z * z * p * (1.0 - p);
            }
        }
        *ogradient = gradient;
        *ohessian = hessian;
    }

    void updateXBeta(double delta, int index, bool useWeights) {
        axpyXBeta(delta, index);
    }

    void axpyXBeta(const double beta, const int j) {
        if (beta == 0.0) return;
        const int slot = basisOf[j] + 1;
        for (GenericIterator<RealType> it(hX, j); it; ++it) {
            state[it.index() * S + slot] += beta * static_cast<double>(it.value());
        }
    }

    void zeroXBeta() {
        std::fill(state.begin(), state.end(), 0.0);
    }

    void computeXBeta(double* beta, bool useWeights) {
        zeroXBeta();
        for (size_t j = 0; j < J; ++j) {
            axpyXBeta(beta[j], static_cast<int>(j));
        }
    }

    // One summary per subject (the sum of its state), for the convergence criteria that compare
    // xBeta between cycles
    const std::vector<double> getXBeta() {
        std::vector<double> sum(K, 0.0);
        for (size_t i = 0; i < K; ++i) {
            for (int s = 0; s < S; ++s) sum[i] += state[i * S + s];
        }
        return sum;
    }

    void computeRemainingStatistics(bool useWeights) { } // nothing cached per row

    void computeFixedTermsInLogLikelihood(bool useCrossValidation) { }

    void computeFixedTermsInGradientAndHessian(bool useCrossValidation) { }

    void computeNumeratorForGradient(int index, bool useWeights) { }

    double getLogLikelihood(bool useCrossValidation) {
        return logLikelihood(nullptr);
    }

    double getPredictiveLogLikelihood(double* weights) {
        return logLikelihood(weights);
    }

    double getGradientObjective(bool useCrossValidation) { // sum_r w_r y_r eta_r, as the logistic engine
        double criterion = 0.0;
        forEachRow(nullptr, [&](size_t i, size_t r, int t, double w, bool y) {
            if (y) criterion += w * eta(i, r, t);
        });
        return criterion;
    }

    void computeFisherInformation(int indexOne, int indexTwo, double* oinfo, bool useWeights) {
        const std::vector<double> two = denseColumn(indexTwo);
        const int b1 = basisOf[indexOne];
        const int b2 = basisOf[indexTwo];
        double info = 0.0;
        for (GenericIterator<RealType> it(hX, indexOne); it; ++it) {
            const size_t i = it.index();
            const double sw = subjectWeight(i);
            if (sw == 0.0 || two[i] == 0.0) continue;
            const double x = static_cast<double>(it.value()) * two[i];
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const int t = pooled.intervalOf(i, r);
                const double z = x * (b1 < 0 ? 1.0 : basisValue(t, b1)) * (b2 < 0 ? 1.0 : basisValue(t, b2));
                if (z == 0.0) continue;
                const double p = probability(eta(i, r, t));
                info += sw * rowWeight(r) * z * p * (1.0 - p);
            }
        }
        *oinfo = info;
    }

    // u_i = sum over the rows of subject i of w_r z_r (y_r - p_r): one score per subject
    bool scoreContributions(int index, std::vector<double>& u) {
        u.assign(K, 0.0);
        const int b = basisOf[index];
        for (GenericIterator<RealType> it(hX, index); it; ++it) {
            const size_t i = it.index();
            const double sw = subjectWeight(i);
            if (sw == 0.0) continue;
            const double x = static_cast<double>(it.value());
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const int t = pooled.intervalOf(i, r);
                const double z = (b < 0) ? x : x * basisValue(t, b);
                if (z == 0.0) continue;
                const double y = pooled.eventOf(i, r, hY[i]) ? 1.0 : 0.0;
                u[i] += sw * rowWeight(r) * z * (y - probability(eta(i, r, t)));
            }
        }
        return true;
    }

    void getPredictiveEstimates(double* y, double* weights) {
        throw std::logic_error("Predictions are not yet implemented for pooled logistic regression");
    }

    void computeThirdDerivative(int index, double* othird, bool useWeights) {
        throw std::logic_error("Third derivatives are not implemented for pooled logistic regression");
    }

    void computeMMGradientAndHessian(std::vector<GradientHessian>& gh, const std::vector<bool>& fixBeta,
                                     bool useWeights) {
        throw std::logic_error("MM updates are not implemented for pooled logistic regression");
    }

    void computeSchoenfeldResiduals(int indexOne, std::vector<double>* residuals, std::vector<double>* times,
                                    std::vector<int>* strata, double* covariate, double* score, bool useWeights) {
        throw std::logic_error("Schoenfeld residuals are not defined for pooled logistic regression");
    }

    // Optimizations of the logistic engine that read its per-row state: off
    bool setLazyIntercept(bool on) { return false; }
    void flushIntercept() { }
    void setHeldOutRowsUnused(bool unused) { }
    FoldPartitionPtr makeFoldPartition(const std::vector<int>& segmentOfRow, int folds) const { return nullptr; }
    void setFoldPartition(FoldPartitionPtr partition, int fold) { }
    AbstractModelSpecifics::GradientDrift getGradientDrift() const { return AbstractModelSpecifics::GradientDrift(); }
    void setDriftTracking(bool on) { }
    bool supportsSecondOrder() const { return false; }
    bool separationDirections(const std::vector<int>& columns, std::vector<int>& direction) { return false; }
    bool dualObjective(const std::vector<int>& F, const std::vector<double>& lambda, const std::vector<double>& beta,
                       int intercept, double* dual) { return false; }

private:
    using Base::hX;
    using Base::hY;
    using Base::hKWeight;
    using Base::K;
    using Base::J;

    double subjectWeight(size_t i) const {
        return hKWeight.size() == K ? static_cast<double>(hKWeight[i]) : 1.0;
    }

    double rowWeight(size_t r) const {
        return pooled.rowWeight.empty() ? 1.0 : pooled.rowWeight[r];
    }

    double basisValue(int t, int b) const {
        return pooled.basis[static_cast<size_t>(t) * B + b];
    }

    double eta(size_t i, size_t r, int t) const {
        const double* s = &state[i * S];
        double value = s[0];
        for (int b = 0; b < B; ++b) {
            value += basisValue(t, b) * s[b + 1];
        }
        if (!pooled.rowOffset.empty()) value += pooled.rowOffset[r];
        if (!pooled.subjectOffset.empty()) value += pooled.subjectOffset[i];
        return value;
    }

    static double probability(double eta) {
        return 1.0 / (1.0 + std::exp(-eta));
    }

    // f(i, r, t, w, y) over the rows with positive weight; weights (one per subject) replace the case weights
    template <typename F>
    void forEachRow(const double* weights, F f) const {
        for (size_t i = 0; i < K; ++i) {
            const double sw = weights ? weights[i] : subjectWeight(i);
            if (sw == 0.0) continue;
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const double w = sw * rowWeight(r);
                if (w != 0.0) f(i, r, pooled.intervalOf(i, r), w, pooled.eventOf(i, r, hY[i]));
            }
        }
    }

    // sum_r w_r (y_r eta_r - log(1 + exp(eta_r)))
    double logLikelihood(const double* weights) const {
        double sum = 0.0;
        forEachRow(weights, [&](size_t i, size_t r, int t, double w, bool y) {
            const double e = eta(i, r, t);
            sum += w * ((y ? e : 0.0) - (std::max(e, 0.0) + std::log1p(std::exp(-std::abs(e)))));
        });
        return sum;
    }

    std::vector<double> denseColumn(int index) const {
        std::vector<double> x(K, 0.0);
        for (GenericIterator<RealType> it(hX, index); it; ++it) {
            x[it.index()] = static_cast<double>(it.value());
        }
        return x;
    }

    const PooledRows& pooled;
    int B = 0;                  // time-basis columns
    int S = 1;                  // state per subject: B + 1
    std::vector<int> basisOf;   // time-basis column of each column; -1: none
    std::vector<double> state;  // K x S, by subject
};

} // namespace bsccs

#endif /* POOLEDLOGISTICMODELSPECIFICS_HPP_ */
