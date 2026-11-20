/*
 * PooledLogisticModelSpecifics.hpp
 *
 * CPU engine for pooled logistic regression ("plr"): a binary regression (logit or cloglog link) over the
 * person-interval rows of each subject, without expanding the covariates to those rows. A row of X is a subject;
 * the rows of each subject (PooledRows.h), explicit or implicit, carry an interval into a small dense time basis T.
 * A column j has a time-basis column b(j) (time main effects, and time x covariate interactions, whose X column
 * holds the covariate's values) or none, and enters row r of subject i as x_ij T[t_r, b(j)] (or x_ij). With slot
 * s(j) = b(j) + 1 (0 without a time-basis column), tau_0 = 1 and tau_{b + 1}(t) = T[t, b], the linear predictor of
 * row r is
 *
 *     eta_r = offset_r + sum_s tau_s(t_r) state_is,     state_is = sum_{j: s(j) = s} x_ij beta_j.
 *
 * The state (subjects x S, S = B + 1) is stored with exp(eta_r) in two exact factors, E_i = exp(state_i0 +
 * offset_i) per subject and R_r = exp(eta_r - state_i0 - offset_i) per row (kept only with a time basis or row
 * offsets), each recomputed from the state when it changes. An update of a column without a time-basis column
 * costs one exp per entry (O(nnz_j)); a time column recomputes R_r on the rows where its basis is not 0. The
 * gradient and Hessian of column j cost O(rows of the subjects in column j), a product and a division per row.
 * Under the cloglog link the non-event rows of a subject contribute -E_i A_i, A_i = sum over those rows of
 * w_r R_r, cached per subject, so a column without a time-basis column costs O(nnz_j (1 + events)), as for
 * Poisson and Cox.
 *
 * Second-order kernels (logit link, double precision) use the per-subject slot sums
 * M_ss' = sum_r w_r d2_r tau_s tau_s', so a Hessian block costs O(rows x S^2 + sum_i nnz_i^2), not
 * O(rows x nnz^2) as on the expanded rows.
 *
 * Subclass of the logistic ModelSpecifics, which keeps the weights of the subjects (case weights and
 * cross-validation folds); every per-row computation is overridden here, and the optimizations of the logistic
 * engine that read its per-row state (lazy intercept, fold partitions) are off. Zero certificates
 * (CyclicCoordinateDescent::certifiedZero()) have drift bounds of their own here (recordDrift()).
 */

#ifndef POOLEDLOGISTICMODELSPECIFICS_HPP_
#define POOLEDLOGISTICMODELSPECIFICS_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
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
        slotOf.assign(J, 0);
        for (const auto& column : pooled.timeColumns) {
            slotOf[this->modelData.getColumnIndex(column.first)] = column.second + 1;
        }
        state.assign(K * S, 0.0);
        eventStart.assign(K + 1, 0);
        eventRow.clear();
        eventWeight.assign(K, 0.0);
        for (size_t i = 0; i < K; ++i) {
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                if (pooled.eventOf(i, r, hY[i])) {
                    eventRow.push_back(r);
                    eventWeight[i] += rowWeight(r);
                }
            }
            eventStart[i + 1] = eventRow.size();
        }
        subjectExp.assign(K, 1.0);
        rowExp.assign((B > 0 || !pooled.rowOffset.empty()) ? pooled.rows() : 0, 1.0);
        if (pooled.cloglog) nonEvent.assign(K, 0.0);
        this->driftTracking = false; // the drift structures are built when tracking is next switched on
        driftBuilt = false;
        maxAbsStateSlot.assign(S, 0.0);
        refreshAll();
        rowWiseF.clear();
        rowStartF.clear();
        spareF.clear();
        spareRowStartF.clear();
        hasDirection = false;
    }

    void computeGradientAndHessian(int index, double* ogradient, double* ohessian, bool useWeights) {
        const int s = slotOf[index];
        const double* rw = pooled.rowWeight.empty() ? nullptr : pooled.rowWeight.data();
        const double* re = rowExp.empty() ? nullptr : rowExp.data();
        double gradient = 0.0, hessian = 0.0;
        for (GenericIterator<RealType> it(hX, index); it; ++it) {
            const size_t i = it.index();
            const double sw = subjectWeight(i);
            if (sw == 0.0) continue;
            const double x = static_cast<double>(it.value());
            const double Ei = subjectExp[i];
            const size_t r0 = pooled.rowStart[i], r1 = pooled.rowStart[i + 1];
            if (pooled.cloglog && s == 0) { // the non-event rows factorize
                double g = Ei * nonEvent[i], h = g;
                for (size_t e = eventStart[i]; e < eventStart[i + 1]; ++e) {
                    const size_t r = eventRow[e];
                    double loss, d1, d2;
                    cloglogEvent(eta(i, r, pooled.intervalOf(i, r), &state[i * S]), &loss, &d1, &d2);
                    g += rowWeight(r) * d1;
                    h += rowWeight(r) * d2;
                }
                gradient += sw * x * g;
                hessian += sw * x * x * h;
            } else if (!pooled.cloglog && s == 0 && Ei > 0.0 && Ei < std::numeric_limits<double>::max()) {
                double g = -eventWeight[i], h = 0.0; // sum_r w_r (p_r - y_r), sum_r w_r p_r (1 - p_r)
                for (size_t r = r0; r < r1; ++r) {
                    const double e = re ? Ei * re[r] : Ei;
                    const double p = std::isinf(e) ? 1.0 : e / (1.0 + e); // R_r may overflow
                    const double w = rw ? rw[r] : 1.0;
                    g += w * p;
                    h += w * p * (1.0 - p);
                }
                gradient += sw * x * g;
                hessian += sw * x * x * h;
            } else {
                for (size_t r = r0; r < r1; ++r) {
                    const int t = pooled.intervalOf(i, r);
                    const double z = x * tau(t, s);
                    if (z == 0.0) continue;
                    double d1, d2;
                    rowDerivatives(i, r, t, &d1, &d2);
                    const double w = sw * (rw ? rw[r] : 1.0);
                    gradient += w * z * d1;
                    hessian += w * z * z * d2;
                }
            }
        }
        *ogradient = gradient;
        *ohessian = hessian;
        if (this->driftTracking) { // for recordDrift(); concurrent calls write distinct indices
            hessianVersion[index] = colIndicator[index] ? this->expVersion : -1;
            hessianValue[index] = hessian;
        }
    }

    // Threads: the gradients of several coordinates at once (each only reads the state) and the Hessian block
    void setThreads(int threads) {
        if (threads <= 1) {
            pool.reset();
        } else if (!pool || pool->size() != threads) {
            pool.reset(new WorkerPool(threads));
        }
    }

    void setSecondOrderThreads(int threads) { setThreads(threads); }

    bool supportsConcurrentGradients() const { return pool != nullptr; }

    bool computeGradientsAndHessians(const std::vector<int>& indices, std::vector<GradientHessian>& gh,
                                     bool useWeights) {
        if (!pool) return false;
        gh.resize(indices.size());
        pool->run(indices.size(), [&](size_t task, int) {
            computeGradientAndHessian(indices[task], &gh[task].first, &gh[task].second, useWeights);
        });
        return true;
    }

    void updateXBeta(double delta, int index, bool useWeights) {
        axpyXBeta(delta, index);
    }

    void axpyXBeta(const double beta, const int j) {
        if (beta == 0.0) return;
        if (this->driftTracking) {
            trackedAxpyXBeta(beta, j);
            return;
        }
        const int s = slotOf[j];
        double largest = 0.0; // for the running maximum of |state| (zero certificates)
        for (GenericIterator<RealType> it(hX, j); it; ++it) {
            const size_t i = it.index();
            state[i * S + s] += beta * static_cast<double>(it.value());
            largest = std::max(largest, std::abs(state[i * S + s]));
            if (s == 0) {
                subjectExp[i] = std::exp(state[i * S] + subjectOffset(i));
            } else {
                refreshRows(i, s);
            }
        }
        maxAbsStateSlot[s] = std::max(maxAbsStateSlot[s], largest);
    }

    void zeroXBeta() {
        std::fill(state.begin(), state.end(), 0.0);
        refreshAll();
    }

    void computeXBeta(double* beta, bool useWeights) {
        std::fill(state.begin(), state.end(), 0.0);
        for (size_t j = 0; j < J; ++j) {
            if (beta[j] == 0.0) continue;
            const int s = slotOf[j];
            for (GenericIterator<RealType> it(hX, static_cast<int>(j)); it; ++it) {
                state[it.index() * S + s] += beta[j] * static_cast<double>(it.value());
            }
        }
        refreshAll();
    }

    // The linear predictor of each person-interval row, subject by subject, with its offsets (lr's xBeta holds an offset
    // as a column with coefficient 1), for the convergence criterion that compares it between cycles (Zhang-Oles; the
    // caller weighs each row by its subject's weight)
    const std::vector<double> getXBeta() {
        std::vector<double> rows(pooled.rows(), 0.0);
        for (size_t i = 0; i < K; ++i) {
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                rows[r] = eta(i, r, pooled.intervalOf(i, r), &state[i * S]);
            }
        }
        return rows;
    }

    const std::vector<double> getXBetaSave() {
        if (savedSummary.size() != pooled.rows()) savedSummary.assign(pooled.rows(), 0.0);
        return savedSummary;
    }

    void saveXBeta() { savedSummary = getXBeta(); }

    void computeRemainingStatistics(bool useWeights) { } // nothing cached per row

    void computeFixedTermsInLogLikelihood(bool useCrossValidation) { }

    void computeFixedTermsInGradientAndHessian(bool useCrossValidation) { }

    void computeNumeratorForGradient(int index, bool useWeights) { }

    double getLogLikelihood(bool useCrossValidation) {
        return logLikelihood(nullptr, nullptr, 0.0);
    }

    double getPredictiveLogLikelihood(double* weights) {
        if (this->legacy) { // the legacy score of "lr" on the rows: y log p, the event rows only
            double sum = 0.0;
            forEachRow(weights, [&](size_t i, size_t r, int interval, double w, bool y) {
                if (!y) return;
                double loss, d1, d2;
                rowTerms(eta(i, r, interval, &state[i * S]), y, &loss, &d1, &d2);
                sum -= w * loss;
            });
            return sum;
        }
        return logLikelihood(weights, nullptr, 0.0);
    }

    double getGradientObjective(bool useCrossValidation) { // sum_r w_r y_r eta_r, as the logistic engine
        double criterion = 0.0;
        forEachRow(nullptr, [&](size_t i, size_t r, int t, double w, bool y) {
            if (y) criterion += w * eta(i, r, t, &state[i * S]);
        });
        return criterion;
    }

    // Observed information (the expected one under the logit link)
    void computeFisherInformation(int indexOne, int indexTwo, double* oinfo, bool useWeights) {
        const std::vector<double> two = denseColumn(indexTwo);
        const int s1 = slotOf[indexOne];
        const int s2 = slotOf[indexTwo];
        double info = 0.0;
        for (GenericIterator<RealType> it(hX, indexOne); it; ++it) {
            const size_t i = it.index();
            const double sw = subjectWeight(i);
            if (sw == 0.0 || two[i] == 0.0) continue;
            const double x = static_cast<double>(it.value()) * two[i];
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const int t = pooled.intervalOf(i, r);
                const double z = x * tau(t, s1) * tau(t, s2);
                if (z == 0.0) continue;
                double loss, d1, d2;
                rowTerms(eta(i, r, t, &state[i * S]), pooled.eventOf(i, r, hY[i]), &loss, &d1, &d2);
                info += sw * rowWeight(r) * z * d2;
            }
        }
        *oinfo = info;
    }

    // u_i = -sum over the rows of subject i of w_r z_r d loss_r / d eta: one score per subject
    bool scoreContributions(int index, std::vector<double>& u) {
        u.assign(K, 0.0);
        const int s = slotOf[index];
        for (GenericIterator<RealType> it(hX, index); it; ++it) {
            const size_t i = it.index();
            const double sw = subjectWeight(i);
            if (sw == 0.0) continue;
            const double x = static_cast<double>(it.value());
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const int t = pooled.intervalOf(i, r);
                const double z = x * tau(t, s);
                if (z == 0.0) continue;
                double loss, d1, d2;
                rowTerms(eta(i, r, t, &state[i * S]), pooled.eventOf(i, r, hY[i]), &loss, &d1, &d2);
                u[i] -= sw * rowWeight(r) * z * d1;
            }
        }
        return true;
    }

    // Second-order API (AbstractModelSpecifics.h): logit link in double precision. The cloglog loss is not
    // generalized self-concordant (the premise of the "gap" certificates), so those fits stay on coordinate descent
    bool supportsSecondOrder() const {
        return std::is_same<RealType, double>::value && !pooled.cloglog;
    }

    // X_F by subject: positions in F and values. The layouts of the last two sets are kept: a Newton iteration
    // takes the gradient over F, then the Hessian block and the direction over its working set
    bool secondOrderPrepare(const std::vector<int>& F) {
        if (!supportsSecondOrder()) return false;
        if (F == rowWiseF && rowStartF.size() == K + 1) return true;
        std::swap(rowWiseF, spareF);
        std::swap(rowStartF, spareRowStartF);
        std::swap(entryPosition, spareEntryPosition);
        std::swap(entryValue, spareEntryValue);
        if (F == rowWiseF && rowStartF.size() == K + 1) return true;
        rowWiseF = F;
        rowStartF.assign(K + 1, 0);
        for (int j : F) {
            for (GenericIterator<RealType> it(hX, j); it; ++it) ++rowStartF[it.index() + 1];
        }
        for (size_t i = 0; i < K; ++i) rowStartF[i + 1] += rowStartF[i];
        entryPosition.resize(rowStartF[K]);
        entryValue.resize(rowStartF[K]);
        std::vector<size_t> next(rowStartF.begin(), rowStartF.end() - 1);
        for (size_t a = 0; a < F.size(); ++a) {
            for (GenericIterator<RealType> it(hX, F[a]); it; ++it) {
                const size_t e = next[it.index()]++;
                entryPosition[e] = static_cast<int>(a);
                entryValue[e] = static_cast<double>(it.value());
            }
        }
        return true;
    }

    // g = the gradient of the negative log likelihood over F
    bool fullGradient(const std::vector<int>& F, std::vector<double>& g) {
        if (!secondOrderPrepare(F)) return false;
        g.assign(F.size(), 0.0);
        std::vector<double> m(S);
        for (size_t i = 0; i < K; ++i) {
            const double sw = subjectWeight(i);
            if (sw == 0.0 || rowStartF[i] == rowStartF[i + 1]) continue;
            subjectSums(i, &m, nullptr);
            for (size_t e = rowStartF[i]; e < rowStartF[i + 1]; ++e) {
                const int a = entryPosition[e];
                g[a] += sw * entryValue[e] * m[slotOf[F[a]]];
            }
        }
        return true;
    }

    bool hessianBlock(const std::vector<int>& F, std::vector<double>& H) {
        if (!secondOrderPrepare(F)) return false;
        const size_t n = F.size();
        H.assign(n * n, 0.0);
        if (n == 0) return true;
        // The accumulation order of the columns, as the logistic engine's (LogisticSecondOrder.hpp): by entries, most
        // first, so the pairs of the densest columns (most of the work) fall in a block of the permuted upper triangle
        // that stays in cache. Each subject's entries are laid out again in that order: ranks ascend within a subject
        std::vector<size_t> count(n, 0);
        for (int a : entryPosition) ++count[a];
        std::vector<int> order(n);
        for (size_t a = 0; a < n; ++a) order[a] = static_cast<int>(a);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return count[a] > count[b]; });
        const size_t entries = rowStartF[K];
        std::vector<int> rank(entries), entrySlot(entries);
        std::vector<double> value(entries);
        {
            std::vector<size_t> next(rowStartF.begin(), rowStartF.end() - 1);
            for (size_t k = 0; k < n; ++k) {
                const int j = F[order[k]];
                for (GenericIterator<RealType> it(hX, j); it; ++it) {
                    const size_t e = next[it.index()]++;
                    rank[e] = static_cast<int>(k);
                    entrySlot[e] = slotOf[j];
                    value[e] = static_cast<double>(it.value());
                }
            }
        }
        // The permuted upper triangle, column c from the pairs (a <= c) of each subject. A task adds the pairs whose
        // later rank is in its range of ranks, about equal in work, so every entry is a sum over subjects in
        // increasing order, the same bit for bit on any number of threads. On one thread the ranges are cache tiles,
        // as the logistic engine's: one per 128 columns, at most 16
        const size_t tasks = pool ?
            std::max<size_t>(1, std::min<size_t>(8 * static_cast<size_t>(pool->size()), n / 32)) :
            std::max<size_t>(1, std::min<size_t>(16, n / 128));
        std::vector<size_t> rangeStart(tasks + 1, n);
        rangeStart[0] = 0;
        {
            std::vector<double> work(n);
            double before = 0.0, total = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double c = static_cast<double>(count[order[k]]);
                work[k] = c * (before + 1.0);
                before += c / static_cast<double>(K);
                total += work[k];
            }
            double run = 0.0;
            size_t task = 0;
            for (size_t k = 0; k < n && task + 1 < tasks; ++k) {
                run += work[k];
                while (task + 1 < tasks && run >= total * static_cast<double>(task + 1) / static_cast<double>(tasks)) {
                    rangeStart[++task] = k + 1;
                }
            }
        }
        std::vector<double> permuted(n * n, 0.0);
        // subjects a chunk at a time: the slot sums M of the chunk's subjects (about 1 MB at most for S up to 45), then
        // the pairs
        const size_t chunk = std::max<size_t>(64, std::min<size_t>(4096, (static_cast<size_t>(1) << 17) / (S * S)));
        const size_t block = 256;
        std::vector<double> M(chunk * S * S);
        for (size_t i0 = 0; i0 < K; i0 += chunk) {
            const size_t i1 = std::min(K, i0 + chunk);
            runTasks((i1 - i0 + block - 1) / block, [&](size_t b, int) {
                std::vector<double> Mi(S * S);
                for (size_t i = i0 + b * block; i < std::min(i1, i0 + (b + 1) * block); ++i) {
                    if (subjectWeight(i) == 0.0 || rowStartF[i] == rowStartF[i + 1]) continue;
                    subjectSums(i, nullptr, &Mi);
                    std::copy(Mi.begin(), Mi.end(), M.begin() + (i - i0) * S * S);
                }
            });
            runTasks(tasks, [&](size_t task, int) {
                const int lo = static_cast<int>(rangeStart[task]), hi = static_cast<int>(rangeStart[task + 1]);
                if (lo >= hi) return;
                for (size_t i = i0; i < i1; ++i) {
                    const double sw = subjectWeight(i);
                    const size_t e0 = rowStartF[i], e1 = rowStartF[i + 1];
                    if (sw == 0.0 || e0 == e1) continue;
                    const double* Mi = &M[(i - i0) * S * S];
                    size_t f = std::lower_bound(rank.begin() + e0, rank.begin() + e1, lo) - rank.begin();
                    for (; f < e1 && rank[f] < hi; ++f) {
                        const double xc = sw * value[f];
                        const double* Mc = Mi + entrySlot[f] * S;
                        double* column = &permuted[static_cast<size_t>(rank[f]) * n];
                        for (size_t e = e0; e <= f; ++e) {
                            column[rank[e]] += xc * value[e] * Mc[entrySlot[e]];
                        }
                    }
                }
            });
        }
        // back to the order of F: entry (r, c), r <= c, of the permuted upper triangle is H(order[r], order[c])
        for (size_t c = 0; c < n; ++c) {
            const size_t b = static_cast<size_t>(order[c]);
            for (size_t r = 0; r <= c; ++r) {
                const size_t a = static_cast<size_t>(order[r]);
                H[a + n * b] = permuted[r + n * c];
                H[b + n * a] = permuted[r + n * c];
            }
        }
        return true;
    }

    bool hessianVector(const std::vector<int>& F, const std::vector<double>& v, std::vector<double>& Hv) {
        if (!secondOrderPrepare(F) || v.size() != F.size()) return false;
        Hv.assign(F.size(), 0.0);
        std::vector<double> c(S), m(S);
        for (size_t i = 0; i < K; ++i) {
            const double sw = subjectWeight(i);
            if (sw == 0.0 || rowStartF[i] == rowStartF[i + 1]) continue;
            std::fill(c.begin(), c.end(), 0.0);
            for (size_t e = rowStartF[i]; e < rowStartF[i + 1]; ++e) {
                const int a = entryPosition[e];
                c[slotOf[F[a]]] += entryValue[e] * v[a];
            }
            std::fill(m.begin(), m.end(), 0.0);
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const int t = pooled.intervalOf(i, r);
                double d1, d2;
                rowDerivatives(i, r, t, &d1, &d2);
                const double q = rowWeight(r) * d2 * linear(t, c.data());
                m[0] += q;
                for (int b = 0; b < B; ++b) m[b + 1] += q * basisValue(t, b);
            }
            for (size_t e = rowStartF[i]; e < rowStartF[i + 1]; ++e) {
                const int a = entryPosition[e];
                Hv[a] += sw * entryValue[e] * m[slotOf[F[a]]];
            }
        }
        return true;
    }

    bool hessianDiagonal(const std::vector<int>& F, std::vector<double>& h) {
        if (!secondOrderPrepare(F)) return false;
        h.assign(F.size(), 0.0);
        std::vector<double> M(S * S);
        for (size_t i = 0; i < K; ++i) {
            const double sw = subjectWeight(i);
            if (sw == 0.0 || rowStartF[i] == rowStartF[i + 1]) continue;
            subjectSums(i, nullptr, &M);
            for (size_t e = rowStartF[i]; e < rowStartF[i + 1]; ++e) {
                const int s = slotOf[F[entryPosition[e]]];
                h[entryPosition[e]] += sw * entryValue[e] * entryValue[e] * M[s * S + s];
            }
        }
        return true;
    }

    bool setDirection(const std::vector<int>& F, const std::vector<double>& d) {
        if (!secondOrderPrepare(F) || d.size() != F.size()) return false;
        direction.assign(K * S, 0.0);
        for (size_t i = 0; i < K; ++i) {
            for (size_t e = rowStartF[i]; e < rowStartF[i + 1]; ++e) {
                const int a = entryPosition[e];
                direction[i * S + slotOf[F[a]]] += entryValue[e] * d[a];
            }
        }
        hasDirection = true;
        return true;
    }

    double trialLogLikelihood(double t) {
        if (!supportsSecondOrder() || !hasDirection) return std::numeric_limits<double>::quiet_NaN();
        return logLikelihood(nullptr, &direction, t);
    }

    bool commitStep(double t) {
        if (!supportsSecondOrder() || !hasDirection) return false;
        for (size_t k = 0; k < state.size(); ++k) state[k] += t * direction[k];
        refreshAll();
        return true;
    }

    // max over the rows with weight of |z_r'd|
    double directionMaxAbs() const {
        if (!supportsSecondOrder() || !hasDirection) return std::numeric_limits<double>::quiet_NaN();
        double largest = 0.0;
        forEachRow(nullptr, [&](size_t i, size_t r, int t, double w, bool y) {
            largest = std::max(largest, std::abs(linear(t, &direction[i * S])));
        });
        return largest;
    }

    // The work of one dense Newton iteration (or of about 45 Hv under CG) over F in CCD cycles over F, with the
    // per-unit costs of the logistic engine (LogisticSecondOrder.hpp)
    double newtonCostRatio(const std::vector<int>& F) {
        return costRatio(F, false);
    }

    double newtonCgCostRatio(const std::vector<int>& F, bool preconditioned) {
        return costRatio(F, true);
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

    // Zero certificates: the drift totals of the logistic engine (ModelSpecifics.h), kept by recordDrift() over the
    // person-interval rows; the logit link in double precision with 0/1 subject and row weights only
    AbstractModelSpecifics::GradientDrift getGradientDrift() const {
        AbstractModelSpecifics::GradientDrift drift = Base::getGradientDrift();
        drift.valid = driftSupported() && this->binaryWeights && rowWeightsBinary && this->driftTracking;
        drift.endReadEps = 8.0; // e = E_i R_r, then p = e / (1 + e): 4 eps per reading
        return drift;
    }

    void setDriftTracking(bool on) {
        on = on && driftSupported();
        if (on && !this->driftTracking) {
            buildDrift();
            nonNormal = countNonNormal();
        }
        Base::setDriftTracking(on); // a new epoch when switched on
    }

    // A column of 1s (slot 0, or a time-basis column of 0s and 1s): the rows of its subjects (where the basis is 1)
    double getCertifiableRows(int index) {
        if (!driftSupported()) return 0.0;
        buildDrift();
        return colIndicator[index] ? colRows[index] : 0.0;
    }

    // Separation over the person-interval rows, as the logistic rule on the expanded rows: d l / d beta_j =
    // sum_r w_r z_r (y_r - p_r), z_r = x_ij tau_s(j)(t_r), is one-signed when every row with z_r != 0 and weight is
    // an event exactly where z_r > 0 (+Inf), or a non-event exactly there (-Inf)
    bool separationDirections(const std::vector<int>& columns, std::vector<int>& direction) {
        direction.assign(columns.size(), 0);
        for (size_t c = 0; c < columns.size(); ++c) {
            const int s = slotOf[columns[c]];
            bool up = true, down = true, any = false;
            for (GenericIterator<RealType> it(hX, columns[c]); it && (up || down); ++it) {
                const size_t i = it.index();
                if (it.value() == 0 || subjectWeight(i) <= 0.0) continue;
                for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                    const double z = static_cast<double>(it.value()) * tau(pooled.intervalOf(i, r), s);
                    if (z == 0.0 || rowWeight(r) <= 0.0) continue;
                    any = true;
                    const bool event = pooled.eventOf(i, r, hY[i]);
                    up = up && ((z > 0.0) == event);
                    down = down && ((z > 0.0) != event);
                }
            }
            direction[c] = !any ? 0 : (up ? 1 : (down ? -1 : 0));
        }
        return true;
    }

    // The distinct values z_r of column j over the person-interval rows (with 0 when some row has 0), so that a
    // divergent coefficient stops at the limit of the expanded rows
    bool likelihoodColumnValues(int j, std::vector<double>& values) {
        values.clear();
        const int s = slotOf[j];
        size_t entries = 0, keep = 4096;
        for (GenericIterator<RealType> it(hX, j); it; ++it) {
            const size_t i = it.index();
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const double z = static_cast<double>(it.value()) * tau(pooled.intervalOf(i, r), s);
                if (z == 0.0) continue;
                values.push_back(z);
                ++entries;
            }
            if (values.size() > keep) { // keep the distinct ones
                std::sort(values.begin(), values.end());
                values.erase(std::unique(values.begin(), values.end()), values.end());
                keep = 2 * values.size() + 4096;
            }
        }
        if (entries < pooled.rows()) values.push_back(0.0);
        return true;
    }

    // The duality-gap certificate of L1 logistic regression (LogisticSecondOrder.hpp) over the person-interval rows:
    // theta_r = s w_r (y_r - q_r), q_r = sigma(eta_r + c) + shift, and X_j'theta = sum_i x_ij sum_r theta_r tau_s(j)(t_r)
    bool dualObjective(const std::vector<int>& F, const std::vector<double>& lambda, const std::vector<double>& beta,
                       int intercept, double* dual) {
        if (!supportsSecondOrder() || lambda.size() != F.size() || beta.size() != F.size()) return false;
        double weightSum = 0.0, positive = 0.0;
        bool finite = true;
        forEachRow(nullptr, [&](size_t i, size_t r, int t, double w, bool y) {
            finite = finite && w > 0.0 && std::isfinite(eta(i, r, t, &state[i * S]));
            weightSum += w;
            positive += y ? w : 0.0;
        });
        if (!finite) return false;
        if (intercept >= 0) {
            if (!(positive > 0.0 && positive < weightSum) || slotOf[F[intercept]] != 0) return false;
            size_t entries = 0;
            for (GenericIterator<RealType> it(hX, F[intercept]); it; ++it) {
                if (static_cast<double>(it.value()) != 1.0) return false;
                ++entries;
            }
            if (entries != K) return false;
        }
        auto sigma = [](double t) {
            if (t >= 0.0) return 1.0 / (1.0 + std::exp(-t));
            const double e = std::exp(t);
            return e / (1.0 + e);
        };
        auto residual = [&](double c, double* curvature) { // sum_r w_r (y_r - sigma(eta_r + c)), compensated
            double sum = 0.0, compensation = 0.0, slope = 0.0;
            forEachRow(nullptr, [&](size_t i, size_t r, int t, double w, bool y) {
                const double q = sigma(eta(i, r, t, &state[i * S]) + c);
                const double term = w * ((y ? 1.0 : 0.0) - q);
                const double next = sum + term;
                compensation += (std::abs(sum) >= std::abs(term)) ? (sum - next) + term : (term - next) + sum;
                sum = next;
                slope += w * q * (1.0 - q);
            });
            if (curvature) *curvature = slope;
            return sum + compensation;
        };
        double c = 0.0;
        if (intercept >= 0) { // the residual decreases in c: bracket its root, then Newton kept inside the bracket
            double lo = 0.0, hi = 0.0;
            const double r0 = residual(0.0, nullptr);
            if (r0 > 0.0) {
                double step = 1.0;
                for (hi = 1.0; residual(hi, nullptr) > 0.0 && step < 1e3; step *= 2.0) { lo = hi; hi += step; }
            } else if (r0 < 0.0) {
                double step = 1.0;
                for (lo = -1.0; residual(lo, nullptr) < 0.0 && step < 1e3; step *= 2.0) { hi = lo; lo -= step; }
            }
            c = 0.5 * (lo + hi);
            for (int iteration = 0; iteration < 200 && r0 != 0.0; ++iteration) {
                double curvature = 0.0;
                const double r = residual(c, &curvature);
                if (r == 0.0) break;
                if (r > 0.0) lo = c; else hi = c;
                double next = (curvature > 0.0) ? c + r / curvature : 0.5 * (lo + hi);
                if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
                if (next == c || hi - lo <= 4.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(c))) {
                    c = next;
                    break;
                }
                c = next;
            }
        }
        const double shift = (intercept >= 0) ? residual(c, nullptr) / weightSum : 0.0;
        if (!std::isfinite(shift)) return false;
        // a_r = w_r (y_r - q_r), summed by subject and slot; theta'eta
        std::vector<double> m(K * S, 0.0);
        double thetaEta = 0.0;
        bool feasible = true;
        forEachRow(nullptr, [&](size_t i, size_t r, int t, double w, bool y) {
            const double e = eta(i, r, t, &state[i * S]);
            const double q = sigma(e + c) + shift;
            feasible = feasible && q >= 0.0 && q <= 1.0;
            const double a = w * ((y ? 1.0 : 0.0) - q);
            thetaEta += a * e;
            for (int s = 0; s < S; ++s) m[i * S + s] += a * tau(t, s);
        });
        if (!feasible) return false;
        double scale = 1.0, thetaO = thetaEta;
        for (size_t p = 0; p < F.size(); ++p) {
            const int s = slotOf[F[p]];
            double dot = 0.0;
            for (GenericIterator<RealType> it(hX, F[p]); it; ++it) {
                dot += static_cast<double>(it.value()) * m[it.index() * S + s];
            }
            if (!std::isfinite(dot)) return false;
            if (lambda[p] > 0.0 && std::abs(dot) > 0.0) scale = std::min(scale, lambda[p] / std::abs(dot));
            thetaO -= beta[p] * dot;
        }
        thetaO *= scale;
        double value = 0.0, compensation = 0.0;
        forEachRow(nullptr, [&](size_t i, size_t r, int t, double w, bool y) {
            const double yr = y ? 1.0 : 0.0;
            const double q = sigma(eta(i, r, t, &state[i * S]) + c) + shift;
            const double u = yr - scale * (yr - q);
            feasible = feasible && u >= 0.0 && u <= 1.0;
            const double term = -w * ((u > 0.0 ? u * std::log(u) : 0.0) + (u < 1.0 ? (1.0 - u) * std::log(1.0 - u) : 0.0));
            const double next = value + term;
            compensation += (std::abs(value) >= std::abs(term)) ? (value - next) + term : (term - next) + value;
            value = next;
        });
        if (!feasible) return false;
        *dual = value + compensation - thetaO;
        return std::isfinite(*dual);
    }

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

    double subjectOffset(size_t i) const {
        return pooled.subjectOffset.empty() ? 0.0 : pooled.subjectOffset[i];
    }

    double basisValue(int t, int b) const {
        return pooled.basis[static_cast<size_t>(t) * B + b];
    }

    double tau(int t, int s) const {
        return s == 0 ? 1.0 : basisValue(t, s - 1);
    }

    // sum_s tau_s(t) v_s
    double linear(int t, const double* v) const {
        double value = v[0];
        for (int b = 0; b < B; ++b) value += basisValue(t, b) * v[b + 1];
        return value;
    }

    // eta of row r (interval t) of subject i, whose state is s
    double eta(size_t i, size_t r, int t, const double* s) const {
        double value = linear(t, s) + subjectOffset(i);
        if (!pooled.rowOffset.empty()) value += pooled.rowOffset[r];
        return value;
    }

    // An event row under the cloglog link, mu = e^eta: the loss -log(1 - e^-mu) and its derivatives in eta,
    // -mu / (e^mu - 1) and mu e^-mu (e^-mu + mu - 1) / (1 - e^-mu)^2
    static void cloglogEvent(double eta, double* loss, double* d1, double* d2) {
        const double mu = std::exp(eta);
        const double em = std::exp(-mu);
        const double q = -std::expm1(-mu); // 1 - e^-mu
        *loss = (mu > 0.693) ? -std::log1p(-em) : -std::log(q);
        *d1 = -mu * em / q;
        const double c = (mu < 1e-3) ? mu * mu * (0.5 - mu / 6.0 + mu * mu / 24.0) : em + mu - 1.0; // series where it cancels
        *d2 = mu * em * c / (q * q);
    }

    // The loss (negative log likelihood) of a row and its first and second derivatives in eta
    void rowTerms(double eta, bool y, double* loss, double* d1, double* d2) const {
        if (pooled.cloglog) {
            if (y) {
                cloglogEvent(eta, loss, d1, d2);
            } else {
                const double mu = std::exp(eta);
                *loss = mu;
                *d1 = mu;
                *d2 = mu;
            }
        } else {
            const double p = 1.0 / (1.0 + std::exp(-eta));
            *loss = std::max(eta, 0.0) + std::log1p(std::exp(-std::abs(eta))) - (y ? eta : 0.0);
            *d1 = p - (y ? 1.0 : 0.0);
            *d2 = p * (1.0 - p);
        }
    }

    // Subject i: m_s = sum_r w_r d1_r tau_s(t_r) and M_ss' = sum_r w_r d2_r tau_s tau_s' (row weights only)
    void subjectSums(size_t i, std::vector<double>* m, std::vector<double>* M) const {
        if (m) std::fill(m->begin(), m->end(), 0.0);
        if (M) std::fill(M->begin(), M->end(), 0.0);
        for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
            const double w = rowWeight(r);
            if (w == 0.0) continue;
            const int t = pooled.intervalOf(i, r);
            double d1, d2;
            rowDerivatives(i, r, t, &d1, &d2);
            for (int s = 0; s < S; ++s) {
                const double ts = tau(t, s);
                if (ts == 0.0) continue; // adds nothing to row s (and column s) of M
                if (m) (*m)[s] += w * d1 * ts;
                if (M) {
                    for (int s2 = 0; s2 < S; ++s2) (*M)[s * S + s2] += w * d2 * ts * tau(t, s2);
                }
            }
        }
    }

    // The first and second derivatives in eta of the loss of row r (interval t) of subject i, from the cached
    // exp(eta_r) = E_i R_r under the logit link
    void rowDerivatives(size_t i, size_t r, int t, double* d1, double* d2) const {
        const double Ei = subjectExp[i];
        if (!pooled.cloglog && Ei > 0.0 && Ei < std::numeric_limits<double>::max()) {
            const double e = rowExp.empty() ? Ei : Ei * rowExp[r];
            const double p = std::isinf(e) ? 1.0 : e / (1.0 + e); // R_r may overflow
            *d1 = p - (pooled.eventOf(i, r, hY[i]) ? 1.0 : 0.0);
            *d2 = p * (1.0 - p);
            return;
        }
        double loss;
        rowTerms(eta(i, r, t, &state[i * S]), pooled.eventOf(i, r, hY[i]), &loss, d1, d2);
    }

    // R_r of the rows of subject i where tau_s is not 0 (all rows: s < 0), and A_i under cloglog; Count: keeps
    // nonNormal
    template <bool Count = false>
    void refreshRows(size_t i, int s) {
        const double* st = &state[i * S];
        if (!rowExp.empty()) {
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const int t = pooled.intervalOf(i, r);
                if (s > 0 && tau(t, s) == 0.0) continue;
                double value = linear(t, st) - st[0];
                if (!pooled.rowOffset.empty()) value += pooled.rowOffset[r];
                if (Count) {
                    const double old = rowExp[r];
                    rowExp[r] = std::exp(value);
                    nonNormal += notNormal(rowExp[r]) - notNormal(old);
                } else {
                    rowExp[r] = std::exp(value);
                }
            }
        }
        if (pooled.cloglog) { // A_i
            double sum = 0.0;
            size_t e = eventStart[i];
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                if (e < eventStart[i + 1] && eventRow[e] == r) {
                    ++e;
                    continue;
                }
                sum += rowWeight(r) * (rowExp.empty() ? 1.0 : rowExp[r]);
            }
            nonEvent[i] = sum;
        }
    }

    void refreshAll() {
        for (size_t i = 0; i < K; ++i) {
            subjectExp[i] = std::exp(state[i * S] + subjectOffset(i));
            refreshRows(i, -1);
        }
        scanState(); // every E_i and R_r from this state
        if (this->driftTracking) { // a new epoch
            nonNormal = countNonNormal();
            this->restartDrift();
        }
    }

    // ---- Zero certificates: drift bounds (see CyclicCoordinateDescent::certifiedZero()) ----
    //
    // The logistic engine keeps e = exp(eta) per row and multiplies it by exp(delta x) in an update. Here
    // e_r = E_i R_r, and an update recomputes E_i = exp(state_i0 + offset_i) (slot 0) or R_r = exp(eta_r - state_i0
    // - offset_i) (slot s > 0, the rows where tau_s != 0) from the state. The log of the ratio of the computed new
    // and old values is then delta x tau within kappa, the rounding of the two arguments and of the two exps
    // (recordDrift()). The arguments' rounding grows with |state|, so kappa uses the largest |state_is| seen since
    // the last refreshAll() (every stored E_i and R_r was computed from a state since then; a slot-0 update does
    // not recompute R_r, whose rounding is that of the state_i0 when it was last computed). The bound needs E_i and
    // R_r to be normal numbers (a relative error): while any is not, every update starts a new epoch. The running
    // maximum is kept whether tracking or not (a scan of the state in refreshAll(), a comparison per entry in an
    // update), so that it covers every state since the last refreshAll() when tracking is switched on.

    static long notNormal(double v) {
        return (v >= std::numeric_limits<double>::min() && v <= std::numeric_limits<double>::max()) ? 0 : 1;
    }

    bool driftSupported() const {
        return std::is_same<RealType, double>::value && !this->legacy && !pooled.cloglog;
    }

    long countNonNormal() const {
        long count = 0;
        for (double v : subjectExp) count += notNormal(v);
        for (double v : rowExp) count += notNormal(v);
        return count;
    }

    void scanState() {
        std::fill(maxAbsStateSlot.begin(), maxAbsStateSlot.end(), 0.0);
        for (size_t i = 0; i < K; ++i) {
            for (int s = 0; s < S; ++s) {
                maxAbsStateSlot[s] = std::max(maxAbsStateSlot[s], std::abs(state[i * S + s]));
            }
        }
    }

    // Once per model (the data do not change), on the first use: not paid by fits without certificates
    void buildDrift() {
        if (driftBuilt) return;
        rowWeightsBinary = true;
        for (double w : pooled.rowWeight) rowWeightsBinary = rowWeightsBinary && (w == 0.0 || w == 1.0);
        maxRowOffset = 0.0;
        for (double o : pooled.rowOffset) maxRowOffset = std::max(maxRowOffset, std::abs(o));
        maxSubjectOffset = 0.0;
        for (double o : pooled.subjectOffset) maxSubjectOffset = std::max(maxSubjectOffset, std::abs(o));
        maxTauSlot.assign(S, 0.0);
        maxTauSlot[0] = 1.0;
        std::vector<char> basis01(S, 1); // tau_s in {0, 1}
        const size_t T = (B > 0) ? pooled.basis.size() / B : 0;
        for (size_t t = 0; t < T; ++t) {
            for (int b = 0; b < B; ++b) {
                const double v = basisValue(static_cast<int>(t), b);
                maxTauSlot[b + 1] = std::max(maxTauSlot[b + 1], std::abs(v));
                if (v != 0.0 && v != 1.0) basis01[b + 1] = 0;
            }
        }
        // colRows: the rows whose e an update recomputes, which are also the rows its gradient reads: every row of
        // the subjects with x != 0 (slot 0; x = 0 leaves E_i as it is), or the rows where tau_s != 0 of every
        // subject in the column (slot s; R_r is recomputed from the current state_i0 even where x = 0)
        colRows.assign(J, 0.0);
        colMaxAbsX.assign(J, 0.0);
        colIndicator.assign(J, 0);
        for (size_t j = 0; j < J; ++j) {
            const int s = slotOf[j];
            bool ones = true;
            for (GenericIterator<RealType> it(hX, static_cast<int>(j)); it; ++it) {
                const size_t i = it.index();
                const double x = static_cast<double>(it.value());
                ones = ones && x == 1.0;
                colMaxAbsX[j] = std::max(colMaxAbsX[j], std::abs(x));
                if (s == 0 && x == 0.0) continue;
                for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                    if (s == 0 || tau(pooled.intervalOf(i, r), s) != 0.0) colRows[j] += 1.0;
                }
            }
            colIndicator[j] = (ones && basis01[s]) ? 1 : 0;
        }
        hessianVersion.assign(J, -1);
        hessianValue.assign(J, 0.0);
        driftBuilt = true;
    }

    // axpyXBeta() while tracking: also the running maximum of |state|, the count of E_i and R_r that are not
    // normal numbers, and the drift of the update
    void trackedAxpyXBeta(const double beta, const int j) {
        const int s = slotOf[j];
        const long before = nonNormal;
        double largest = 0.0;
        for (GenericIterator<RealType> it(hX, j); it; ++it) {
            const size_t i = it.index();
            double& v = state[i * S + s];
            v += beta * static_cast<double>(it.value());
            largest = std::max(largest, std::abs(v));
            if (s == 0) {
                const double old = subjectExp[i];
                subjectExp[i] = std::exp(v + subjectOffset(i));
                nonNormal += notNormal(subjectExp[i]) - notNormal(old);
            } else {
                refreshRows<true>(i, s);
            }
        }
        maxAbsStateSlot[s] = std::max(maxAbsStateSlot[s], largest);
        if (before > 0 || nonNormal > 0) {
            this->restartDrift();
        } else {
            recordDrift(beta, j, s);
        }
        ++this->expVersion;
    }

    // The drift of p = e / (1 + e) in an update of column j (slot s) by delta, as the logistic engine bounds it
    // (ModelSpecifics.hpp, updateXBetaImpl()), with the log of every row's factor within |delta x tau| + kappa:
    // - kappa: the arguments of exp are sums of at most B + 3 terms, each at most max|tau_q| max|state_q| or an
    //   offset, so each rounds within (B + 3) eps / 2 of their absolute sum, and an exp within eps (doubled
    //   throughout for margin);
    // - a column of 1s (colIndicator) moves every row the way of delta, by at most tanh((|delta| + kappa) / 4) and
    //   by at most p (1 - p) G, G = exp(|delta| + kappa) - 1, and against it by at most kappa / 4; right after the
    //   column's gradient (hessianVersion), the rows' sum of p (1 - p) is that gradient's Hessian (0/1 weights);
    //   0.25 bounds the largest p (1 - p) (no work per row in the gradient)
    // - another column moves every row by at most tanh((|delta| max|x| max|tau_s| + kappa) / 4), either way.
    void recordDrift(double delta, int j, int s) {
        const double eps = std::numeric_limits<double>::epsilon();
        double kappa;
        if (s == 0) {
            const double m = maxAbsStateSlot[0];
            kappa = eps * (2.0 * m + 2.0 * (m + maxSubjectOffset)) + 4.0 * eps;
        } else {
            double sum = maxRowOffset;
            for (int q = 0; q < S; ++q) sum += maxTauSlot[q] * maxAbsStateSlot[q];
            kappa = 2.0 * (B + 4) * eps * sum + 4.0 * eps;
        }
        kappa *= 1.0 + 1E-12;
        const double terms = colRows[j];
        const double ad = std::abs(delta) * (1.0 + 4.0 * eps);
        double up = 0.0, down = 0.0, maxUp = 0.0, maxDown = 0.0;
        if (colIndicator[j]) {
            const bool rises = delta > 0.0;
            const double growth = std::expm1(ad + kappa) * (1.0 + 4.0 * eps) + 4.0 * eps;
            const double largestMove = std::min(1.0, std::tanh(0.25 * (ad + kappa)) * (1.0 + 1E-12) + 10.0 * eps);
            const double wrong = std::min(1.0, 0.25 * kappa * (1.0 + 1E-12) + 10.0 * eps); // against the step
            double sum = terms * largestMove, largest = largestMove;
            const double h = hessianValue[j];
            const bool fromHessian = hessianVersion[j] == this->expVersion && h >= 0.0 &&
                h <= std::numeric_limits<double>::max();
            if (fromHessian) {
                sum = std::min(growth * (h * (1.0 + (terms + 8.0) * eps) + 4.0 * eps * terms) * (1.0 + 1E-12) +
                               10.0 * eps * terms, sum);
                largest = std::min(largestMove, 0.25 * growth * (1.0 + 1E-12) + 10.0 * eps);
            }
            const double wrongSum = terms * wrong;
            if (rises) {
                up = sum; maxUp = largest; down = wrongSum; maxDown = wrong;
            } else {
                down = sum; maxDown = largest; up = wrongSum; maxUp = wrong;
            }
            if (fromHessian) { // the step's way by the column's Hessian bucket; against it for every column
                this->recordSidedDrift(AbstractModelSpecifics::GradientDrift::bucketOf(h), rises ? sum : 0.0,
                                       rises ? 0.0 : sum, rises ? growth : 0.0, rises ? 0.0 : growth);
                this->recordSidedDrift(0, rises ? 0.0 : wrongSum, rises ? wrongSum : 0.0, 0.0, 0.0);
            } else {
                this->recordSidedDrift(0, up, down, 0.0, 0.0);
            }
        } else {
            const double scale = colMaxAbsX[j] * maxTauSlot[s];
            const double largestMove = std::min(1.0,
                std::tanh(0.25 * (ad * scale * (1.0 + 4.0 * eps) + kappa)) * (1.0 + 1E-12) + 10.0 * eps);
            up = down = terms * largestMove;
            maxUp = maxDown = largestMove;
            this->recordSidedDrift(0, up, down, 0.0, 0.0);
        }
        this->accumulate(this->driftUp, up);
        this->accumulate(this->driftDown, down);
        this->accumulate(this->driftMaxUp, maxUp);
        this->accumulate(this->driftMaxDown, maxDown);
        ++this->driftUpdates;
        if (!std::isfinite(this->driftUp + this->driftDown + this->driftMaxUp + this->driftMaxDown)) {
            this->restartDrift();
        }
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

    // -sum_r w_r loss_r at the state plus t dir (a compensated sum)
    double logLikelihood(const double* weights, const std::vector<double>* dir, double t) const {
        double sum = 0.0, compensation = 0.0;
        forEachRow(weights, [&](size_t i, size_t r, int interval, double w, bool y) {
            double e = eta(i, r, interval, &state[i * S]);
            if (dir) e += t * linear(interval, &(*dir)[i * S]);
            double loss, d1, d2;
            rowTerms(e, y, &loss, &d1, &d2);
            const double term = -w * loss;
            const double next = sum + term;
            compensation += (std::abs(sum) >= std::abs(term)) ? (sum - next) + term : (term - next) + sum;
            sum = next;
        });
        return sum + compensation;
    }

    void setLrCostModel(bool on) { lrCostModel = on; }

    double costRatio(const std::vector<int>& F, bool conjugateGradient) const {
        if (!supportsSecondOrder()) return std::numeric_limits<double>::quiet_NaN();
        return lrCostModel ? lrCostRatio(F, conjugateGradient) : ownCostRatio(F, conjugateGradient);
    }

    // The work of plr itself (the default): a CCD visit of an entry covers its subject's rows in every slot
    double ownCostRatio(const std::vector<int>& F, bool conjugateGradient) const {
        const double aCycle = 4.0, aRows = 0.75, aCholesky = 0.035, aLinear = 1.0, aRow = 2.0, aAll = 40.0;
        std::vector<int> count(K, 0); // entries of F by subject
        double cycle = 0.0, entries = 0.0;
        for (int j : F) {
            for (GenericIterator<RealType> it(hX, j); it; ++it) {
                const size_t i = it.index();
                ++count[i];
                entries += 1.0;
                cycle += static_cast<double>(pooled.rowStart[i + 1] - pooled.rowStart[i]) * S;
            }
        }
        double pairs = 0.0;
        for (size_t i = 0; i < K; ++i) pairs += 0.5 * static_cast<double>(count[i]) * count[i];
        const double rows = static_cast<double>(pooled.rows());
        const double f = static_cast<double>(F.size());
        const double fixed = 2.0 * aLinear * entries + aAll * rows * S;
        const double newton = conjugateGradient ?
            45.0 * (2.0 * aLinear * entries + 2.0 * aRow * rows * S) + fixed :
            aRows * (pairs + rows * S * S) + aCholesky * f * f * f / 3.0 + fixed;
        return (cycle > 0.0) ? newton / (aCycle * cycle) : std::numeric_limits<double>::quiet_NaN();
    }

    // lr's cost model (lrNewtonCostRatio) on the expanded person-interval rows, counted without expanding them
    // (createControl(plrCostModel = "lr")). A row holds a subject's entries of time-independent columns (all its rows, as
    // lr stores them) and those of time-basis and interaction columns whose basis value is not 0 in its interval
    double lrCostRatio(const std::vector<int>& F, bool conjugateGradient) const {
        const double aCycle = 4.0, aRows = 0.75, aCholesky = 0.035, aLinear = 1.0, aRow = 2.0, aAll = 40.0;
        std::vector<double> count(K * S, 0.0); // entries of F by subject and slot (slot > 0: nonzero values)
        for (int j : F) {
            const int s = slotOf[j];
            for (GenericIterator<RealType> it(hX, j); it; ++it) {
                if (s == 0 || it.value() != static_cast<RealType>(0)) count[it.index() * S + s] += 1.0;
            }
        }
        double entries = 0.0, rowsSquared = 0.0; // exact: integers below 2^53
        for (size_t i = 0; i < K; ++i) {
            for (size_t r = pooled.rowStart[i]; r < pooled.rowStart[i + 1]; ++r) {
                const int t = pooled.intervalOf(i, r);
                double c = count[i * S];
                for (int s = 1; s < S; ++s) {
                    if (tau(t, s) != 0.0) c += count[i * S + s];
                }
                entries += c;
                rowsSquared += 0.5 * c * c;
            }
        }
        const double rows = static_cast<double>(pooled.rows());
        const double f = static_cast<double>(F.size());
        const double fixed = 2.0 * aLinear * entries + aAll * rows;
        const double newton = conjugateGradient ?
            45.0 * (2.0 * aLinear * entries + 2.0 * aRow * rows) + fixed :
            aRows * rowsSquared + aCholesky * f * f * f / 3.0 + fixed;
        return (entries > 0.0) ? newton / (aCycle * entries) : std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> denseColumn(int index) const {
        std::vector<double> x(K, 0.0);
        for (GenericIterator<RealType> it(hX, index); it; ++it) {
            x[it.index()] = static_cast<double>(it.value());
        }
        return x;
    }

    const PooledRows& pooled;
    int B = 0;                      // time-basis columns
    int S = 1;                      // state per subject: B + 1
    std::vector<int> slotOf;        // slot of each column: its time-basis column + 1, 0 for none
    std::vector<double> state;      // K x S, by subject
    std::vector<double> savedSummary;
    bool lrCostModel = false; // createControl(plrCostModel = "lr")

    std::vector<size_t> eventStart; // K + 1: the event rows of subject i are eventRow[eventStart[i] ...]
    std::vector<size_t> eventRow;
    std::vector<double> nonEvent;   // A_i
    std::vector<double> eventWeight; // sum of w_r over the event rows of each subject
    std::vector<double> subjectExp; // E_i = exp(state_i0 + offset_i)
    std::vector<double> rowExp;     // R_r = exp(eta_r - state_i0 - offset_i); empty when 1 (no basis, no row offsets)

    // second order: X_F by subject, and the state of the direction
    std::vector<int> rowWiseF;
    std::vector<size_t> rowStartF;
    std::vector<int> entryPosition;
    std::vector<double> entryValue;
    std::vector<int> spareF; // the layout before (secondOrderPrepare())
    std::vector<size_t> spareRowStartF;
    std::vector<int> spareEntryPosition;
    std::vector<double> spareEntryValue;
    std::vector<double> direction;
    bool hasDirection = false;

    std::unique_ptr<WorkerPool> pool; // only with more than one thread

    // zero certificates (buildDrift(), trackedAxpyXBeta())
    bool driftBuilt = false;
    bool rowWeightsBinary = true;
    std::vector<double> colRows;      // the rows whose e an update of each column recomputes
    std::vector<double> colMaxAbsX;   // max |x| per column
    std::vector<char> colIndicator;   // a column of 1s whose slot's basis is 0 or 1 (certifiable)
    std::vector<double> maxTauSlot;   // max |tau_s|
    double maxRowOffset = 0.0;
    double maxSubjectOffset = 0.0;
    std::vector<double> maxAbsStateSlot; // max |state_is| since the last refreshAll()
    long nonNormal = 0;               // E_i and R_r that are not normal numbers (while tracking)
    std::vector<long> hessianVersion; // per column: expVersion of its last gradient (-1: not certifiable)
    std::vector<double> hessianValue; // and its Hessian

    template <typename Task>
    void runTasks(size_t count, Task task) {
        if (pool) {
            pool->run(count, task);
        } else {
            for (size_t t = 0; t < count; ++t) task(t, 0);
        }
    }
};

} // namespace bsccs

#endif /* POOLEDLOGISTICMODELSPECIFICS_HPP_ */
