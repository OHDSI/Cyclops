/*
 * LogisticModelSpecifics.hpp
 *
 * CPU engine for logistic regression ("lr") that keeps each row's fitted probability
 * p = 1 / (1 + exp(-xBeta)) next to the exact xBeta. The gradient and Hessian of a coefficient
 * are then sums of p and p (1 - p) over the column's entries, without exp() or divisions, and
 * a coordinate update moves p by its odds ratio: one exp() per indicator column instead of one
 * per entry.
 *
 * Subclass of the logistic ModelSpecifics: xBeta is updated as before, so everything that reads
 * xBeta (convergence checks, predictions) is unchanged. exp(xBeta) and the denominators are no
 * longer updated with each coefficient: every base method that reads them recomputes them first.
 */

#ifndef LOGISTICMODELSPECIFICS_HPP_
#define LOGISTICMODELSPECIFICS_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "ModelSpecifics.h"

namespace bsccs {

template <typename RealType>
class LogisticModelSpecifics :
        public ModelSpecifics<LogisticRegression<RealType>, RealType> {
public:
    typedef ModelSpecifics<LogisticRegression<RealType>, RealType> Base;
    typedef typename Base::RealVector RealVector;

    LogisticModelSpecifics(const ModelData<RealType>& input) : Base(input) { }

    virtual ~LogisticModelSpecifics() { }

    AbstractModelSpecifics* clone(ComputeDeviceArguments computeDevice) const {
        return new LogisticModelSpecifics<RealType>(this->modelData);
    }

    void setWeights(double* inWeights, double* cenWeights, bool useCrossValidation) {
        Base::setWeights(inWeights, cenWeights, useCrossValidation);
        // Cross-validation returns to the same folds, often on the same engine
        hasViews = hasViews && std::equal(hKWeight.begin(), hKWeight.end(), viewWeights.begin(),
                                          viewWeights.end());
    }

    void computeGradientAndHessian(int index, double* ogradient, double* ohessian, bool useWeights);

    void updateXBeta(double delta, int index, bool useWeights);

    void computeRemainingStatistics(bool useWeights);

    // Base methods that read exp(xBeta) and the denominators

    double getLogLikelihood(bool useCrossValidation) {
        updateExpXBeta();
        return Base::getLogLikelihood(useCrossValidation);
    }

    double getPredictiveLogLikelihood(double* weights) {
        updateExpXBeta();
        return Base::getPredictiveLogLikelihood(weights);
    }

    void computeThirdDerivative(int index, double* othird, bool useWeights) {
        updateExpXBeta();
        Base::computeThirdDerivative(index, othird, useWeights);
    }

    void computeFisherInformation(int indexOne, int indexTwo, double* oinfo, bool useWeights) {
        updateExpXBeta();
        Base::computeFisherInformation(indexOne, indexTwo, oinfo, useWeights);
    }

    void computeMMGradientAndHessian(std::vector<GradientHessian>& gh,
                                     const std::vector<bool>& fixBeta, bool useWeights) {
        updateExpXBeta();
        Base::computeMMGradientAndHessian(gh, fixBeta, useWeights);
    }

private:
    using Base::hX;
    using Base::hXBeta;
    using Base::offsExpXBeta;
    using Base::hKWeight;
    using Base::hXjY;
    using Base::K;
    using Base::J;

    // Entries of a column: rows (nullptr: every row, in order) and values (nullptr: all 1)
    struct Column {
        const int* rows;
        const RealType* values;
        size_t n;
    };

    Column describeColumn(int index) const;

    void updateExpXBeta() {
        if (!hasExpXBeta) {
            Base::computeRemainingStatistics(true); // does not change p
            hasExpXBeta = true;
        }
    }

    // p from e = exp(xBeta) as the base engine computes it, e / (1 + e), so p reaches 1 at the
    // same xBeta; 1 where exp() overflows (the base engine has NaN there)
    static RealType probabilityFromExp(RealType e) {
        return std::isinf(e) ? static_cast<RealType>(1) : e / (static_cast<RealType>(1) + e);
    }

    static RealType probability(RealType xBeta) {
        return probabilityFromExp(std::exp(xBeta));
    }

    template <bool hasRows, bool hasValues, bool isWeighted>
    void sumColumn(const Column& c, RealType* gradient, RealType* hessian) const;

    template <bool hasRows, bool hasValues>
    void dispatchSumColumn(const Column& c, bool isWeighted, RealType* gradient, RealType* hessian) const;

    template <bool hasRows, bool hasValues>
    void moveColumn(const Column& c, RealType delta);

    void setupViews();

    // p of each row: moved by updateXBeta(), and recomputed by computeRemainingStatistics(), which
    // callers run whenever xBeta changes otherwise (as they must for the base engine's statistics)
    RealVector fitted;
    bool hasExpXBeta = false; // exp(xBeta) and the denominators match xBeta

    // Each column restricted to rows with non-zero weight, for weighted (cross-validation) fits
    struct View {
        std::vector<int> rows;
        RealVector values; // empty for indicator and intercept columns
    };
    std::vector<View> views; // empty when no weight is zero
    RealVector viewWeights; // the weights the views were made for
    bool hasViews = false;
    bool unitWeights = false; // all non-zero weights are 1
};

template <typename RealType>
typename LogisticModelSpecifics<RealType>::Column
LogisticModelSpecifics<RealType>::describeColumn(int index) const {
    switch (hX.getFormatType(index)) {
    case INDICATOR :
        return { hX.getCompressedColumnVector(index), nullptr, static_cast<size_t>(hX.getNumberOfEntries(index)) };
    case SPARSE :
        return { hX.getCompressedColumnVector(index), hX.getDataVector(index),
                 static_cast<size_t>(hX.getNumberOfEntries(index)) };
    case DENSE :
        return { nullptr, hX.getDataVector(index), K };
    default : // INTERCEPT
        return { nullptr, nullptr, K };
    }
}

template <typename RealType>
void LogisticModelSpecifics<RealType>::setupViews() {
    unitWeights = true;
    std::vector<int> weightedRows;
    for (size_t k = 0; k < K; ++k) {
        if (hKWeight[k] != static_cast<RealType>(0)) {
            weightedRows.push_back(static_cast<int>(k));
            unitWeights = unitWeights && hKWeight[k] == static_cast<RealType>(1);
        }
    }

    viewWeights.assign(hKWeight.begin(), hKWeight.end());
    hasViews = true;
    views.clear();
    if (weightedRows.size() == K) { // the columns themselves
        return;
    }

    views.resize(J);
    for (size_t j = 0; j < J; ++j) {
        const Column c = describeColumn(static_cast<int>(j));
        View& view = views[j];
        if (!c.rows && !c.values) { // intercept
            view.rows = weightedRows;
            continue;
        }
        view.rows.reserve(c.n);
        if (c.values) {
            view.values.reserve(c.n);
        }
        for (size_t i = 0; i < c.n; ++i) {
            const int k = c.rows ? c.rows[i] : static_cast<int>(i);
            if (hKWeight[k] != static_cast<RealType>(0)) {
                view.rows.push_back(k);
                if (c.values) {
                    view.values.push_back(c.values[i]);
                }
            }
        }
    }
}

template <typename RealType> template <bool hasRows, bool hasValues, bool isWeighted>
void LogisticModelSpecifics<RealType>::sumColumn(const Column& c,
                                                 RealType* ogradient, RealType* ohessian) const {
    const RealType* p = fitted.data();
    const RealType* w = hKWeight.data();
    const RealType one = static_cast<RealType>(1);

    // One running sum in column order, as the base engine adds them
    RealType gradient = static_cast<RealType>(0);
    RealType hessian = static_cast<RealType>(0);
    for (size_t i = 0; i < c.n; ++i) {
        const int k = hasRows ? c.rows[i] : static_cast<int>(i);
        const RealType x = hasValues ? c.values[i] : one;
        const RealType wx = isWeighted ? w[k] * x : x;
        gradient += wx * p[k];
        hessian += wx * (hasValues ? x : one) * p[k] * (one - p[k]);
    }

    *ogradient = gradient;
    *ohessian = hessian;
}

template <typename RealType> template <bool hasRows, bool hasValues>
void LogisticModelSpecifics<RealType>::dispatchSumColumn(const Column& c, bool isWeighted,
                                                         RealType* gradient, RealType* hessian) const {
    if (isWeighted) {
        sumColumn<hasRows, hasValues, true>(c, gradient, hessian);
    } else {
        sumColumn<hasRows, hasValues, false>(c, gradient, hessian);
    }
}

template <typename RealType>
void LogisticModelSpecifics<RealType>::computeGradientAndHessian(int index, double* ogradient,
                                                                 double* ohessian, bool useWeights) {
    Column c = describeColumn(index);
    bool isWeighted = false;
    if (useWeights) {
        if (!hasViews) {
            setupViews();
        }
        if (!views.empty()) { // only rows with non-zero weight
            const View& view = views[index];
            c = { view.rows.data(), view.values.empty() ? nullptr : view.values.data(), view.rows.size() };
        }
        isWeighted = !unitWeights;
    }

    RealType gradient;
    RealType hessian;
    if (c.rows) {
        if (c.values) {
            dispatchSumColumn<true, true>(c, isWeighted, &gradient, &hessian);
        } else {
            dispatchSumColumn<true, false>(c, isWeighted, &gradient, &hessian);
        }
    } else {
        if (c.values) {
            dispatchSumColumn<false, true>(c, isWeighted, &gradient, &hessian);
        } else {
            dispatchSumColumn<false, false>(c, isWeighted, &gradient, &hessian);
        }
    }

    *ogradient = static_cast<double>(gradient - hXjY[index]);
    *ohessian = static_cast<double>(hessian);
}

template <typename RealType> template <bool hasRows, bool hasValues>
void LogisticModelSpecifics<RealType>::moveColumn(const Column& c, RealType delta) {
    RealType* p = fitted.data();
    const RealType one = static_cast<RealType>(1);
    const RealType zero = static_cast<RealType>(0);
    const RealType nearOne = one - std::sqrt(std::numeric_limits<RealType>::epsilon());
    const RealType factor = std::exp(delta); // odds ratio of a row with x = 1

    for (size_t i = 0; i < c.n; ++i) {
        const int k = hasRows ? c.rows[i] : static_cast<int>(i);
        if (hasValues) { // one exp() per entry either way, so p from the exact xBeta
            hXBeta[k] += delta * c.values[i];
            p[k] = probability(hXBeta[k]);
            continue;
        }
        hXBeta[k] += delta;

        // New odds = factor * old odds. Where that gives 0, NaN or nearly 1, p comes from the
        // exact xBeta instead: near 1 so that p rounds to 1 where the base engine's does
        // (separated data then stop the same way), and at 0 because no factor moves p = 0.
        const RealType moved = factor * p[k];
        const RealType next = moved / ((one - p[k]) + moved);
        p[k] = (next > zero && next < nearOne) ? next : probability(hXBeta[k]);
    }
}

template <typename RealType>
void LogisticModelSpecifics<RealType>::updateXBeta(double delta, int index, bool useWeights) {
    const RealType realDelta = static_cast<RealType>(delta);
    const Column c = describeColumn(index);
    if (c.rows) {
        if (c.values) {
            moveColumn<true, true>(c, realDelta);
        } else {
            moveColumn<true, false>(c, realDelta);
        }
    } else {
        if (c.values) {
            moveColumn<false, true>(c, realDelta);
        } else {
            moveColumn<false, false>(c, realDelta);
        }
    }
    hasExpXBeta = false;
}

template <typename RealType>
void LogisticModelSpecifics<RealType>::computeRemainingStatistics(bool useWeights) {
    Base::computeRemainingStatistics(useWeights);
    hasExpXBeta = true;

    fitted.resize(K);
    for (size_t k = 0; k < K; ++k) {
        fitted[k] = probabilityFromExp(offsExpXBeta[k]);
    }
}

} // namespace bsccs

#endif /* LOGISTICMODELSPECIFICS_HPP_ */
