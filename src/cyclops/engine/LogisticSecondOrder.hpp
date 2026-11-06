/*
 * LogisticSecondOrder.hpp
 *
 * The second-order API of AbstractModelSpecifics (Newton plan 2.2, Phase 7) for logistic regression in double
 * precision: l = sum_k w_k (y_k eta_k - log(1 + exp(eta_k))), so g = X_F'(w (p - y)) and H = X_F' diag(w p (1 - p)) X_F,
 * with p read from e = offsExpXBeta as the coordinate-descent gradient reads it. Every call flushes a pending lazy
 * intercept first. Rows of weight 0 are skipped: under a fold fit with held-out rows unused their eta is stale. A
 * commit moves only the rows the direction moves (only the training rows when held-out rows are unused), then
 * starts a new drift epoch. The row layout over F is the one of GroupedSecondOrder.hpp (one row per "stratum").
 * Included at the end of ModelSpecifics.hpp, after GroupedSecondOrder.hpp.
 */

#ifndef LOGISTICSECONDORDER_HPP_
#define LOGISTICSECONDORDER_HPP_

namespace bsccs {

namespace logistic {
inline double log1pExp(double x) { // log(1 + exp(x)) without overflow
	return (x > 0.0) ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}
} // namespace logistic

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrReady() const {
	return logisticSecondOrder && offsExpXBeta.size() >= K && hKWeight.size() == K && hXjY.size() >= J;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrFullGradient(const std::vector<int>& F, std::vector<double>& g) {
	flushIntercept();
	if (!lrReady() || !secondOrderPrepare(F)) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	g.assign(F.size(), 0.0);
	for (size_t i = 0; i < L.rows.size(); ++i) {
		const int k = L.rows[i];
		const double w = static_cast<double>(hKWeight[k]);
		if (w == 0.0) {
			continue;
		}
		const double r = w * readP(offsExpXBeta[k]);
		for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
			const int p = gsMap[L.position[e]];
			if (p >= 0) {
				g[p] += L.value[e] * r;
			}
		}
	}
	for (size_t p = 0; p < F.size(); ++p) {
		g[p] -= static_cast<double>(hXjY[F[p]]);
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrHessianBlock(const std::vector<int>& F, std::vector<double>& H) {
	flushIntercept();
	if (!lrReady() || !secondOrderPrepare(F)) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	const size_t n = F.size();
	H.assign(n * n, 0.0);
	for (size_t i = 0; i < L.rows.size(); ++i) {
		const int k = L.rows[i];
		const double w = static_cast<double>(hKWeight[k]);
		if (w == 0.0) {
			continue;
		}
		const double p = readP(offsExpXBeta[k]);
		const double c = w * p * (1.0 - p);
		if (c == 0.0) {
			continue;
		}
		for (size_t a = L.rowStart[i]; a < L.rowStart[i + 1]; ++a) {
			const int pa = gsMap[L.position[a]];
			if (pa < 0) {
				continue;
			}
			const double ca = c * L.value[a];
			for (size_t b = L.rowStart[i]; b <= a; ++b) {
				const int pb = gsMap[L.position[b]];
				if (pb >= 0) { // lower triangle
					H[std::max(pa, pb) + std::min(pa, pb) * n] += ca * L.value[b];
				}
			}
		}
	}
	for (size_t j = 0; j < n; ++j) {
		for (size_t i = j + 1; i < n; ++i) {
			H[j + i * n] = H[i + j * n];
		}
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrHessianVector(const std::vector<int>& F, const std::vector<double>& v,
                                                         std::vector<double>& Hv) {
	flushIntercept();
	if (!lrReady() || !secondOrderPrepare(F) || v.size() != F.size()) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	std::vector<double> z;
	groupedProductXv(v, z);
	Hv.assign(F.size(), 0.0);
	for (size_t i = 0; i < L.rows.size(); ++i) {
		const int k = L.rows[i];
		const double w = static_cast<double>(hKWeight[k]);
		if (w == 0.0 || z[i] == 0.0) {
			continue;
		}
		const double p = readP(offsExpXBeta[k]);
		const double r = w * p * (1.0 - p) * z[i];
		for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
			const int q = gsMap[L.position[e]];
			if (q >= 0) {
				Hv[q] += L.value[e] * r;
			}
		}
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrHessianDiagonal(const std::vector<int>& F, std::vector<double>& h) {
	flushIntercept();
	if (!lrReady() || !secondOrderPrepare(F)) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	h.assign(F.size(), 0.0);
	for (size_t i = 0; i < L.rows.size(); ++i) {
		const int k = L.rows[i];
		const double w = static_cast<double>(hKWeight[k]);
		if (w == 0.0) {
			continue;
		}
		const double p = readP(offsExpXBeta[k]);
		const double c = w * p * (1.0 - p);
		for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
			const int q = gsMap[L.position[e]];
			if (q >= 0) {
				h[q] += c * L.value[e] * L.value[e];
			}
		}
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrSetDirection(const std::vector<int>& F, const std::vector<double>& d) {
	flushIntercept();
	if (!lrReady() || !secondOrderPrepare(F) || d.size() != F.size()) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	groupedProductXv(d, gsDirection);
	lrDirection.assign(K, 0.0);
	for (size_t i = 0; i < L.rows.size(); ++i) {
		lrDirection[L.rows[i]] = gsDirection[i];
	}
	gsHasDirection = true;
	return true;
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::lrTrialLogLikelihood(double t) {
	flushIntercept();
	if (!lrReady() || !gsHasDirection || lrDirection.size() != K) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	double sum = 0.0, compensation = 0.0;
	for (size_t k = 0; k < K; ++k) {
		const double w = static_cast<double>(hKWeight[k]);
		if (w != 0.0) {
			const double eta = static_cast<double>(hXBeta[k]) + t * lrDirection[k];
			grouped::addCompensated(sum, compensation,
				w * (static_cast<double>(hY[k]) * eta - logistic::log1pExp(eta)));
		}
	}
	return sum + compensation;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrCommitStep(double t) {
	flushIntercept();
	if (!lrReady() || !gsHasDirection || lrDirection.size() != K) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	for (size_t i = 0; i < L.rows.size(); ++i) {
		const int k = L.rows[i];
		if (gsDirection[i] == 0.0 || (trainingRowsOnly && hKWeight[k] == static_cast<RealType>(0))) {
			continue; // held-out rows stay as coordinate descent leaves them
		}
		hXBeta[k] += static_cast<RealType>(t * gsDirection[i]);
		offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k);
	}
	denominatorsStale = true; // denomPid = 1 + e, rebuilt on demand
	restartDrift();           // e rebuilt
	makeDirty();
	return true;
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::lrDirectionMaxAbs() const {
	if (!gsHasDirection || lrDirection.size() != K) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	double largest = 0.0;
	for (size_t k = 0; k < K; ++k) {
		if (hKWeight[k] != static_cast<RealType>(0)) {
			largest = std::max(largest, std::abs(lrDirection[k]));
		}
	}
	return largest;
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::lrNewtonCostRatio(const std::vector<int>& F, bool conjugateGradient,
                                                             bool preconditioned) {
	flushIntercept();
	if (!lrReady() || !secondOrderPrepare(F)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const grouped::Layout& L = gsLayout[0];
	// Per-unit costs in ns: a CCD visit per entry; for one Newton iteration, row outer products and the Cholesky
	// factor (dense) or about 45 Hv of two passes over the entries and one over the rows (CG), the gradient, and
	// per row the direction, two trials (log1p, exp) and the commit's exp
	const double aCycle = 4.0, aRows = 0.75, aCholesky = 0.035, aLinear = 1.0, aRow = 2.0, aAll = 40.0;
	double entries = 0.0, rows = 0.0;
	for (size_t i = 0; i < L.rows.size(); ++i) {
		double m = 0.0;
		for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
			m += gsMap[L.position[e]] >= 0;
		}
		entries += m;
		rows += 0.5 * m * m;
	}
	const double f = static_cast<double>(F.size());
	const double fixed = 2.0 * aLinear * entries + aAll * static_cast<double>(K);
	const double newton = conjugateGradient ?
		45.0 * (2.0 * aLinear * entries + aRow * static_cast<double>(L.rows.size())) + fixed :
		aRows * rows + aCholesky * f * f * f / 3.0 + fixed;
	return (entries > 0.0) ? newton / (aCycle * entries) : std::numeric_limits<double>::quiet_NaN();
}

} // namespace bsccs

#endif /* LOGISTICSECONDORDER_HPP_ */
