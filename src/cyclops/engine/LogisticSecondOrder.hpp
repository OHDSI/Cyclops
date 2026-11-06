/*
 * LogisticSecondOrder.hpp
 *
 * The second-order API of AbstractModelSpecifics (Newton plan 2.2, Phase 7) for logistic regression in double
 * precision: l = sum_k w_k (y_k eta_k - log(1 + exp(eta_k))), so g = X_F'(w (p - y)) and H = X_F' diag(w p (1 - p)) X_F,
 * with p read from e = offsExpXBeta as the coordinate-descent gradient reads it. Every call flushes a pending lazy
 * intercept first. Rows of weight 0 are skipped: under a fold fit with held-out rows unused their eta is stale. A
 * commit moves only the rows the direction moves (only the training rows when held-out rows are unused), then
 * starts a new drift epoch. Everything is computed from X's own column storage: the gradient, Hv, the diagonal and
 * the direction are column passes, and the dense Hessian block transposes a chunk of rows at a time.
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


// Column passes over X's own storage (no row-wise copy: the full gradient of prox-Newton's violator check, Hv, the
// direction): the dot of column j with a row vector, and the row vector plus a times column j
template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::lrColumnDot(int j, const std::vector<double>& a, bool squared) const {
	double sum = 0.0;
	for (GenericIterator<RealType> it(hX, j); it; ++it) {
		const double x = static_cast<double>(it.value());
		sum += (squared ? x * x : x) * a[it.index()];
	}
	return sum;
}

template <class BaseModel, typename RealType>
void ModelSpecifics<BaseModel,RealType>::lrColumnAxpy(int j, double a, std::vector<double>& y) const {
	if (a == 0.0) {
		return;
	}
	for (GenericIterator<RealType> it(hX, j); it; ++it) {
		y[it.index()] += a * static_cast<double>(it.value());
	}
}

// w p (curvature = false) or w p (1 - p) (true) by row, 0 for rows of weight 0
template <class BaseModel, typename RealType>
void ModelSpecifics<BaseModel,RealType>::lrRowWeights(bool curvature, std::vector<double>& a) const {
	a.assign(K, 0.0);
	for (size_t k = 0; k < K; ++k) {
		const double w = static_cast<double>(hKWeight[k]);
		if (w != 0.0) {
			const double p = readP(offsExpXBeta[k]);
			a[k] = curvature ? w * p * (1.0 - p) : w * p;
		}
	}
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrFullGradient(const std::vector<int>& F, std::vector<double>& g) {
	flushIntercept();
	if (!lrReady()) {
		return false;
	}
	std::vector<double> a;
	lrRowWeights(false, a);
	g.resize(F.size());
	for (size_t p = 0; p < F.size(); ++p) {
		g[p] = lrColumnDot(F[p], a, false) - static_cast<double>(hXjY[F[p]]);
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrHessianBlock(const std::vector<int>& F, std::vector<double>& H) {
	flushIntercept();
	if (!lrReady()) {
		return false;
	}
	const size_t n = F.size();
	H.assign(n * n, 0.0);
	if (n == 0) {
		return true;
	}

	// H = X_F' diag(w p (1 - p)) X_F from each row's outer product. The rows of F's columns are transposed a chunk of
	// rows at a time into a small buffer (columns keep a cursor into their ascending rows), so no row-wise copy of X_F
	// is held: memory is H plus the buffer
	std::vector<const int*> rows(n);
	std::vector<const RealType*> values(n);
	std::vector<size_t> count(n), cursor(n, 0);
	double entries = 0.0;
	for (size_t p = 0; p < n; ++p) {
		const FormatType format = hX.getFormatType(F[p]);
		const bool dense = format == DENSE || format == INTERCEPT;
		rows[p] = dense ? nullptr : hX.getCompressedColumnVector(F[p]);
		values[p] = (format == SPARSE || format == DENSE) ? hX.getDataVector(F[p]) : nullptr;
		count[p] = dense ? K : hX.getNumberOfEntries(F[p]);
		entries += static_cast<double>(count[p]);
	}
	const size_t chunk = std::max<size_t>(1024, std::min<size_t>(K, static_cast<size_t>(
		static_cast<double>(1 << 21) / std::max(entries / static_cast<double>(K), 1e-9)))); // about 2M entries
	std::vector<int> rowStart(chunk + 1);
	std::vector<int> position;
	std::vector<double> value;
	for (size_t r0 = 0; r0 < K; r0 += chunk) {
		const size_t r1 = std::min(K, r0 + chunk);
		std::fill(rowStart.begin(), rowStart.end(), 0);
		for (size_t p = 0; p < n; ++p) { // count, without moving the cursors
			if (rows[p] == nullptr) {
				for (size_t k = r0; k < r1; ++k) {
					++rowStart[k - r0 + 1];
				}
			} else {
				for (size_t e = cursor[p]; e < count[p] && static_cast<size_t>(rows[p][e]) < r1; ++e) {
					++rowStart[rows[p][e] - r0 + 1];
				}
			}
		}
		for (size_t i = 0; i < r1 - r0; ++i) {
			rowStart[i + 1] += rowStart[i];
		}
		position.resize(rowStart[r1 - r0]);
		value.resize(rowStart[r1 - r0]);
		std::vector<int> next(rowStart.begin(), rowStart.begin() + (r1 - r0));
		for (size_t p = 0; p < n; ++p) { // fill column by column: positions ascend within each row
			if (rows[p] == nullptr) {
				for (size_t k = r0; k < r1; ++k) {
					const int e = next[k - r0]++;
					position[e] = static_cast<int>(p);
					value[e] = (values[p] == nullptr) ? 1.0 : static_cast<double>(values[p][k]);
				}
			} else {
				for (; cursor[p] < count[p] && static_cast<size_t>(rows[p][cursor[p]]) < r1; ++cursor[p]) {
					const size_t k = rows[p][cursor[p]];
					const int e = next[k - r0]++;
					position[e] = static_cast<int>(p);
					value[e] = (values[p] == nullptr) ? 1.0 : static_cast<double>(values[p][cursor[p]]);
				}
			}
		}
		for (size_t i = 0; i < r1 - r0; ++i) {
			const size_t k = r0 + i;
			const double w = static_cast<double>(hKWeight[k]);
			if (w == 0.0) {
				continue;
			}
			const double pk = readP(offsExpXBeta[k]);
			const double c = w * pk * (1.0 - pk);
			if (c == 0.0) {
				continue;
			}
			for (int a = rowStart[i]; a < rowStart[i + 1]; ++a) {
				const double ca = c * value[a];
				double* column = H.data() + position[a];
				for (int b = rowStart[i]; b <= a; ++b) {
					column[position[b] * n] += ca * value[b];
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
	if (!lrReady() || v.size() != F.size()) {
		return false;
	}
	std::vector<double> c, z(K, 0.0);
	lrRowWeights(true, c);
	for (size_t p = 0; p < F.size(); ++p) {
		lrColumnAxpy(F[p], v[p], z);
	}
	for (size_t k = 0; k < K; ++k) {
		z[k] *= c[k];
	}
	Hv.resize(F.size());
	for (size_t p = 0; p < F.size(); ++p) {
		Hv[p] = lrColumnDot(F[p], z, false);
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrHessianDiagonal(const std::vector<int>& F, std::vector<double>& h) {
	flushIntercept();
	if (!lrReady()) {
		return false;
	}
	std::vector<double> c;
	lrRowWeights(true, c);
	h.resize(F.size());
	for (size_t p = 0; p < F.size(); ++p) {
		h[p] = lrColumnDot(F[p], c, true);
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrSetDirection(const std::vector<int>& F, const std::vector<double>& d) {
	flushIntercept();
	if (!lrReady() || d.size() != F.size()) {
		return false;
	}
	lrDirection.assign(K, 0.0);
	for (size_t p = 0; p < F.size(); ++p) {
		lrColumnAxpy(F[p], d[p], lrDirection);
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
	for (size_t k = 0; k < K; ++k) {
		if (lrDirection[k] == 0.0 || (trainingRowsOnly && hKWeight[k] == static_cast<RealType>(0))) {
			continue; // held-out rows stay as coordinate descent leaves them
		}
		hXBeta[k] += static_cast<RealType>(t * lrDirection[k]);
		offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], static_cast<int>(k));
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
	if (!lrReady()) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	// Per-unit costs in ns: a CCD visit per entry; for one Newton iteration, row outer products and the Cholesky
	// factor (dense) or about 45 Hv of two passes over the entries and two over the rows (CG), the gradient, and
	// per row the direction, two trials (log1p, exp) and the commit's exp
	const double aCycle = 4.0, aRows = 0.75, aCholesky = 0.035, aLinear = 1.0, aRow = 2.0, aAll = 40.0;
	std::vector<int> count(K, 0); // entries of F by row
	double entries = 0.0;
	for (int j : F) {
		for (GenericIterator<RealType> it(hX, j); it; ++it) {
			++count[it.index()];
			entries += 1.0;
		}
	}
	double rows = 0.0;
	for (size_t k = 0; k < K; ++k) {
		rows += 0.5 * static_cast<double>(count[k]) * static_cast<double>(count[k]);
	}
	const double f = static_cast<double>(F.size());
	const double fixed = 2.0 * aLinear * entries + aAll * static_cast<double>(K);
	const double newton = conjugateGradient ?
		45.0 * (2.0 * aLinear * entries + 2.0 * aRow * static_cast<double>(K)) + fixed :
		aRows * rows + aCholesky * f * f * f / 3.0 + fixed;
	return (entries > 0.0) ? newton / (aCycle * entries) : std::numeric_limits<double>::quiet_NaN();
}

} // namespace bsccs

#endif /* LOGISTICSECONDORDER_HPP_ */
