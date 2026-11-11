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
void ModelSpecifics<BaseModel,RealType>::setThreads(int threads) {
	if (!logisticSecondOrder || threads <= 1) {
		lrPool.reset();
	} else if (!lrPool || lrPool->size() != threads) {
		lrPool.reset(new WorkerPool(threads));
	}
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::lrHessianBlock(const std::vector<int>& F, std::vector<double>& H) {
	flushIntercept();
	if (!lrReady()) {
		return false;
	}
	const size_t n = F.size();
	if (n == 0) {
		H.clear();
		return true;
	}

	// H = X_F' diag(w p (1 - p)) X_F from each row's outer product. The rows of F's columns are transposed a chunk of
	// rows at a time into a small buffer (columns keep a cursor into their ascending rows), so no row-wise copy of X_F
	// is held: memory is the (permuted) block plus the buffer, then the block and H
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
	// The accumulation order of the columns: by entries, most first, so the pairs of the densest columns (most of the
	// work) fall in a block that stays in cache; each row then adds its pairs into whole columns of the upper
	// triangle, contiguous in memory (the order of F put each pair n doubles apart). Every entry is a sum over rows in
	// increasing order, whatever the order of the columns or the threads, so the block is the same bit for bit
	std::vector<size_t> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return count[a] > count[b]; });
	std::vector<double> permuted(n * n, 0.0);
	double* acc = permuted.data();

	const size_t chunk = std::max<size_t>(1024, std::min<size_t>(K, static_cast<size_t>(
		static_cast<double>(1 << 21) / std::max(entries / static_cast<double>(K), 1e-9)))); // about 2M entries
	std::vector<int> rowStart(chunk + 1);
	std::vector<int> position;
	std::vector<double> value;
	// Contiguous ranges of columns (in accumulation order), about equal in work, one task each: a column's pairs are
	// those with the earlier columns of its rows, about its entries times the entries of the denser columns in a row.
	// A task adds the pairs whose later column is in its range; in each row those are a run of the sorted positions.
	// On one thread the ranges are cache tiles, worth it only for a large block (every range passes over the rows):
	// one per 128 columns, at most 16 (measured best on real FeatureExtraction designs); with threads, 8 per thread,
	// at least 32 columns each
	const size_t tasks = lrPool ?
		std::max<size_t>(1, std::min<size_t>(8 * static_cast<size_t>(lrPool->size()), n / 32)) :
		std::max<size_t>(1, std::min<size_t>(16, n / 128));
	std::vector<size_t> rangeStart(tasks + 1, n); // the starts the loop below does not set stay n: empty ranges
	{
		std::vector<double> work(n);
		double before = 0.0, total = 0.0;
		for (size_t slot = 0; slot < n; ++slot) {
			const double c = static_cast<double>(count[order[slot]]);
			work[slot] = c * (before + 1.0);
			before += c / static_cast<double>(K);
			total += work[slot];
		}
		double run = 0.0;
		size_t task = 0;
		rangeStart[0] = 0;
		for (size_t slot = 0; slot < n && task + 1 < tasks; ++slot) {
			run += work[slot];
			while (task + 1 < tasks && run >= total * static_cast<double>(task + 1) / static_cast<double>(tasks)) {
				rangeStart[++task] = slot + 1;
			}
		}
	}
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
		for (size_t slot = 0; slot < n; ++slot) { // fill in accumulation order: positions ascend within each row
			const size_t p = order[slot];
			if (rows[p] == nullptr) {
				for (size_t k = r0; k < r1; ++k) {
					const int e = next[k - r0]++;
					position[e] = static_cast<int>(slot);
					value[e] = (values[p] == nullptr) ? 1.0 : static_cast<double>(values[p][k]);
				}
			} else {
				for (; cursor[p] < count[p] && static_cast<size_t>(rows[p][cursor[p]]) < r1; ++cursor[p]) {
					const size_t k = rows[p][cursor[p]];
					const int e = next[k - r0]++;
					position[e] = static_cast<int>(slot);
					value[e] = (values[p] == nullptr) ? 1.0 : static_cast<double>(values[p][cursor[p]]);
				}
			}
		}
		// task t adds the pairs whose later column is in its range: each entry has one owner
		auto accumulate = [&](size_t task, int) {
			const int lo = static_cast<int>(rangeStart[task]);
			const int hi = static_cast<int>(rangeStart[task + 1]);
			if (lo >= hi) {
				return;
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
				const int first = static_cast<int>(std::lower_bound(position.begin() + rowStart[i],
					position.begin() + rowStart[i + 1], lo) - position.begin());
				for (int a = first; a < rowStart[i + 1] && position[a] < hi; ++a) {
					const double ca = c * value[a];
					double* column = acc + static_cast<size_t>(position[a]) * n;
					for (int b = rowStart[i]; b <= a; ++b) {
						column[position[b]] += ca * value[b];
					}
				}
			}
		};
		if (lrPool) {
			lrPool->run(tasks, accumulate);
		} else {
			for (size_t task = 0; task < tasks; ++task) {
				accumulate(task, 0);
			}
		}
	}
	// the upper triangle, in accumulation order, back to the order of F
	std::vector<int>().swap(position);
	std::vector<double>().swap(value);
	H.resize(n * n); // every entry is written below
	std::vector<size_t> slotOf(n);
	for (size_t slot = 0; slot < n; ++slot) {
		slotOf[order[slot]] = slot;
	}
	for (size_t b = 0; b < n; ++b) {
		for (size_t a = 0; a < n; ++a) {
			const size_t sa = slotOf[a], sb = slotOf[b];
			H[a + b * n] = (sa <= sb) ? permuted[sa + sb * n] : permuted[sb + sa * n];
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
