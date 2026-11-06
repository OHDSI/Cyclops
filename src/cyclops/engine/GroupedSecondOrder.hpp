/*
 * GroupedSecondOrder.hpp
 *
 * The second-order API of AbstractModelSpecifics (Newton plan 2.2, Phase 7) for the grouped models cpr, sccs and
 * clr (its Breslow form; clr_efron and clr_exact are other models). With u_k = offsExpXBeta[k] over every row of a
 * stratum, D_s = sum_{k in s} u_k and n_s the stratum's (weighted) event count, so that held-out strata have
 * n_s = 0, the log likelihood is sum_k w_k y_k eta_k - sum_s n_s log D_s: one risk set per stratum. With
 * h_s = n_s / D_s,
 *   g = X_F'(h u) - X_F'wy,   H = X_F' diag(h u) X_F - sum_s a_s a_s',   a_s = sqrt(n_s) / D_s sum_{k in s} u_k x_k,
 * the gradient and Hessian of the negative log likelihood. Each a_s touches only the columns of its stratum's rows
 * (a sparse A). The kernels visit only the rows with an entry in F (grouped::Layout); a trial step exponentiates
 * only the rows it moves. Included at the end of ModelSpecifics.hpp.
 */

#ifndef GROUPEDSECONDORDER_HPP_
#define GROUPEDSECONDORDER_HPP_

namespace bsccs {

namespace grouped {
inline void addCompensated(double& sum, double& compensation, double value) { // Neumaier
	const double t = sum + value;
	compensation += (std::abs(sum) >= std::abs(value)) ? (sum - t) + value : (value - t) + sum;
	sum = t;
}
} // namespace grouped

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::groupedReady() const {
	return groupedSecondOrder && denomPid.size() >= N && hNWeight.size() >= N && offsExpXBeta.size() >= K &&
		hKWeight.size() == K && hXjY.size() >= J;
}

template <class BaseModel, typename RealType>
void ModelSpecifics<BaseModel,RealType>::prepareGroupedStrata() {
	if (gsStratumSize.size() == N) {
		return; // strata are fixed for these models
	}
	gsStratumSize.assign(N, 0);
	gsSortedStrata = true;
	for (size_t k = 0; k < K; ++k) {
		++gsStratumSize[hPid[k]];
		gsSortedStrata = gsSortedStrata && (k == 0 || hPid[k - 1] <= hPid[k]);
	}
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::secondOrderPrepare(const std::vector<int>& F) {

	if (!groupedSecondOrder && !logisticSecondOrder) {
		return false;
	}
	if (gsLayout[0].prepared && gsColumns == F) {
		return true;
	}
	for (int j : F) {
		if (j < 0 || static_cast<size_t>(j) >= J) {
			return false;
		}
	}
	gsHasDirection = false;
	// Columns F as a view of a layout: gsMap takes the layout's positions to positions in F (-1 when absent). Two
	// layouts, most recent first; a subset of the current one (prox-Newton's working set) is a view of it unless
	// it has under a quarter of its entries
	auto view = [this, &F](int slot) {
		const grouped::Layout& L = gsLayout[slot];
		if (!L.prepared) {
			return false;
		}
		if (gsColumnIndex.size() != J) {
			gsColumnIndex.assign(J, -1);
		}
		for (size_t p = 0; p < F.size(); ++p) {
			gsColumnIndex[F[p]] = static_cast<int>(p);
		}
		std::vector<int> map(L.F.size());
		size_t found = 0;
		double entries = 0.0;
		for (size_t q = 0; q < L.F.size(); ++q) {
			map[q] = gsColumnIndex[L.F[q]];
			if (map[q] >= 0) {
				++found;
				entries += static_cast<double>(hX.getNumberOfNonZeroEntries(L.F[q]));
			}
		}
		for (int j : F) {
			gsColumnIndex[j] = -1;
		}
		if (found != F.size() || (L.F != F && entries < 0.25 * static_cast<double>(L.rowStart.back()))) {
			return false;
		}
		if (slot == 1) {
			std::swap(gsLayout[0], gsLayout[1]);
		}
		gsMap.swap(map);
		gsColumns = F;
		return true;
	};
	if (view(0) || view(1)) {
		return true;
	}
	std::swap(gsLayout[0], gsLayout[1]);
	grouped::Layout& L = gsLayout[0];

	// Rows with an entry in F, ordered by stratum then row (the rows of these models are stored by stratum);
	// logistic regression has no strata
	if (!logisticSecondOrder) {
		prepareGroupedStrata();
	}
	if (gsRowIndex.size() != K) {
		gsRowIndex.assign(K, -1);
	}
	for (size_t p = 0; p < F.size(); ++p) {
		for (GenericIterator<RealType> it(hX, F[p]); it; ++it) {
			gsRowIndex[it.index()] = 0;
		}
	}
	L.rows.clear();
	for (size_t k = 0; k < K; ++k) {
		if (gsRowIndex[k] == 0) {
			L.rows.push_back(static_cast<int>(k));
		}
	}
	if (!logisticSecondOrder && !gsSortedStrata) {
		std::stable_sort(L.rows.begin(), L.rows.end(), [this](int a, int b) { return hPid[a] < hPid[b]; });
	}
	for (size_t i = 0; i < L.rows.size(); ++i) {
		gsRowIndex[L.rows[i]] = static_cast<int>(i);
	}

	// Their entries, row by row, positions ascending within a row
	L.rowStart.assign(L.rows.size() + 1, 0);
	for (size_t p = 0; p < F.size(); ++p) {
		for (GenericIterator<RealType> it(hX, F[p]); it; ++it) {
			++L.rowStart[gsRowIndex[it.index()] + 1];
		}
	}
	for (size_t i = 0; i < L.rows.size(); ++i) {
		L.rowStart[i + 1] += L.rowStart[i];
	}
	L.position.resize(L.rowStart.back());
	L.value.resize(L.rowStart.back());
	std::vector<size_t> next(L.rowStart.begin(), L.rowStart.end() - 1);
	for (size_t p = 0; p < F.size(); ++p) {
		for (GenericIterator<RealType> it(hX, F[p]); it; ++it) {
			const size_t e = next[gsRowIndex[it.index()]]++;
			L.position[e] = static_cast<int>(p);
			L.value[e] = static_cast<double>(it.value());
		}
	}
	for (int k : L.rows) {
		gsRowIndex[k] = -1;
	}

	// Stratum segments of those rows, and the work of the A'A term: sum_s t_s (t_s + 1) / 2 over the t_s columns
	// each stratum touches
	L.segmentStratum.clear();
	L.segmentStart.assign(1, 0);
	for (size_t i = 0; i < L.rows.size() && !logisticSecondOrder; ++i) {
		if (i == 0 || hPid[L.rows[i]] != hPid[L.rows[i - 1]]) {
			if (i > 0) {
				L.segmentStart.push_back(static_cast<int>(i));
			}
			L.segmentStratum.push_back(hPid[L.rows[i]]);
		}
	}
	L.segmentStart.push_back(static_cast<int>(L.rows.size()));
	if (L.rows.empty()) {
		L.segmentStart.assign(1, 0);
	}
	L.F = F;
	L.prepared = true;
	gsMap.resize(F.size());
	for (size_t p = 0; p < F.size(); ++p) {
		gsMap[p] = static_cast<int>(p);
	}
	gsColumns = F;
	return true;
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::stratumHazard(size_t s) const {
	const double n = static_cast<double>(hNWeight[s]);
	return (n != 0.0) ? n / static_cast<double>(denomPid[s]) : 0.0;
}

template <class BaseModel, typename RealType>
void ModelSpecifics<BaseModel,RealType>::groupedProductXv(const std::vector<double>& v, std::vector<double>& z) const {
	const grouped::Layout& L = gsLayout[0];
	z.assign(L.rows.size(), 0.0);
	for (size_t i = 0; i < L.rows.size(); ++i) {
		double sum = 0.0;
		for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
			const int p = gsMap[L.position[e]];
			if (p >= 0) {
				sum += L.value[e] * v[p];
			}
		}
		z[i] = sum;
	}
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::fullGradient(const std::vector<int>& F, std::vector<double>& g) {

	if (logisticSecondOrder) {
		return lrFullGradient(F, g);
	}

	if (!groupedReady() || !secondOrderPrepare(F)) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	g.assign(F.size(), 0.0);
	for (size_t seg = 0; seg < L.segmentStratum.size(); ++seg) {
	const double h = stratumHazard(L.segmentStratum[seg]);
	for (int i = L.segmentStart[seg]; i < L.segmentStart[seg + 1]; ++i) {
		const int k = L.rows[i];
		const double r = h * static_cast<double>(offsExpXBeta[k]);
		if (r != 0.0) {
			for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
				const int p = gsMap[L.position[e]];
				if (p >= 0) {
					g[p] += L.value[e] * r;
				}
			}
		}
	}
	}
	for (size_t p = 0; p < F.size(); ++p) {
		g[p] -= static_cast<double>(hXjY[F[p]]);
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::hessianBlock(const std::vector<int>& F, std::vector<double>& H) {

	if (logisticSecondOrder) {
		return lrHessianBlock(F, H);
	}

	if (!groupedReady() || !secondOrderPrepare(F)) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	const size_t n = F.size();
	H.assign(n * n, 0.0);
	if (n == 0) {
		return true;
	}

	// S2 = X_F' diag(h u) X_F, lower triangle, from each row's outer product
	for (size_t seg = 0; seg < L.segmentStratum.size(); ++seg) {
	const double h = stratumHazard(L.segmentStratum[seg]);
	for (int i = L.segmentStart[seg]; i < L.segmentStart[seg + 1]; ++i) {
		const int k = L.rows[i];
		const double r = h * static_cast<double>(offsExpXBeta[k]);
		if (r != 0.0) {
			for (size_t a = L.rowStart[i]; a < L.rowStart[i + 1]; ++a) {
				const int pa = gsMap[L.position[a]];
				if (pa < 0) {
					continue;
				}
				const double ra = r * L.value[a];
				for (size_t b = L.rowStart[i]; b <= a; ++b) {
					const int pb = gsMap[L.position[b]];
					if (pb >= 0) { // lower triangle
						H[std::max(pa, pb) + std::min(pa, pb) * n] += ra * L.value[b];
					}
				}
			}
		}
	}
	}

	// - sum_s a_s a_s' over the columns each stratum touches, in ascending order
	std::vector<double> acc(n, 0.0);
	std::vector<char> mark(n, 0);
	std::vector<int> touched;
	for (size_t g = 0; g < L.segmentStratum.size(); ++g) {
		const int s = L.segmentStratum[g];
		const double d = static_cast<double>(hNWeight[s]);
		if (d == 0.0) {
			continue;
		}
		touched.clear();
		for (int i = L.segmentStart[g]; i < L.segmentStart[g + 1]; ++i) {
			const double u = static_cast<double>(offsExpXBeta[L.rows[i]]);
			for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
				const int p = gsMap[L.position[e]];
				if (p < 0) {
					continue;
				}
				if (!mark[p]) {
					mark[p] = 1;
					touched.push_back(p);
				}
				acc[p] += u * L.value[e];
			}
		}
		std::sort(touched.begin(), touched.end());
		const double scale = std::sqrt(d) / static_cast<double>(denomPid[s]);
		for (int p : touched) {
			acc[p] *= scale;
		}
		for (size_t a = 0; a < touched.size(); ++a) {
			const int pa = touched[a];
			double* column = H.data() + pa;
			const double va = acc[pa];
			for (size_t b = 0; b <= a; ++b) {
				column[touched[b] * n] -= va * acc[touched[b]];
			}
		}
		for (int p : touched) {
			acc[p] = 0.0;
			mark[p] = 0;
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
bool ModelSpecifics<BaseModel,RealType>::hessianVector(const std::vector<int>& F, const std::vector<double>& v,
                                                       std::vector<double>& Hv) {

	if (logisticSecondOrder) {
		return lrHessianVector(F, v, Hv);
	}

	if (!groupedReady() || !secondOrderPrepare(F) || v.size() != F.size()) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	std::vector<double> z;
	groupedProductXv(v, z);
	// row k: u_k (h_s z_k - q_s), q_s = n_s M_s / D_s^2, M_s = sum_{k in s} u_k z_k
	Hv.assign(F.size(), 0.0);
	for (size_t g = 0; g < L.segmentStratum.size(); ++g) {
		const int s = L.segmentStratum[g];
		if (hNWeight[s] == static_cast<RealType>(0)) {
			continue;
		}
		double M = 0.0;
		for (int i = L.segmentStart[g]; i < L.segmentStart[g + 1]; ++i) {
			M += static_cast<double>(offsExpXBeta[L.rows[i]]) * z[i];
		}
		const double D = static_cast<double>(denomPid[s]);
		const double q = static_cast<double>(hNWeight[s]) * M / (D * D);
		const double h = stratumHazard(s);
		for (int i = L.segmentStart[g]; i < L.segmentStart[g + 1]; ++i) {
			const double r = static_cast<double>(offsExpXBeta[L.rows[i]]) * (h * z[i] - q);
			for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
				const int p = gsMap[L.position[e]];
				if (p >= 0) {
					Hv[p] += L.value[e] * r;
				}
			}
		}
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::setDirection(const std::vector<int>& F, const std::vector<double>& d) {

	if (logisticSecondOrder) {
		return lrSetDirection(F, d);
	}

	if (!groupedReady() || !secondOrderPrepare(F) || d.size() != F.size()) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	groupedProductXv(d, gsDirection);

	// What a trial step does not change: sum_k w_k y_k eta_k, and in each stratum the u of the rows it does not
	// move (summed directly, not as D_s minus the moving rows)
	if (gsMoving.size() != K) {
		gsMoving.assign(K, 0);
	}
	for (size_t i = 0; i < L.rows.size(); ++i) {
		gsMoving[L.rows[i]] = gsDirection[i] != 0.0;
	}
	double numerator = 0.0, numeratorCompensation = 0.0;
	gsRest.assign(N, 0.0);
	for (size_t k = 0; k < K; ++k) {
		if (hY[k] != static_cast<RealType>(0)) {
			grouped::addCompensated(numerator, numeratorCompensation,
				static_cast<double>(BaseModel::logLikeNumeratorContrib(hY[k], hXBeta[k])) * static_cast<double>(hKWeight[k]));
		}
		if (!gsMoving[k]) {
			gsRest[hPid[k]] += static_cast<double>(offsExpXBeta[k]);
		}
	}
	gsSlope = 0.0;
	for (size_t i = 0; i < L.rows.size(); ++i) {
		const int k = L.rows[i];
		if (hY[k] != static_cast<RealType>(0)) {
			gsSlope += static_cast<double>(hY[k]) * static_cast<double>(hKWeight[k]) * gsDirection[i]; // y eta terms
		}
		gsMoving[k] = 0;
	}
	// strata no row of F reaches
	std::vector<char> reached(N, 0);
	for (int s : L.segmentStratum) {
		reached[s] = 1;
	}
	double constant = 0.0, constantCompensation = 0.0;
	for (size_t s = 0; s < N; ++s) {
		if (!reached[s] && hNWeight[s] != static_cast<RealType>(0)) {
			grouped::addCompensated(constant, constantCompensation, static_cast<double>(hNWeight[s]) * std::log(gsRest[s]));
		}
	}
	gsNumerator = numerator + numeratorCompensation;
	gsConstant = constant + constantCompensation;
	gsHasDirection = true;
	return true;
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::trialLogLikelihood(double t) {

	if (logisticSecondOrder) {
		return lrTrialLogLikelihood(t);
	}

	if (!groupedReady() || !gsHasDirection) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const grouped::Layout& L = gsLayout[0];
	gsTrialExp.resize(L.rows.size());
	gsTrialT = t;
	double denominator = gsConstant, denominatorCompensation = 0.0;
	for (size_t g = 0; g < L.segmentStratum.size(); ++g) {
		const int s = L.segmentStratum[g];
		const double n = static_cast<double>(hNWeight[s]);
		if (n == 0.0) {
			continue;
		}
		double D = gsRest[s];
		for (int i = L.segmentStart[g]; i < L.segmentStart[g + 1]; ++i) {
			if (gsDirection[i] != 0.0) {
				const double e = (t == 0.0) ? 1.0 : std::exp(t * gsDirection[i]);
				gsTrialExp[i] = e;
				D += static_cast<double>(offsExpXBeta[L.rows[i]]) * e;
			}
		}
		grouped::addCompensated(denominator, denominatorCompensation, n * std::log(D));
	}
	double value = gsNumerator + t * gsSlope - (denominator + denominatorCompensation);
	if (BaseModel::likelihoodHasFixedTerms) {
		value += static_cast<double>(logLikelihoodFixedTerm);
	}
	return value; // an overflowing exp gives -Inf or NaN, which the line searches reject
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::commitStep(double t) {

	if (logisticSecondOrder) {
		return lrCommitStep(t);
	}

	if (!groupedReady() || !gsHasDirection) {
		return false;
	}
	const grouped::Layout& L = gsLayout[0];
	// The moving rows' u from the accepted trial's exp(t z) (the line searches end on it), the others unchanged;
	// each reached stratum's D as the unmoved rest plus the moved rows
	const bool reuse = t == gsTrialT && gsTrialExp.size() == L.rows.size();
	for (size_t g = 0; g < L.segmentStratum.size(); ++g) {
		const int s = L.segmentStratum[g];
		double D = gsRest[s];
		for (int i = L.segmentStart[g]; i < L.segmentStart[g + 1]; ++i) {
			const int k = L.rows[i];
			if (gsDirection[i] != 0.0) {
				hXBeta[k] += static_cast<RealType>(t * gsDirection[i]);
				const bool known = reuse && hNWeight[s] != static_cast<RealType>(0); // strata the trials skipped
				offsExpXBeta[k] = known ? static_cast<RealType>(static_cast<double>(offsExpXBeta[k]) * gsTrialExp[i]) :
					BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k);
			}
			D += static_cast<double>(offsExpXBeta[k]) * (gsDirection[i] != 0.0);
		}
		denomPid[s] = static_cast<RealType>(D);
	}
	denomMove = 0.0;
	restartDrift(); // e rebuilt
	denominatorsStale = false;
	makeDirty();
	return true;
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::directionMaxAbs() const {

	if (logisticSecondOrder) {
		return lrDirectionMaxAbs();
	}

	if (!gsHasDirection) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const grouped::Layout& L = gsLayout[0];
	double largest = 0.0;
	for (size_t i = 0; i < L.rows.size(); ++i) {
		if (hNWeight[hPid[L.rows[i]]] != static_cast<RealType>(0)) { // strata without events are not in the likelihood
			largest = std::max(largest, std::abs(gsDirection[i]));
		}
	}
	return largest;
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::newtonCostRatio(const std::vector<int>& F) {

	if (logisticSecondOrder) {
		return lrNewtonCostRatio(F, false, false);
	}

	if (!groupedReady() || !secondOrderPrepare(F)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const grouped::Layout& L = gsLayout[0];
	// Per-unit costs in ns: a CCD visit per entry (gradient by stratum, exp and denominator update); for one dense
	// iteration, row outer products, the per-stratum outer products of A, the Cholesky factor, the passes over the
	// rows of F (gradient, two trials) and over all rows (the direction's fixed parts, and the commit's exp)
	const double aCycle = 6.0, aRows = 0.75, aStrata = 1.0, aCholesky = 0.035, aLinear = 1.0, aAll = 8.0;
	// over the view: F's entries, row outer products and stratum pairs (the kernels also pass over the layout)
	double cycle = 0.0, rows = 0.0, pairs = 0.0;
	std::vector<char> mark(F.size(), 0);
	std::vector<int> touched;
	for (size_t g = 0; g < L.segmentStratum.size(); ++g) {
		touched.clear();
		for (int i = L.segmentStart[g]; i < L.segmentStart[g + 1]; ++i) {
			double m = 0.0;
			for (size_t e = L.rowStart[i]; e < L.rowStart[i + 1]; ++e) {
				const int p = gsMap[L.position[e]];
				if (p >= 0) {
					m += 1.0;
					if (!mark[p]) {
						mark[p] = 1;
						touched.push_back(p);
					}
				}
			}
			cycle += m;
			rows += 0.5 * m * m;
		}
		for (int p : touched) {
			mark[p] = 0;
		}
		pairs += 0.5 * static_cast<double>(touched.size()) * static_cast<double>(touched.size() + 1);
	}
	const double f = static_cast<double>(F.size());
	const double linear = static_cast<double>(L.rowStart.back()) + static_cast<double>(L.rows.size());
	const double newton = aRows * rows + aStrata * pairs + aCholesky * f * f * f / 3.0 + 4.0 * aLinear * linear +
		aAll * static_cast<double>(K);
	return (cycle > 0.0) ? newton / (aCycle * cycle) : std::numeric_limits<double>::quiet_NaN();
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::newtonCgCostRatio(const std::vector<int>& F, bool preconditioned) {
	if (logisticSecondOrder) {
		return lrNewtonCostRatio(F, true, preconditioned);
	}
	return std::numeric_limits<double>::quiet_NaN();
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::hessianDiagonal(const std::vector<int>& F, std::vector<double>& h) {

	if (logisticSecondOrder) {
		return lrHessianDiagonal(F, h);
	}

	if (!groupedReady()) {
		return false;
	}
	h.assign(F.size(), 0.0);
	for (size_t p = 0; p < F.size(); ++p) {
		double gradient = 0.0;
		if (hX.getNumberOfNonZeroEntries(F[p]) > 0) {
			computeGradientAndHessian(F[p], &gradient, &h[p], true); // weights are 1 without any
		}
	}
	return true;
}

template <class BaseModel, typename RealType>
bool ModelSpecifics<BaseModel,RealType>::separationDirections(const std::vector<int>& columns,
                                                              std::vector<int>& direction) {

	// Each stratum's weighted mean of x lies between its smallest and largest x, so the observed sum of x over
	// events equals sum_s n_s min_s x exactly when every event sits at its stratum's minimum: the likelihood then
	// does not increase in the coefficient (MLE at -Inf); likewise for the maximum
	if (!groupedSecondOrder || hNWeight.size() < N || hXjY.size() < J) {
		return false;
	}
	prepareGroupedStrata();
	direction.assign(columns.size(), 0);
	std::vector<double> low(N), high(N);
	std::vector<int> count(N, 0);
	std::vector<int> touched;
	for (size_t c = 0; c < columns.size(); ++c) {
		const int j = columns[c];
		touched.clear();
		for (GenericIterator<RealType> it(hX, j); it; ++it) {
			const int k = it.index();
			const double x = static_cast<double>(it.value());
			const int s = hPid[k];
			if (x == 0.0 || hNWeight[s] == static_cast<RealType>(0)) {
				continue;
			}
			if (count[s]++ == 0) {
				touched.push_back(s);
				low[s] = high[s] = x;
			} else {
				low[s] = std::min(low[s], x);
				high[s] = std::max(high[s], x);
			}
		}
		double lowest = 0.0, highest = 0.0, scale = 0.0;
		for (int s : touched) {
			const double d = static_cast<double>(hNWeight[s]);
			double l = low[s], h = high[s];
			if (count[s] < gsStratumSize[s]) { // rows with x = 0
				l = std::min(l, 0.0);
				h = std::max(h, 0.0);
			}
			lowest += d * l;
			highest += d * h;
			scale += d * std::max(std::abs(l), std::abs(h));
			count[s] = 0;
		}
		const double observed = static_cast<double>(hXjY[j]);
		const double tolerance = 1e-12 * std::max(scale, 1.0);
		if (highest - lowest > tolerance) { // the column varies within some stratum with events
			if (std::abs(observed - lowest) <= tolerance) {
				direction[c] = -1;
			} else if (std::abs(observed - highest) <= tolerance) {
				direction[c] = 1;
			}
		}
	}
	return true;
}

} // namespace bsccs

#endif /* GROUPEDSECONDORDER_HPP_ */
