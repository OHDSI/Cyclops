/*
 * NewtonSolver.cpp
 *
 * Certified "gap" stops and dense Newton moves inside CyclicCoordinateDescent::findMode (Newton plan 2.4-2.6, 3).
 * findMode stays the only driver: a Newton step moves every free coordinate at once through the engine's
 * second-order API, and CCD continues from the committed state whenever Newton is not worth it or fails.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <sstream>
#include <random>
#include <string>

#include "CyclicCoordinateDescent.h"
#include "Iterators.h"
#include "ModelData.h"
#include "NewtonSolver.h"

namespace bsccs {

namespace {

// Newton-CG's preconditioner without Woodbury, after Frangella, Tropp & Udell (2023):
// P = E + V S V', V S V' a randomized Nystrom approximation of the likelihood Hessian whose test matrix holds the
// unpenalized coordinates and nystromRank random sign columns, and E = Lambda + the part of the likelihood diagonal
// that V S V' misses (at least 1e-10 max S): the curvature outside V, coordinate by coordinate. (FTU's scalar tail,
// the sketch's last eigenvalue for every coordinate, also puts curvature on directions where the likelihood has none;
// E = Lambda alone leaves out that curvature, which dominates when the prior is spread.) Applied as P^-1 = E^-1/2
// [(I - Z Z') + Z diag(1/(1+Sigma)) Z'] E^-1/2, Z Sigma^1/2 the singular value decomposition of E^-1/2 V S^1/2: no
// cancellation between large terms, and Sigma <= max S / min E <= 1e10. Any fixed positive definite preconditioner
// serves CG, so it is built once per fit
struct NystromPreconditioner {
	Eigen::MatrixXd Z;
	Eigen::VectorXd invOnePlusSigma;
	Eigen::VectorXd invRootE;
	bool built = false;

	Eigen::VectorXd apply(const Eigen::VectorXd& v) const {
		const Eigen::VectorXd y = v.cwiseProduct(invRootE);
		const Eigen::VectorXd a = Z.transpose() * y;
		return ((y - Z * a) + Z * a.cwiseProduct(invOnePlusSigma)).cwiseProduct(invRootE);
	}
};

} // namespace

struct CyclicCoordinateDescent::NewtonContext {
	std::vector<int> F;             // free coordinates (normal, Laplace, elastic net or no priors)
	std::vector<double> precision;  // quadratic precision: 1 / sigma_j^2, or the elastic net's lambda2 (0 otherwise)
	std::vector<double> lambda;     // L1 rate: Laplace, or the elastic net's lambda1 (0 otherwise)
	std::vector<int> unpenalized;   // positions in F with precision 0, ascending (F and precision are fixed for the fit)
	bool anyLaplace = false;        // some L1 rate: prox-Newton
	bool allNormal = true;
	bool allStrong = true;          // every precision positive (normal, elastic net with alpha < 1): R certifies
	double tauEff = 0.0;
	double nuNormal = NAN;          // all-Normal form of nu, fixed for the fit
	double eta = 5.0;               // eta trust region: t max_k |x_k'd| <= eta (2.6)
	int rejections = 0;             // consecutive rejected line searches
	bool conjugateGradient = false; // Newton-CG instead of a dense factorization (2.5)
	bool jacobi = false;            // Jacobi-preconditioned CG (diagnostics only: it loses on these designs)
	// Rank-k Woodbury preconditioner P = M - B'B, M = S2 + Lambda (Phase 4 study: 8-16x fewer Hv), built at the
	// first Newton-CG iteration and rebuilt after a solve of more than 25 Hv
	bool woodburyAllowed = false;
	bool woodburyBuilt = false;
	bool woodburyStale = false;
	bool woodburyOffered = false;   // the engine gave Woodbury parts (counting-process Cox): no Nystrom preconditioner
	Eigen::LLT<Eigen::MatrixXd> woodburyM;
	Eigen::MatrixXd woodburyW;      // M^-1 B'
	Eigen::LLT<Eigen::MatrixXd> woodburyCapacitance; // I - B M^-1 B'
	NystromPreconditioner nystrom;  // without Woodbury, once the fit's Hessian-vector products reach its cost
	bool nystromTried = false;
	int lastPlainSolve = -1;        // Hv of the last forcing solve without a preconditioner
	double lastPlainPerDigit = -1.0; // the same per digit of residual reduction
	bool nystromJudged = false;     // its first forcing solve compared with lastPlainPerDigit
	double entries = 0.0;           // entries of X_F (a dense or intercept column counts K)
	int stopType = GAP;             // explicit types under algorithm = "newton" map onto Newton iterations
	double tolerance = 0.0;
	double lastObjective = NAN;
};

namespace {

// max over rows with positive weight of sum_j a_j |x_kj|^power, columns F
template <typename Positive>
double rowMaximum(const ModelData<double>& data, const std::vector<int>& F, const std::vector<double>& a,
                  int power, int K, Positive positive) {
	std::vector<double> rows(K, 0.0);
	for (size_t i = 0; i < F.size(); ++i) {
		if (a[i] == 0.0) {
			continue;
		}
		for (GenericIterator<double> it(data.getX(), F[i]); it; ++it) {
			const double x = std::abs(it.value());
			rows[it.index()] += a[i] * (power == 2 ? x * x : x);
		}
	}
	double largest = 0.0;
	for (int k = 0; k < K; ++k) {
		if (positive(k)) {
			largest = std::max(largest, rows[k]);
		}
	}
	return largest;
}

// Memory budget of one fit's Newton work (2.5): dense H, its factor and workspace, row-wise copies of columns of X
const double newtonMemoryBudget = 256.0 * 1024 * 1024; // bytes

// A Schur bound needs one Hessian-vector product per unpenalized coordinate; above this many there is none
const int schurMaxUnpenalized = 32;

// The Woodbury preconditioner's rank, and the bytes of what it keeps over n free coordinates: M and its Cholesky
// factor, and the woodburyRank columns of B' and of W = M^-1 B'
const int woodburyRank = 32;

double woodburyBytes(double n) {
	return 16.0 * n * n + 8.0 * 2.0 * woodburyRank * n;
}

// Random sign columns of the Nystrom preconditioner's test matrix: 64 beat 256 on the real ridge and SAR fits, where the
// build (one Hessian-vector product per column) costs more than the larger rank saves
const int nystromRank = 64;

// The Nystrom preconditioner's costs in Hessian-vector products (deterministic estimates, to be calibrated by the
// auto cost model): building it, one product per test column plus about 12 n m^2 flops of dense factorizations, which
// run about 10 times faster per flop than a product (about 4 flops per entry, bound by memory); and applying it in one
// CG iteration, two passes over n x m doubles against a product's two passes over the entries (index and value)
double nystromPrice(size_t n, int unpenalized, double entries) {
	const double m = std::min<double>(static_cast<double>(n), unpenalized + nystromRank);
	return m + 12.0 * n * m * m / std::max(40.0 * entries, 1.0);
}

double nystromApplyCost(size_t n, int columns, double entries) {
	return static_cast<double>(n) * columns / std::max(1.5 * entries, 1.0);
}

// The Nystrom preconditioner at the current point, given the likelihood diagonal h. False (CG then runs without one)
// when its matrices would exceed `available` bytes or the construction fails
bool buildNystrom(AbstractModelSpecifics& engine, const std::vector<int>& F, const std::vector<double>& precision,
                  const std::vector<int>& unpenalized, const std::vector<double>& h, double available,
                  NystromPreconditioner& P, int* products) {
	const size_t n = F.size();
	const int m = std::min<int>(static_cast<int>(n), static_cast<int>(unpenalized.size()) + nystromRank);
	// live at once: the test matrix and its QR, the products, B, V, W, both singular value decompositions' work and
	// factors, and the result
	const double bytes = 8.0 * (10.0 * static_cast<double>(n) * m + 4.0 * static_cast<double>(m) * m);
	if (bytes > available) {
		return false;
	}
	Eigen::MatrixXd omega = Eigen::MatrixXd::Zero(n, m);
	for (size_t c = 0; c < unpenalized.size(); ++c) {
		omega(unpenalized[c], c) = 1.0;
	}
	std::mt19937 generator(20261111); // random signs from the engine's own bits: the same on every platform and thread count
	for (int c = static_cast<int>(unpenalized.size()); c < m; ++c) {
		for (size_t i = 0; i < n; ++i) omega(i, c) = (generator() & 1u) ? 1.0 : -1.0;
	}
	{
		const Eigen::HouseholderQR<Eigen::MatrixXd> qr(omega);
		omega = qr.householderQ() * Eigen::MatrixXd::Identity(n, m);
	}
	Eigen::MatrixXd Y(n, m);
	std::vector<double> e(n), out;
	for (int c = 0; c < m; ++c) {
		for (size_t i = 0; i < n; ++i) e[i] = omega(i, c);
		if (!engine.hessianVector(F, e, out)) return false;
		++*products;
		for (size_t i = 0; i < n; ++i) Y(i, c) = out[i];
	}
	if (!Y.allFinite()) return false;
	const double shift = std::sqrt(static_cast<double>(n)) * std::numeric_limits<double>::epsilon() * Y.norm();
	if (!std::isfinite(shift)) return false;
	Y += shift * omega;
	const Eigen::MatrixXd M = omega.transpose() * Y;
	const Eigen::LLT<Eigen::MatrixXd> C(0.5 * (M + M.transpose()));
	if (C.info() != Eigen::Success) return false; // (LLT does not detect NaN: B is checked below)
	const Eigen::MatrixXd B = C.matrixL().solve(Y.transpose()).transpose();
	if (!B.allFinite()) return false;
	const Eigen::BDCSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU);
	if (svd.info() != Eigen::Success) return false; // on non-finite input it leaves its results unwritten
	const Eigen::VectorXd S = svd.singularValues().array().square() - shift; // sorted, largest first
	if (!(S(0) > 0.0) || !std::isfinite(S(0))) return false;
	const double floor = 1e-10 * S(0); // the relative threshold of SymmetricSolve: Sigma <= 1e10 below
	if (!(floor >= std::numeric_limits<double>::min())) return false; // not subnormal
	int k = 0; // directions above the floor (at least the first)
	while (k < S.size() && S(k) > floor) ++k;
	const Eigen::MatrixXd V = svd.matrixU().leftCols(k);
	Eigen::VectorXd E(n);
	for (size_t i = 0; i < n; ++i) {
		if (!std::isfinite(h[i])) return false; // e.g. 0/0 on a stratum without events
		double captured = 0.0;
		for (int c = 0; c < k; ++c) captured += V(i, c) * V(i, c) * S(c);
		E(i) = precision[i] + std::max(h[i] - captured, floor); // h_ii >= (V S V')_ii up to the shift and rounding
	}
	if (!E.allFinite()) return false; // an infinite E would make P^-1 singular
	P.invRootE = E.cwiseSqrt().cwiseInverse();
	const Eigen::MatrixXd W = P.invRootE.asDiagonal() * V * S.head(k).cwiseSqrt().asDiagonal();
	if (!W.allFinite()) return false;
	const Eigen::BDCSVD<Eigen::MatrixXd> w(W, Eigen::ComputeThinU);
	if (w.info() != Eigen::Success) return false;
	P.Z = w.matrixU();
	P.invOnePlusSigma = (1.0 + w.singularValues().array().square()).inverse().matrix();
	P.built = P.Z.allFinite() && P.invOnePlusSigma.allFinite() && P.invRootE.allFinite();
	return P.built;
}

// A lower bound P <= H for Newton-CG certificates when some free coordinates are unpenalized (Newton plan 3.2).
// With U the unpenalized positions in F, N the penalized ones, and the likelihood Hessian over F split as
// [[A, B'], [B, C]] (A over U), P = [[A, B'], [B, B A^-1 B' + Lambda_N]]: H - P = [[0, 0], [0, C - B A^-1 B']], the
// Schur complement of A in the likelihood Hessian, which is positive semidefinite, so P <= H and H^-1 <= P^-1 whenever
// A is positive definite. P = L diag(A, Lambda_N) L' with L = [[I, 0], [B A^-1, I]], so
// r'P^-1 r = r_U'A^-1 r_U + t'Lambda_N^-1 t, t = r_N - B A^-1 r_U, a sum of nonnegative terms. Without U, P = Lambda (the
// all-Normal bracket). The columns [A; B] are |U| Hessian-vector products H e_u at the current point.
// Rounding: quantities computed through A carry a relative error of order |U| cond(A) eps (Cholesky's backward error
// grown by the conditioning of A), so A is used only when cond(A) <= 1e10, the threshold of SymmetricSolve, and each
// such quantity is raised by slack = (64 + 4 |U| cond(A)) eps times its size.
struct SchurBound {
	explicit SchurBound(const std::vector<int>& U) : U(U) {}

	const std::vector<int>& U;     // positions in F with precision 0, ascending (NewtonContext::unpenalized)
	Eigen::MatrixXd Bt;            // |U| x n: column i is row i of H_likelihood [e_U]
	Eigen::LLT<Eigen::MatrixXd> A;
	double slack = 0.0;

	// An upper bound on r'P^-1 r, for P with the prior precisions e on N
	double quadratic(const Eigen::VectorXd& r, const std::vector<double>& e) const {
		const Eigen::Index n = r.size();
		const Eigen::Index u = static_cast<Eigen::Index>(U.size());
		Eigen::VectorXd rU(u);
		for (Eigen::Index c = 0; c < u; ++c) rU(c) = r(U[c]);
		const Eigen::VectorXd y = A.solve(rU);
		double sum = A.matrixL().solve(rU).squaredNorm(); // r_U'A^-1 r_U
		for (Eigen::Index i = 0, c = 0; i < n; ++i) {
			if (c < u && U[c] == i) { // in U
				++c;
				continue;
			}
			const double t = r(i) - Bt.col(i).dot(y);
			sum += t * t / e[i];
		}
		return (1.0 + slack) * sum;
	}
};

// The Schur bound at the current point, over the positions bound.U of F with precision 0. False, and no certificate, when
// there are none or more than schurMaxUnpenalized, when the bound and nu_P (schurNu) would need more than `available`
// bytes, or when the likelihood block A of the unpenalized coordinates is not positive definite with cond(A) <= 1e10
// (collinear unpenalized columns)
bool buildSchurBound(AbstractModelSpecifics& engine, const std::vector<int>& F, int K, double available,
                     SchurBound& bound, int* products) {
	const size_t n = F.size();
	const Eigen::Index u = static_cast<Eigen::Index>(bound.U.size());
	// live together in schurNu: Bt (|U| x n, counted twice), the U columns of every row (|U| x K) and three vectors of K
	const double bytes = 8.0 * (2.0 * u * static_cast<double>(n) + (u + 3.0) * K);
	if (u == 0 || u > schurMaxUnpenalized || bytes > available) {
		return false;
	}
	bound.Bt.resize(u, n);
	std::vector<double> e(n, 0.0), out;
	for (Eigen::Index c = 0; c < u; ++c) {
		e[bound.U[c]] = 1.0;
		if (!engine.hessianVector(F, e, out)) return false;
		++*products;
		e[bound.U[c]] = 0.0;
		for (size_t i = 0; i < n; ++i) bound.Bt(c, i) = out[i];
	}
	if (!bound.Bt.allFinite()) return false;
	Eigen::MatrixXd A(u, u);
	for (Eigen::Index c = 0; c < u; ++c) A.col(c) = bound.Bt.col(bound.U[c]);
	A = 0.5 * (A + A.transpose());
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(A, Eigen::EigenvaluesOnly);
	const double smallest = eigen.eigenvalues()(0), largest = eigen.eigenvalues()(u - 1);
	if (!(smallest > 0.0) || !(largest <= 1e10 * smallest)) return false;
	bound.A.compute(A);
	if (bound.A.info() != Eigen::Success) return false;
	bound.slack = (64.0 + 4.0 * u * (largest / smallest)) * std::numeric_limits<double>::epsilon();
	return true;
}

// nu_P = max_k sqrt(x_k'P^-1 x_k) >= max_k sqrt(x_k'H^-1 x_k) = nu over rows with positive weight, x_k row k of X_F.
// With g_i = A^-1 B_i for i in N, x'P^-1 x = x_U'A^-1 x_U + sum_N (x_i - g_i'x_U)^2 / lambda_i, which expands to
// x_U'Q x_U - 2 sum_N x_i g_i'x_U / lambda_i + sum_N x_i^2 / lambda_i, Q = A^-1 + sum_N g_i g_i' / lambda_i: one pass over
// X_F with the U columns of each row at hand. The expansion can cancel: the value is raised by the bound's slack times
// the size of its terms, and kept above x_U'A^-1 x_U (a term of the sum)
template <typename Positive>
double schurNu(const ModelData<double>& data, const std::vector<int>& F, const std::vector<double>& precision,
               const SchurBound& bound, int K, Positive positive) {
	const size_t n = F.size();
	const Eigen::Index u = static_cast<Eigen::Index>(bound.U.size());
	Eigen::MatrixXd XU = Eigen::MatrixXd::Zero(u, K); // the U columns of each row
	for (Eigen::Index c = 0; c < u; ++c) {
		for (GenericIterator<double> it(data.getX(), F[bound.U[c]]); it; ++it) {
			XU(c, it.index()) = it.value();
		}
	}
	Eigen::MatrixXd Q = bound.A.solve(Eigen::MatrixXd::Identity(u, u));
	std::vector<double> square(K, 0.0), cross(K, 0.0), size(K, 0.0); // sum_N x_i^2 / lambda_i, x_i g_i'x_U / lambda_i, |.|
	for (size_t i = 0, c = 0; i < n; ++i) {
		if (c < bound.U.size() && bound.U[c] == static_cast<int>(i)) { // in U
			++c;
			continue;
		}
		const Eigen::VectorXd g = bound.A.solve(Eigen::VectorXd(bound.Bt.col(i))), gAbs = g.cwiseAbs();
		Q.noalias() += g * g.transpose() / precision[i];
		for (GenericIterator<double> it(data.getX(), F[i]); it; ++it) {
			const int k = it.index();
			const double x = it.value();
			square[k] += x * x / precision[i];
			cross[k] += (x / precision[i]) * g.dot(XU.col(k));
			size[k] += (std::abs(x) / precision[i]) * gAbs.dot(XU.col(k).cwiseAbs());
		}
	}
	double largest = 0.0;
	for (int k = 0; k < K; ++k) {
		if (!positive(k)) continue;
		const Eigen::VectorXd xu = XU.col(k);
		const double quadratic = xu.dot(Q * xu);
		const double value = square[k] + quadratic - 2.0 * cross[k];
		const double floor = bound.A.matrixL().solve(xu).squaredNorm(); // x_U'A^-1 x_U
		largest = std::max(largest, std::max(value + bound.slack * (square[k] + std::abs(quadratic) + 2.0 * size[k]), floor));
	}
	return std::sqrt(largest);
}

} // namespace

std::string CyclicCoordinateDescent::secondOrderIneligibility() const {
	if (usingGPU || usingCUDA) {
		return "it runs on a GPU";
	}
	if (syncCV) {
		return "cross-validation folds are synchronized";
	}
	if (!modelSpecifics.supportsSecondOrder()) {
		return "the model has no second-order kernels (so far cox, cox_counting, cpr, sccs, clr and lr in double precision on the CPU)";
	}
	for (int j = 0; j < J; ++j) {
		double precision = 0.0, lambda = 0.0;
		if (!fixBeta[j] && !jointPrior->getNewtonPenalty(j, &precision, &lambda)) {
			return "a free coefficient has a prior other than normal, Laplace, elastic net or none";
		}
	}
	return "";
}

// One H at the current point: Lambda^2 / 2 and, when H is well conditioned, gsc (3.1)
bool CyclicCoordinateDescent::certifyGap(NewtonContext& context, double* gscValue, double* lambda2Half,
                                         Eigen::VectorXd* step) {
	*gscValue = std::numeric_limits<double>::infinity();
	*lambda2Half = NAN;
	const size_t n = context.F.size();
	std::vector<double> gradient, block;
	if (!modelSpecifics.fullGradient(context.F, gradient) || !modelSpecifics.hessianBlock(context.F, block)) {
		return false;
	}
	++convergenceReport.hessians;
	Eigen::VectorXd s(n);
	Eigen::MatrixXd H = Eigen::Map<Eigen::MatrixXd>(block.data(), n, n);
	for (size_t i = 0; i < n; ++i) {
		s(i) = gradient[i] + context.precision[i] * hBeta[context.F[i]];
		H(i, i) += context.precision[i];
	}
	const newton::SymmetricSolve solve(H, s);
	const double lambda2 = s.dot(solve.x);
	*lambda2Half = 0.5 * lambda2;
	if (step != nullptr) {
		*step = -solve.x; // the Newton step (minimum norm when H is singular)
	}
	if (solve.method != newton::SolveMethod::CHOLESKY || !(lambda2 >= 0.0)) {
		return false;
	}
	const auto* data = dynamic_cast<const ModelData<double>*>(&hXI);
	if (data == nullptr) {
		return false;
	}
	auto positive = [this](int k) {
		return !useCrossValidation || static_cast<size_t>(k) >= hWeights.size() || hWeights[k] > 0.0;
	};
	const Eigen::VectorXd inverse = solve.inverseDiagonal();
	std::vector<double> root(n);
	for (size_t i = 0; i < n; ++i) {
		root[i] = std::sqrt(std::max(inverse(i), 0.0));
	}
	double nu = rowMaximum(*data, context.F, root, 1, K, positive); // diagonal form
	if (context.allNormal) {
		nu = std::min(nu, context.nuNormal);
	}
	*gscValue = newton::gsc(2.0 * nu, std::sqrt(lambda2)); // c = 2 nu for the Cox engine
	return true;
}

// The Newton step d from a point the CCD path has just certified (or stopped at uncertified on a singular H), as the
// Newton path takes its last step: it costs a trial and a commit, and leaves the coefficients quadratically closer
// to the optimum than the certificate alone. Kept only when it decreases the penalized objective.
bool CyclicCoordinateDescent::finishingStep(NewtonContext& context, const Eigen::VectorXd& d) {
	const size_t n = context.F.size();
	if (static_cast<size_t>(d.size()) != n || !d.allFinite() || d.cwiseAbs().maxCoeff() == 0.0) {
		return false;
	}
	const std::vector<double> direction(d.data(), d.data() + n);
	if (!modelSpecifics.setDirection(context.F, direction)) {
		return false;
	}
	auto objective = [&](double t) {
		double penalty = 0.0;
		for (size_t i = 0; i < n; ++i) {
			const double b = hBeta[context.F[i]] + t * d(i);
			penalty += 0.5 * context.precision[i] * b * b;
		}
		return -modelSpecifics.trialLogLikelihood(t) + penalty;
	};
	const double largest = modelSpecifics.directionMaxAbs();
	double t = (largest > context.eta) ? context.eta / largest : 1.0;
	const double f0 = objective(0.0);
	bool accepted = false;
	for (int halving = 0; halving < 10 && !accepted; ++halving) {
		const double ft = objective(t);
		accepted = std::isfinite(ft) && ft <= f0;
		if (!accepted) {
			t *= 0.5;
		}
	}
	if (!accepted || !modelSpecifics.commitStep(t)) {
		return false;
	}
	for (size_t i = 0; i < n; ++i) {
		const int j = context.F[i];
		const double change = t * d(i);
		hBeta[j] += change;
		hDelta[j] = std::max(hDelta[j], 2.0 * std::abs(change));
	}
	kktConsecutive = 0;
	xBetaKnown = true;
	sufficientStatisticsKnown = true;
	return true;
}

bool CyclicCoordinateDescent::newtonDirectionDense(NewtonContext& context, Eigen::VectorXd& s, Eigen::VectorXd& d,
                                                   bool* stop, bool* certified, bool* damped) {

	const size_t n = context.F.size();
	std::vector<double> gradient, block;
	if (!modelSpecifics.fullGradient(context.F, gradient) || !modelSpecifics.hessianBlock(context.F, block)) {
		return false;
	}
	++convergenceReport.hessians;
	s.resize(n);
	Eigen::MatrixXd H = Eigen::Map<Eigen::MatrixXd>(block.data(), n, n);
	double kkt = 0.0;
	for (size_t i = 0; i < n; ++i) {
		s(i) = gradient[i] + context.precision[i] * hBeta[context.F[i]];
		H(i, i) += context.precision[i];
		if (H(i, i) > 0.0) {
			kkt = std::max(kkt, std::abs(s(i)) / std::sqrt(H(i, i)));
		}
	}
	if (!s.allFinite() || !H.allFinite()) {
		return false;
	}
	const newton::SymmetricSolve solve(H, s);
	d = -solve.x;
	const double lambda2 = -s.dot(d);
	convergenceReport.droppedDirections = solve.dropped;
	if (!(lambda2 >= 0.0) || !d.allFinite()) {
		return false;
	}
	convergenceReport.lambda2Half = 0.5 * lambda2;

	// Stop test at the current point (3.2): gsc <= tau_eff certifies it, and its last step is still taken; while
	// gsc is unavailable (c Lambda >= 1, or H not positive definite) Lambda^2 / 2 <= tau_eff stops uncertified.
	// gsc >= Lambda^2 / 2, so nu is only computed when that can pass.
	*stop = false;
	*certified = false;
	*damped = solve.method != newton::SolveMethod::CHOLESKY;
	if (context.stopType == GAP) {
		if (0.5 * lambda2 <= context.tauEff) {
			*stop = true;
			if (solve.method == newton::SolveMethod::CHOLESKY) {
				const auto* data = dynamic_cast<const ModelData<double>*>(&hXI);
				auto positive = [this](int k) {
					return !useCrossValidation || static_cast<size_t>(k) >= hWeights.size() || hWeights[k] > 0.0;
				};
				if (data != nullptr) {
					const Eigen::VectorXd inverse = solve.inverseDiagonal();
					std::vector<double> root(n);
					for (size_t i = 0; i < n; ++i) {
						root[i] = std::sqrt(std::max(inverse(i), 0.0));
					}
					double nu = rowMaximum(*data, context.F, root, 1, K, positive);
					if (context.allNormal) {
						nu = std::min(nu, context.nuNormal);
					}
					const double value = newton::gsc(2.0 * nu, std::sqrt(lambda2));
					convergenceReport.gsc = value;
					*certified = value <= context.tauEff;
				}
			}
		}
	} else if (context.stopType == KKT) {
		*stop = kkt < context.tolerance;
	}
	return true;
}


// Newton-CG direction (2.5, 3.2): conjugate gradients on H x = s, preconditioned by Woodbury (Cox counting) or by
// the Nystrom preconditioner once a fit's solves have cost what it does, with an inexact-Newton forcing term, at most
// 250 Hv. Certificates use a bound P <= H: for any x with residual r = s - H x,
// 2 s'x - x'Hx <= Lambda^2 <= 2 s'x - x'Hx + r'P^-1 r, and nu <= nu_P. P = Lambda and nu_P = nuNormal on all-Normal
// fits; with unpenalized (None) coordinates, the Schur bound (SchurBound, schurNu). When the estimate s'x / 2 is within
// tau_eff, a certification solve continues until the bracket certifies, gsc(2 nu_P, sqrt(upper)) <= tau_eff (checked on
// the recurrence residual, then confirmed with an explicit one); until a relative residual of
// 1e-10; or at most 1,000 Hv. After a bracket or a tight solve, Lambda^2 / 2 <= tau_eff stops uncertified when the
// explicit certificate fails.
// Without a bound (more than 32 unpenalized coordinates, their likelihood block ill-conditioned, or over the memory budget)
// only the last two (the longer solve gives a more exact step, after which the next iteration certifies).
bool CyclicCoordinateDescent::newtonDirectionCG(NewtonContext& context, Eigen::VectorXd& s, Eigen::VectorXd& d,
                                                bool* stop, bool* certified) {
	const size_t n = context.F.size();
	std::vector<double> gradient;
	if (!modelSpecifics.fullGradient(context.F, gradient)) {
		return false;
	}
	s.resize(n);
	for (size_t i = 0; i < n; ++i) {
		s(i) = gradient[i] + context.precision[i] * hBeta[context.F[i]];
	}
	if (!s.allFinite()) {
		return false;
	}
	Eigen::VectorXd diagonal = Eigen::VectorXd::Ones(n);
	const bool needDiagonal = context.jacobi || context.stopType == KKT;
	if (needDiagonal) {
		std::vector<double> h;
		if (!modelSpecifics.hessianDiagonal(context.F, h)) {
			return false;
		}
		for (size_t i = 0; i < n; ++i) {
			diagonal(i) = std::max(h[i] + context.precision[i], 1e-300);
		}
	}
	if (context.woodburyAllowed && (!context.woodburyBuilt || context.woodburyStale)) {
		context.woodburyBuilt = false;
		context.woodburyStale = false;
		std::vector<double> S2, B;
		if (modelSpecifics.woodburyParts(context.F, woodburyRank, S2, B)) {
			context.woodburyOffered = true;
			const int k = static_cast<int>(B.size() / n);
			Eigen::MatrixXd M = Eigen::Map<Eigen::MatrixXd>(S2.data(), n, n);
			for (size_t i = 0; i < n; ++i) {
				M(i, i) += context.precision[i];
			}
			context.woodburyM.compute(M);
			if (context.woodburyM.info() == Eigen::Success) {
				const Eigen::Map<Eigen::MatrixXd> Bm(B.data(), k, n);
				context.woodburyW = context.woodburyM.solve(Bm.transpose());
				const Eigen::MatrixXd capacitance = Eigen::MatrixXd::Identity(k, k) - Bm * context.woodburyW;
				context.woodburyCapacitance.compute(capacitance);
				context.woodburyBuilt = context.woodburyCapacitance.info() == Eigen::Success;
			}
		}
	}

	// When the engine offers no Woodbury parts (all but counting-process Cox, or one too large for them), the Nystrom
	// preconditioner once this fit's Hessian-vector products reach what it costs and the last forcing solve cost at least
	// half of it (rent, then buy; a solve can save at most what it costs, so cheaper solves would not repay the build in
	// the few Newton iterations left): a fit whose solves stay cheap never pays for it. It is dropped after the direction
	// whose forcing solve cost more per digit of residual reduction than the last solve without it (not within the
	// direction: its certification solve continues the same CG run), or at once with a CG restart when it stops being
	// positive definite in floating point (below)
	if (!context.woodburyOffered && !context.nystromTried && context.lastPlainSolve >= 0) {
		const double price = nystromPrice(n, static_cast<int>(context.unpenalized.size()), context.entries);
		if (convergenceReport.hessianVectorProducts >= price && context.lastPlainSolve >= 0.5 * price) {
			context.nystromTried = true;
			// the engines that keep a row-wise copy of X_F (all but logistic regression) hold it alongside
			const double rowwise = hXI.getModelType() == ModelType::LOGISTIC ? 0.0 : 12.0 * context.entries;
			std::vector<double> h;
			if (modelSpecifics.hessianDiagonal(context.F, h)) {
				buildNystrom(modelSpecifics, context.F, context.precision, context.unpenalized, h, newtonMemoryBudget - rowwise,
				             context.nystrom, &convergenceReport.hessianVectorProducts);
			}
		}
	}
	const bool preconditioned = context.woodburyBuilt || context.nystrom.built;
	convergenceReport.preconditioner = context.woodburyBuilt ? "woodbury" : (context.nystrom.built ? "nystrom" : "none");

	std::vector<double> in(n), out;
	auto product = [&](const Eigen::VectorXd& v, Eigen::VectorXd& Hv) {
		for (size_t i = 0; i < n; ++i) in[i] = v(i);
		if (!modelSpecifics.hessianVector(context.F, in, out)) return false;
		Hv.resize(n);
		for (size_t i = 0; i < n; ++i) Hv(i) = out[i] + context.precision[i] * v(i);
		++convergenceReport.hessianVectorProducts;
		return Hv.allFinite();
	};

	Eigen::VectorXd x = Eigen::VectorXd::Zero(n), r = s, z, p, Hp;
	auto precondition = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
		if (context.woodburyBuilt) { // P^-1 v = M^-1 v + W (I - B W)^-1 W'v
			return context.woodburyM.solve(v) + context.woodburyW * context.woodburyCapacitance.solve(context.woodburyW.transpose() * v);
		}
		if (context.nystrom.built) {
			return context.nystrom.apply(v);
		}
		return context.jacobi ? Eigen::VectorXd(v.cwiseQuotient(diagonal)) : v;
	};
	// z = P^-1 r and rz = r'z; a Nystrom preconditioner that is not positive definite in floating point at r is
	// dropped, and both are computed again without it. True when it was dropped
	auto preconditionResidual = [&](double& rz) {
		z = precondition(r);
		rz = r.dot(z);
		if (!context.nystrom.built || !(r.squaredNorm() > 0.0) || (rz > 0.0 && std::isfinite(rz))) return false;
		context.nystrom = NystromPreconditioner();
		convergenceReport.preconditioner = "none";
		z = precondition(r);
		rz = r.dot(z);
		return true;
	};
	double rz = 0.0;
	preconditionResidual(rz);
	p = z;
	const double sNorm = s.norm();
	// After a restart from x != 0, r'x = 0 no longer holds: 2 s'x - x'Hx = s'x + x'r in recurrence terms
	bool restarted = false;
	auto decrement = [&]() { return restarted ? s.dot(x) + x.dot(r) : s.dot(x); };
	// CG from x until done() or the budget (or no positive curvature along p); false on a product failure
	auto iterate = [&](const std::function<bool()>& done, int budget) {
		if (done()) return true;
		for (int it = 0; it < budget; ++it) {
			if (!product(p, Hp)) return false;
			const double curvature = p.dot(Hp);
			if (!(curvature > 0.0)) { // H is not positive definite along p: keep x
				break;
			}
			const double alpha = rz / curvature;
			x += alpha * p;
			r -= alpha * Hp;
			double next = 0.0;
			if (preconditionResidual(next)) { // restart CG from x without it
				restarted = true;
				p = z;
			} else {
				p = z + (next / rz) * p;
			}
			rz = next;
			if (done()) break;
		}
		return true;
	};
	const int startProducts = convergenceReport.hessianVectorProducts;
	// The plan's forcing term: a superlinear one (|s_k| / |s_0|) halved the outer iterations but doubled the Hv
	// per iteration on the wide MIMIC designs, and took longer
	const double forcing = std::min(0.5, std::sqrt(sNorm));
	if (!iterate([&]() { return r.norm() <= forcing * sNorm; }, 250)) {
		return false;
	}
	const int solveProducts = convergenceReport.hessianVectorProducts - startProducts;
	context.woodburyStale = context.woodburyBuilt && solveProducts > 25;
	// cost per digit of residual reduction: the forcing tolerance tightens from one Newton iteration to the next
	const double digits = sNorm > 0.0 ? std::max(std::log10(sNorm / std::max(r.norm(), 1e-300 * sNorm)), 0.1) : 0.1;
	bool dropNystrom = false;
	if (!preconditioned) {
		context.lastPlainSolve = solveProducts;
		context.lastPlainPerDigit = solveProducts / digits;
	} else if (context.nystrom.built && !context.nystromJudged) {
		context.nystromJudged = true;
		const double cost = solveProducts * (1.0 + nystromApplyCost(n, static_cast<int>(context.nystrom.Z.cols()), context.entries));
		dropNystrom = cost / digits > context.lastPlainPerDigit; // at the end
	}
	double lambda2 = decrement();
	bool steepest = false;
	if (x.squaredNorm() == 0.0 && sNorm > 0.0) { // no positive curvature at all: a steepest-descent step
		x = s / std::max(diagonal.maxCoeff(), 1.0);
		lambda2 = s.dot(x);
		steepest = true; // x no longer matches the CG residual: no certification from it
	}

	*stop = false;
	*certified = false;
	if (context.stopType == GAP && !steepest && 0.5 * lambda2 <= context.tauEff) {
		const int before = convergenceReport.hessianVectorProducts;
		// The certificate's P and nu_P: Lambda and nuNormal on all-Normal fits, else the Schur bound when there is one
		SchurBound bound(context.unpenalized);
		double nu = context.allNormal ? context.nuNormal : NAN;
		if (!context.allNormal) {
			const auto* data = dynamic_cast<const ModelData<double>*>(&hXI);
			// what a built preconditioner keeps is not available to the bound
			const double retained = (context.woodburyBuilt ? woodburyBytes(static_cast<double>(n)) : 0.0) +
				(context.nystrom.built ? 8.0 * n * (context.nystrom.Z.cols() + 2.0) : 0.0);
			if (data != nullptr && buildSchurBound(modelSpecifics, context.F, K, newtonMemoryBudget - retained, bound,
			                                       &convergenceReport.hessianVectorProducts)) {
				auto positive = [this](int k) {
					return !useCrossValidation || static_cast<size_t>(k) >= hWeights.size() || hWeights[k] > 0.0;
				};
				nu = schurNu(*data, context.F, context.precision, bound, K, positive);
			}
		}
		const bool canCertify = std::isfinite(nu);
		auto upper = [&](double low, const Eigen::VectorXd& residual) { // low + r'P^-1 r >= Lambda^2
			if (context.allNormal) {
				double high = low;
				for (size_t i = 0; i < n; ++i) {
					high += residual(i) * residual(i) / context.precision[i];
				}
				return high;
			}
			return low + bound.quadratic(residual, context.precision);
		};
		// The solve stops once the bracket certifies, preconditioned or not
		bool hit = false, tight = false;
		auto done = [&]() {
			hit = canCertify && newton::gsc(2.0 * nu, std::sqrt(std::max(upper(decrement(), r), 0.0))) <= context.tauEff;
			tight = r.norm() <= 1e-10 * sNorm;
			return hit || tight;
		};
		if (!iterate(done, 1000)) {
			return false;
		}
		convergenceReport.certificationProducts += convergenceReport.hessianVectorProducts - before;
		lambda2 = decrement();
		if (hit || tight) {
			Eigen::VectorXd Hx;
			if (!product(x, Hx)) return false; // explicit, never the recurrence residual
			const double low = 2.0 * s.dot(x) - x.dot(Hx);
			if (canCertify) {
				const double value = newton::gsc(2.0 * nu, std::sqrt(std::max(upper(low, s - Hx), 0.0)));
				convergenceReport.gsc = value;
				*certified = value <= context.tauEff;
			}
			*stop = *certified || 0.5 * low <= context.tauEff;
			lambda2 = low;
		}
	} else if (context.stopType == KKT) {
		*stop = s.cwiseQuotient(diagonal.cwiseSqrt()).cwiseAbs().maxCoeff() < context.tolerance;
	}
	if (dropNystrom) { // it did not pay: the next directions run without it
		context.nystrom = NystromPreconditioner();
		convergenceReport.preconditioner = "none";
	}
	convergenceReport.lambda2Half = 0.5 * lambda2;
	d = -x;
	return d.allFinite();
}

// One prox-Newton iteration for Laplace priors (plan Phase 6, 3.2): the working set W = coefficients without a
// Laplace prior, Laplace nonzeros and violating zeros; the L1-penalized quadratic model over W is minimized by
// cyclic CD, then refined by an exact signed active-set face solve (sign boundaries stop and drop, violators enter,
// a face unbounded along a null direction is followed to a sign boundary); composite Armijo on the penalized
// objective. The stop needs an exact face solve with model decrease -m <= tau_eff and the largest standardized
// residual <= sqrt(2 tau_eff); it is an estimate (uncertified). Phase 2 dropped the separate zero-set condition.
// Dense prox-Newton over the columns W fits the memory budget of dense Newton (2.5): H, its factor and workspace of
// this fit (its threads share them), plus the row-wise copy of X_W for engines that keep one (logistic regression reads its columns)
// R = s' P^-1 s / 2 >= gap (3.1), from the likelihood gradient over F. With every precision p_j > 0 the objective is
// strongly convex with modulus P = diag(p) through the prior alone, so for any subgradient s of it, f(beta) - f* <=
// s' P^-1 s / 2; s is the minimum-norm one, soft-thresholded where an L1 rate applies (elastic net)
double CyclicCoordinateDescent::strongConvexityBound(const NewtonContext& context,
                                                     const std::vector<double>& gradient) const {
	double sum = 0.0;
	for (size_t i = 0; i < context.F.size(); ++i) {
		const double b = hBeta[context.F[i]];
		const double g = gradient[i] + context.precision[i] * b;
		const double l = context.lambda[i];
		const double s = (l == 0.0) ? g : (b != 0.0 ? g + (b > 0.0 ? l : -l) : std::max(std::abs(g) - l, 0.0));
		sum += s * s / context.precision[i];
	}
	return 0.5 * sum;
}

bool CyclicCoordinateDescent::proxFitsDense(const std::vector<int>& W) const {
	const double width = static_cast<double>(W.size());
	double entries = 0.0;
	if (hXI.getModelType() != ModelType::LOGISTIC) {
		for (int j : W) {
			const FormatType format = hXI.getColumnType(j);
			entries += (format == DENSE || format == INTERCEPT) ? static_cast<double>(K) :
				static_cast<double>(hXI.getNumberOfEntries(j));
		}
	}
	return 24.0 * width * width + 12.0 * entries <= newtonMemoryBudget;
}

CyclicCoordinateDescent::NewtonOutcome CyclicCoordinateDescent::proxNewtonIteration(NewtonContext& context) {

	const size_t n = context.F.size();
	std::vector<double> gradient;
	if (!modelSpecifics.fullGradient(context.F, gradient)) {
		return NewtonOutcome::FAILED;
	}
	if (context.allStrong && context.stopType == GAP) { // elastic net: R certifies from the gradient in hand
		convergenceReport.R = strongConvexityBound(context, gradient);
		if (convergenceReport.R <= context.tauEff) {
			return NewtonOutcome::CERTIFIED;
		}
	}
	std::vector<int> W;    // positions in F
	std::vector<int> Wcol; // columns
	for (size_t i = 0; i < n; ++i) {
		const double b = hBeta[context.F[i]];
		const double g = gradient[i] + context.precision[i] * b;
		if (context.lambda[i] == 0.0 || b != 0.0 || std::abs(g) > context.lambda[i]) {
			W.push_back(static_cast<int>(i));
			Wcol.push_back(context.F[i]);
		}
	}
	const size_t w = W.size();
	if (!proxFitsDense(Wcol)) { // CCD when dense prox-Newton over W exceeds the memory budget
		return NewtonOutcome::FAILED;
	}
	if (w == 0) {
		convergenceReport.lambda2Half = 0.0;
		return NewtonOutcome::UNCERTIFIED;
	}
	std::vector<double> block;
	if (!modelSpecifics.hessianBlock(Wcol, block)) {
		return NewtonOutcome::FAILED;
	}
	++convergenceReport.hessians;
	Eigen::MatrixXd H = Eigen::Map<Eigen::MatrixXd>(block.data(), w, w);
	std::vector<double>().swap(block); // the face factor takes its place within the memory budget
	Eigen::VectorXd gw(w), bw(w), lw(w);
	double maxLambda = 1.0;
	for (size_t a = 0; a < w; ++a) {
		const int i = W[a];
		H(a, a) += context.precision[i];
		bw(a) = hBeta[context.F[i]];
		gw(a) = gradient[i] + context.precision[i] * bw(a); // smooth part
		lw(a) = context.lambda[i];
		maxLambda = std::max(maxLambda, lw(a));
	}
	if (!gw.allFinite() || !H.allFinite()) {
		return NewtonOutcome::FAILED;
	}
	const double kktTol = 1e-10 * maxLambda;
	auto subgradient = [&](const Eigen::VectorXd& z, const Eigen::VectorXd& gm, int a) {
		if (lw(a) == 0.0) return gm(a);
		if (z(a) != 0.0) return gm(a) + (z(a) > 0.0 ? lw(a) : -lw(a));
		return std::abs(gm(a)) > lw(a) ? (gm(a) > 0.0 ? gm(a) - lw(a) : gm(a) + lw(a)) : 0.0;
	};
	auto model = [&](const Eigen::VectorXd& z) {
		const Eigen::VectorXd d = z - bw;
		return gw.dot(d) + 0.5 * d.dot(H * d) + (lw.array() * (z.array().abs() - bw.array().abs())).sum();
	};

	// kkt at the current point: |s_j| / sqrt(H_jj); coefficients outside W have s = 0
	double maxR = 0.0;
	{
		const Eigen::VectorXd gm = gw;
		for (size_t a = 0; a < w; ++a) {
			if (H(a, a) > 0.0) maxR = std::max(maxR, std::abs(subgradient(bw, gm, static_cast<int>(a))) / std::sqrt(H(a, a)));
		}
	}

	// Inner cyclic CD on the model, a seed for the face solve below: it stops at 0.01 max r, once its support and
	// signs have held for 5 sweeps with moves below 0.1 max r (the face solve then finishes the model exactly; it ran
	// 400 sweeps past a settled support; signs alone can hold while mass still moves between collinear columns), or
	// after 1,000 sweeps
	Eigen::VectorXd z = bw;
	Eigen::VectorXd q = Eigen::VectorXd::Zero(w); // H (z - b)
	int settled = 0;
	for (int sweep = 0; sweep < 1000; ++sweep) {
		double largest = 0.0;
		bool supportChanged = false;
		for (size_t a = 0; a < w; ++a) {
			const double h = H(a, a);
			if (!(h > 0.0)) continue;
			const double gm = gw(a) + q(a);
			double target;
			if (lw(a) == 0.0) {
				target = z(a) - gm / h;
			} else { // argmin_v h/2 (v - z)^2 + gm (v - z) + lambda |v|
				const double u = z(a) - gm / h;
				target = (std::abs(u) <= lw(a) / h) ? 0.0 : (u > 0.0 ? u - lw(a) / h : u + lw(a) / h);
			}
			const double delta = target - z(a);
			if (delta != 0.0) {
				supportChanged = supportChanged || (z(a) > 0.0) != (target > 0.0) || (z(a) < 0.0) != (target < 0.0);
				z(a) = target;
				q += delta * H.col(a);
				largest = std::max(largest, std::abs(delta) * std::sqrt(h));
			}
		}
		if (largest <= std::max(0.01 * maxR, 1e-12)) break;
		settled = supportChanged ? 0 : settled + 1;
		if (settled >= 5 && largest <= 0.1 * maxR) break; // a settled support whose values have nearly settled too
	}

	// Exact signed active-set face solve, from the inner support and signs. The Cholesky factor of the face matrix is
	// kept across rounds (newton::FaceFactor): an entering coordinate costs a triangular solve and a leaving one a
	// rank-one update, O(m^2) each, not a new O(m^3) factorization. A round without the factor (the first, or a face it
	// cannot hold) uses SymmetricSolve, which handles singular faces. Every violator enters at once
	bool exact = false;
	{
		Eigen::VectorXd zf = z;
		std::vector<int> sign(w, 0);
		std::vector<char> inFace(w, 0);
		std::vector<int> S;
		for (size_t a = 0; a < w; ++a) {
			if (lw(a) == 0.0 || zf(a) != 0.0) {
				inFace[a] = 1;
				sign[a] = (lw(a) == 0.0) ? 0 : (zf(a) > 0.0 ? 1 : -1);
			}
		}
		newton::FaceFactor factor(static_cast<Eigen::Index>(w));
		bool factored = false; // the first round's SymmetricSolve seats the factor when the face is conditioned
		Eigen::VectorXd gm = gw + H * (zf - bw);
		Eigen::VectorXd dz = Eigen::VectorXd::Zero(w);
		double lastResidual = std::numeric_limits<double>::infinity();
		// Rounds on the kept factor cost O(m^2), and a solve can need a few per coordinate of W (all violators enter at
		// once, then leave as they block); a round without it refactorizes, O(m^3), and keeps the budget of 200
		const int maxRounds = 3 * std::max(200, static_cast<int>(w));
		int fallbackRounds = 0;
		for (int iteration = 0; iteration < maxRounds; ++iteration) {
			if (!factored && ++fallbackRounds > 200) break;
			++convergenceReport.faceRounds;
			S.clear();
			for (size_t a = 0; a < w; ++a) if (inFace[a]) S.push_back(static_cast<int>(a));
			const int m = static_cast<int>(S.size());
			dz.setZero();
			bool ray = false; // dz is a null direction of a singular face, not a Newton step
			if (factored) {
				const std::vector<int>& members = factor.members();
				Eigen::VectorXd rF(m);
				for (int k = 0; k < m; ++k) rF(k) = -(gm(members[k]) + lw(members[k]) * sign[members[k]]);
				const Eigen::VectorXd xF = factor.solve(rF);
				for (int k = 0; k < m; ++k) dz(members[k]) = xF(k);
			} else {
				Eigen::MatrixXd HS(m, m);
				Eigen::VectorXd rS(m);
				for (int a = 0; a < m; ++a) {
					rS(a) = -(gm(S[a]) + lw(S[a]) * sign[S[a]]);
					for (int b = 0; b < m; ++b) HS(a, b) = H(S[a], S[b]);
				}
				const newton::SymmetricSolve solve(HS, rS);
				if (solve.method == newton::SolveMethod::RANK_DEFICIENT) ++convergenceReport.rankDeficientFaceRounds;
				if (solve.method == newton::SolveMethod::EIGEN) ++convergenceReport.eigenFaceRounds;
				if (solve.method == newton::SolveMethod::CHOLESKY) {
					factor.seat(solve.cholesky, S); // the face is conditioned: later rounds update its factor
					factored = true;
				}
				Eigen::VectorXd rN = Eigen::VectorXd::Zero(m);
				if (solve.dropped > 0) {
					rN = solve.nullBasis * (solve.nullBasis.transpose() * rS);
				}
				// the face is unbounded below along +rN: follow it to the first sign boundary (there is no full step)
				ray = m > 0 && rN.norm() > std::max(1e-8 * rS.norm(), kktTol);
				for (int a = 0; a < m; ++a) dz(S[a]) = ray ? rN(a) : solve.x(a);
			}
			// stop at the first sign boundary and drop every coordinate that reaches it at that step (several violators that
			// entered together can all be blocked at once, at step 0)
			double step = std::numeric_limits<double>::infinity();
			std::vector<double> stepTo(w, std::numeric_limits<double>::infinity());
			for (int a = 0; a < m; ++a) {
				const int c = S[a];
				if (sign[c] == 0 || dz(c) == 0.0) continue;
				const double zNew = zf(c) + dz(c);
				// a coordinate blocks when the step moves it to or past 0 (one at 0 that does not move stays feasible), or
				// when the ray points to 0
				if (ray ? (dz(c) > 0.0) != (sign[c] > 0) : (zNew > 0.0 ? 1 : (zNew < 0.0 ? -1 : 0)) != sign[c]) {
					stepTo[c] = ray ? -zf(c) / dz(c) : zf(c) / (zf(c) - zNew);
					step = std::min(step, stepTo[c]);
				}
			}
			if (ray && !std::isfinite(step)) break; // unbounded: not exact
			if (std::isfinite(step)) {
				for (int a = 0; a < m; ++a) zf(S[a]) += step * dz(S[a]);
				for (int a = 0; a < m; ++a) {
					const int c = S[a];
					const bool crossed = zf(c) * sign[c] < 0.0; // moved past 0 by rounding (one at 0 stays: it may have just entered)
					if (stepTo[c] <= step || crossed) {
						zf(c) = 0.0;
						sign[c] = 0;
						inFace[c] = 0;
					}
				}
				// the factor holds the face: its columns that left go, from the last down (earlier ones keep their index)
				for (size_t k = factor.members().size(); factored && k-- > 0;) {
					if (!inFace[factor.members()[k]]) factored = factor.remove(k);
				}
				gm = gw + H * (zf - bw);
				continue;
			}
			for (int a = 0; a < m; ++a) zf(S[a]) += dz(S[a]);
			gm = gw + H * (zf - bw);
			double residual = 0.0;
			std::vector<int> violators;
			for (size_t a = 0; a < w; ++a) {
				residual = std::max(residual, std::abs(subgradient(zf, gm, static_cast<int>(a))));
				if (!inFace[a] && lw(a) > 0.0 && std::abs(gm(a)) - lw(a) > kktTol) {
					violators.push_back(static_cast<int>(a));
				}
			}
			if (residual <= kktTol) {
				exact = true;
				z = zf;
				break;
			}
			if (!violators.empty()) { // they enter with the signs that decrease the model
				for (int a : violators) {
					inFace[a] = 1;
					sign[a] = gm(a) > 0.0 ? -1 : 1;
					if (factored && !factor.add(H, a)) {
						factored = false; // the rest of this face solve by SymmetricSolve
					}
				}
				continue;
			}
			if (residual >= 0.5 * lastResidual) break; // refinement stalled
			lastResidual = residual;
		}
		if (!exact && model(zf) < model(z)) {
			z = zf; // the better of the two when the face solve is not exact
		}
	}

	const double decrease = -model(z);
	convergenceReport.lambda2Half = decrease;
	bool stop = false;
	if (context.stopType == GAP) {
		stop = exact && decrease <= context.tauEff && 0.5 * maxR * maxR <= context.tauEff;
	} else if (context.stopType == KKT) {
		stop = maxR < context.tolerance;
	}

	// composite Armijo on the penalized objective over W, inside the eta trust region. Unpenalized coordinates move
	// at most hDelta_j, as in a damped Newton step: where the rows saturate (logistic p of 0 or 1) the curvature
	// vanishes and the model's ray along them is unbounded, which the eta region alone cannot see
	Eigen::VectorXd d = z - bw;
	if (d.cwiseAbs().maxCoeff() == 0.0) {
		return stop ? NewtonOutcome::UNCERTIFIED : NewtonOutcome::FAILED;
	}
	double scale = 1.0;
	for (size_t a = 0; a < w; ++a) {
		const int j = Wcol[a];
		if (lw(a) == 0.0 && context.precision[W[a]] == 0.0 && std::abs(d(a)) > hDelta[j]) {
			scale = std::min(scale, hDelta[j] / std::abs(d(a)));
		}
	}
	if (scale < 1.0) {
		d *= scale;
		z = bw + d;
	}
	const std::vector<double> direction(d.data(), d.data() + w);
	if (!modelSpecifics.setDirection(Wcol, direction)) {
		return NewtonOutcome::FAILED;
	}
	auto objective = [&](double t) {
		double penalty = 0.0;
		for (size_t a = 0; a < w; ++a) {
			const double b = bw(a) + t * d(a);
			penalty += 0.5 * context.precision[W[a]] * b * b + lw(a) * std::abs(b);
		}
		return -modelSpecifics.trialLogLikelihood(t) + penalty;
	};
	const double Delta = gw.dot(d) + (lw.array() * (z.array().abs() - bw.array().abs())).sum();
	const double largest = modelSpecifics.directionMaxAbs();
	const double tStart = (largest > context.eta) ? context.eta / largest : 1.0;
	const double f0 = objective(0.0);
	const double noise = 100.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(f0));
	double t = tStart;
	bool accepted = false;
	for (int halving = 0; halving < 50 && !accepted; ++halving) {
		const double ft = objective(t);
		accepted = std::isfinite(ft) && ((-Delta > noise) ? ft <= f0 + 1e-4 * t * Delta : ft <= f0 + noise);
		if (!accepted) t *= 0.5;
	}
	if (!accepted) {
		context.eta *= 0.5;
		if (stop) return NewtonOutcome::UNCERTIFIED;
		return (++context.rejections >= 2) ? NewtonOutcome::FAILED : NewtonOutcome::REJECTED;
	}
	context.rejections = 0;
	if (t == 1.0) context.eta *= 2.0; else if (t < tStart) context.eta *= 0.5;
	if (!modelSpecifics.commitStep(t)) {
		return NewtonOutcome::FAILED;
	}
	for (size_t a = 0; a < w; ++a) {
		const int j = Wcol[a];
		const double delta = (t == 1.0) ? (z(a) - bw(a)) : t * d(a);
		hBeta[j] = (t == 1.0) ? z(a) : bw(a) + delta; // exact zeros stay zero
		hDelta[j] = std::max(hDelta[j], 2.0 * std::abs(delta));
	}
	kktConsecutive = 0;
	xBetaKnown = true;
	sufficientStatisticsKnown = true;
	convergenceReport.algorithm = "newton";
	return stop ? NewtonOutcome::UNCERTIFIED : NewtonOutcome::ACCEPTED;
}

CyclicCoordinateDescent::NewtonOutcome CyclicCoordinateDescent::newtonIteration(NewtonContext& context) {

	const size_t n = context.F.size();
	Eigen::VectorXd s, d;
	bool stop = false, certified = false, damped = false;
	if (!(context.conjugateGradient ? newtonDirectionCG(context, s, d, &stop, &certified) :
			newtonDirectionDense(context, s, d, &stop, &certified, &damped))) {
		return NewtonOutcome::FAILED;
	}

	// None coordinates get |delta_j| <= hDelta_j when the solve dropped or damped directions: the eta trust region
	// cannot see directions in the null space of X (2.6)
	if (damped) {
		double scale = 1.0;
		for (size_t i = 0; i < n; ++i) {
			const int j = context.F[i];
			if (context.precision[i] == 0.0 && std::abs(d(i)) > hDelta[j]) {
				scale = std::min(scale, hDelta[j] / std::abs(d(i)));
			}
		}
		d *= scale;
	}

	// Armijo backtracking on the penalized objective, starting inside the eta trust region
	const std::vector<double> direction(d.data(), d.data() + n);
	if (!modelSpecifics.setDirection(context.F, direction)) {
		return NewtonOutcome::FAILED;
	}
	auto objective = [&](double t) {
		double penalty = 0.0;
		for (size_t i = 0; i < n; ++i) {
			const double b = hBeta[context.F[i]] + t * d(i);
			penalty += 0.5 * context.precision[i] * b * b;
		}
		return -modelSpecifics.trialLogLikelihood(t) + penalty;
	};
	const double largest = modelSpecifics.directionMaxAbs();
	const double tStart = (largest > context.eta) ? context.eta / largest : 1.0;
	const double f0 = objective(0.0);
	const double slope = s.dot(d);
	const double floor = 100.0 * std::numeric_limits<double>::epsilon() * std::abs(f0);
	double t = tStart;
	double ft = NAN;
	bool accepted = false;
	bool armijo = false;
	for (int halving = 0; halving < 40 && !accepted; ++halving) {
		ft = objective(t);
		if (std::isfinite(ft)) {
			armijo = ft <= f0 + 1e-4 * t * slope;
			// below the floor f cannot see the decrease: accept the first trial, without a certificate's carry-over
			accepted = armijo || (t == tStart && ft - f0 <= floor);
		}
		if (!accepted) {
			t *= 0.5;
		}
	}
	if (!accepted) {
		context.eta *= 0.5;
		if (stop) { // the certificate refers to the current point
			return certified ? NewtonOutcome::CERTIFIED : NewtonOutcome::UNCERTIFIED;
		}
		return (++context.rejections >= 2) ? NewtonOutcome::FAILED : NewtonOutcome::REJECTED;
	}
	context.rejections = 0;
	if (t == 1.0) {
		context.eta *= 2.0;
	} else if (t < tStart) {
		context.eta *= 0.5;
	}

	// Commit (2.9): the engine moves to beta + t d; CCD's per-coordinate bookkeeping follows
	if (!modelSpecifics.commitStep(t)) {
		return NewtonOutcome::FAILED;
	}
	for (size_t i = 0; i < n; ++i) {
		const int j = context.F[i];
		const double delta = t * d(i);
		hBeta[j] += delta;
		hDelta[j] = std::max(hDelta[j], 2.0 * std::abs(delta));
	}
	kktConsecutive = 0;
	xBetaKnown = true;
	sufficientStatisticsKnown = true;
	if (convergenceReport.algorithm != "newton") {
		convergenceReport.algorithm = "newton";
	}

	if (context.stopType == GAP || context.stopType == KKT) {
		if (stop && !certified) {
			return NewtonOutcome::UNCERTIFIED;
		}
		if (stop) {
			if (!armijo) { // bound plus floor
				convergenceReport.gsc += floor;
			}
			return NewtonOutcome::CERTIFIED;
		}
		return NewtonOutcome::ACCEPTED;
	}
	if (context.stopType == ONE_STEP) {
		return NewtonOutcome::UNCERTIFIED;
	}
	// gradient, lange, mittal: relative change of the objective between Newton iterations
	const double change = computeConvergenceCriterion(-ft, -f0);
	return (change < context.tolerance) ? NewtonOutcome::UNCERTIFIED : NewtonOutcome::ACCEPTED;
}

void CyclicCoordinateDescent::findModeGap(const std::function<void()>& cycle, int& iteration, int maxIterations,
                                          double tau, AlgorithmType algorithmType, int convergenceType) {

	NewtonContext context;
	context.stopType = convergenceType;
	context.tolerance = tau;
	for (int j = 0; j < J; ++j) {
		double precision = 0.0;
		double lambda = 0.0;
		const bool separated = divergentDirection.size() == static_cast<size_t>(J) && divergentDirection[j] != 0;
		if (fixBeta[j] || separated) { // separated coefficients (2.7) stay with CCD; Newton works conditionally on them
			continue;
		}
		if (jointPrior->getNewtonPenalty(j, &precision, &lambda)) {
			if (precision == 0.0) {
				context.unpenalized.push_back(static_cast<int>(context.F.size()));
			}
			context.F.push_back(j);
			context.precision.push_back(precision);
			context.lambda.push_back(lambda);
			context.allNormal = context.allNormal && precision > 0.0 && lambda == 0.0;
			context.allStrong = context.allStrong && precision > 0.0;
			context.anyLaplace = context.anyLaplace || lambda > 0.0;
		}
	}
	double weight = 1.0; // tau_eff = tau times the mean positive weight (3.3, D3)
	if (useCrossValidation && !hWeights.empty()) {
		double sum = 0.0;
		int count = 0;
		for (double w : hWeights) {
			if (w > 0.0) {
				sum += w;
				++count;
			}
		}
		weight = (count > 0) ? sum / count : 1.0;
	}
	context.tauEff = tau * weight;

	convergenceReport.present = true;
	convergenceReport.algorithm = "ccd";
	convergenceReport.tauEff = (convergenceType == GAP) ? context.tauEff : NAN;
	if (context.F.empty()) {
		lastReturnFlag = SUCCESS;
		return;
	}
	if (context.allNormal) {
		const auto* data = dynamic_cast<const ModelData<double>*>(&hXI);
		std::vector<double> variance(context.F.size());
		for (size_t i = 0; i < context.F.size(); ++i) {
			variance[i] = 1.0 / context.precision[i];
		}
		auto positive = [this](int k) {
			return !useCrossValidation || static_cast<size_t>(k) >= hWeights.size() || hWeights[k] > 0.0;
		};
		context.nuNormal = (data != nullptr) ? std::sqrt(rowMaximum(*data, context.F, variance, 2, K, positive)) : NAN;
	}
	// Logistic rows saturate (p exactly 0 or 1, e.g. from an extreme start): the curvature of an unpenalized column
	// whose rows all saturate vanishes, a Newton or prox-Newton model has an unbounded flat direction there, and its
	// moves (which also reset coordinate descent's step bounds) can send the fit away. Until it clears, coordinate
	// descent's own trust region recovers
	std::vector<int> unpenalized;
	if (hXI.getModelType() == ModelType::LOGISTIC) {
		for (size_t i = 0; i < context.F.size(); ++i) {
			const FormatType format = hXI.getColumnType(context.F[i]);
			if (context.precision[i] == 0.0 && context.lambda[i] == 0.0 &&
					(format == DENSE || format == INTERCEPT || hXI.getNumberOfEntries(context.F[i]) > 0)) {
				unpenalized.push_back(context.F[i]);
			}
		}
	}
	auto saturated = [&]() {
		if (unpenalized.empty()) {
			return false;
		}
		std::vector<double> h;
		if (!modelSpecifics.hessianDiagonal(unpenalized, h)) {
			return false;
		}
		for (double value : h) {
			if (!(value > 0.0)) {
				return true;
			}
		}
		return false;
	};

	// Profile and confint refits (2.9): chord Newton steps on the reference Hessian, at most 8, before the usual
	// driver and its cost estimates
	if (profileActive && profileReference && convergenceType == GAP && algorithmType != AlgorithmType::CCD &&
			!context.anyLaplace && !saturated()) {
		for (int chord = 0; chord < 8 && iteration < maxIterations; ++chord) {
			++iteration;
			++convergenceReport.newtonIterations;
			const NewtonOutcome outcome = chordIteration(context);
			if (outcome == NewtonOutcome::CERTIFIED || outcome == NewtonOutcome::UNCERTIFIED) {
				--iteration; // the stop test does not move
				--convergenceReport.newtonIterations;
				const bool certified = outcome == NewtonOutcome::CERTIFIED;
				lastReturnFlag = SUCCESS;
				convergenceReport.certificate = certified ? "gsc" : "none";
				convergenceReport.certified = certified;
				convergenceReport.gapEstimate = certified ? convergenceReport.gsc : convergenceReport.lambda2Half;
				return;
			}
			if (outcome == NewtonOutcome::FAILED) {
				--iteration;
				--convergenceReport.newtonIterations;
				break;
			}
		}
	}

	// prox-Newton works on W (unpenalized, nonzeros, violators): its cost is estimated from the current W, again
	// after the warm start and every 16 cycles
	auto workingSet = [&]() {
		std::vector<int> W;
		for (size_t i = 0; i < context.F.size(); ++i) {
			if (context.lambda[i] == 0.0 || hBeta[context.F[i]] != 0.0) W.push_back(context.F[i]);
		}
		return W;
	};
	// newtonCostRatio(W) is in cycles over W; a CCD cycle covers all of F
	auto entryCount = [&](const std::vector<int>& columns) {
		double total = 0.0;
		for (int j : columns) {
			const FormatType format = hXI.getColumnType(j);
			total += (format == DENSE || format == INTERCEPT) ? static_cast<double>(K) : static_cast<double>(hXI.getNumberOfEntries(j));
		}
		return total;
	};
	auto proxCost = [&]() {
		const std::vector<int> W = workingSet();
		if (W.empty()) return 0.0;
		if (!proxFitsDense(W)) return std::numeric_limits<double>::infinity(); // prox-Newton is dense over W: CCD
		return modelSpecifics.newtonCostRatio(W) * entryCount(W) / std::max(entryCount(context.F), 1.0);
	};
	double costRatio = context.anyLaplace ? proxCost() : modelSpecifics.newtonCostRatio(context.F);
	convergenceReport.costRatio = costRatio;

	// Memory gate (2.5): H, its factor and workspace of this fit (its threads share them), plus the row-wise copy of X_F. Above
	// the budget dense Newton is off (Newton-CG takes these fits in a later version): "auto" stays on CCD and
	// "newton" stops at setup
	double entries = 0.0;
	for (int j : context.F) {
		const FormatType format = hXI.getColumnType(j);
		entries += (format == DENSE || format == INTERCEPT) ? static_cast<double>(K) :
			static_cast<double>(hXI.getNumberOfEntries(j));
	}
	const double width = static_cast<double>(context.F.size());
	const double bytes = 24.0 * width * width + 12.0 * entries;
	context.entries = entries;
	const double budget = newtonMemoryBudget;

	// Dense or Newton-CG (2.5): CG when the dense factorization would exceed the memory budget, or when its
	// operation count is lower; CYCLOPS_NEWTON_SOLVER = "dense" or "cg" forces one (tests, gates)
	// Woodbury preconditioning when what it keeps fits beside the row-wise copy of X_F
	context.woodburyAllowed = woodburyBytes(width) + 12.0 * entries <= budget;
	if (const char* woodbury = std::getenv("CYCLOPS_NEWTON_WOODBURY")) {
		context.woodburyAllowed = context.woodburyAllowed && std::atoi(woodbury) != 0;
	}
	// (prox-Newton for Laplace priors is dense over W: no CG estimate, which would copy all of X_F row-wise)
	const double cgRatio = context.anyLaplace ? std::numeric_limits<double>::infinity() :
		modelSpecifics.newtonCgCostRatio(context.F, context.woodburyAllowed);
	context.conjugateGradient = !context.anyLaplace && (bytes > budget || cgRatio < costRatio);
	if (const char* forced = std::getenv("CYCLOPS_NEWTON_SOLVER")) {
		const std::string solver(forced);
		if (solver == "dense" && bytes <= budget) context.conjugateGradient = false;
		if (solver == "cg") context.conjugateGradient = true;
	}
	if (const char* jacobi = std::getenv("CYCLOPS_NEWTON_JACOBI")) {
		context.jacobi = std::atoi(jacobi) != 0;
	}
	convergenceReport.solver = context.conjugateGradient ? "cg" : "dense";
	double newtonCost = context.conjugateGradient ? cgRatio : costRatio;
	convergenceReport.costRatio = newtonCost;

	const int warmStart = context.anyLaplace ? 10 : 3; // 2.4 step 2: 10 when Laplace coordinates screen
	const int stopFrom = 3; // stop tests start after 3 cycles even when Newton waits for the longer warm start
	bool newtonPhase = false;
	int phaseIterations = 0;
	int cycles = 0;
	int failures = 0;
	int suppressUntil = 0;
	int certifyAfter = 0;
	int certifyFailures = 0;
	double R = NAN;
	double gHat = NAN;
	std::vector<double> decreases;
	lastReturnFlag = MAX_ITERATIONS;

	auto finish = [this](UpdateReturnFlags flag, const std::string& certificate, double estimate) {
		lastReturnFlag = flag;
		convergenceReport.certificate = certificate;
		convergenceReport.certified = certificate != "none";
		convergenceReport.gapEstimate = estimate;
	};

	while (true) {
		logger->yield();
		if (iteration >= maxIterations) {
			finish(MAX_ITERATIONS, "none", context.allNormal ? R :
				(std::isfinite(convergenceReport.lambda2Half) ? convergenceReport.lambda2Half : gHat));
			break;
		}

		if (newtonPhase) {
			++iteration;
			++phaseIterations;
			++convergenceReport.newtonIterations;
			const NewtonOutcome outcome = context.anyLaplace ? proxNewtonIteration(context) : newtonIteration(context);
			if (outcome == NewtonOutcome::CERTIFIED) {
				if (context.anyLaplace) {
					finish(SUCCESS, "R", convergenceReport.R);
				} else {
					finish(SUCCESS, "gsc", convergenceReport.gsc);
				}
				break;
			}
			if (outcome == NewtonOutcome::UNCERTIFIED) {
				finish(SUCCESS, "none", convergenceReport.lambda2Half);
				break;
			}
			if (outcome == NewtonOutcome::FAILED || phaseIterations >= 50) { // 2.4 step 5: back to CCD
				newtonPhase = false;
				++failures;
				++convergenceReport.fallbacks;
				suppressUntil = cycles + (1 << std::min(failures, 20)) * 20;
				// CCD resumes as from a fresh start at this point: Newton's moves widened its step bounds (hDelta grows
				// to twice each move), which on a flat or saturated direction would let it run away
				for (int j : context.F) {
					hDelta[j] = initialBound;
				}
				if (noiseLevel > SILENT) {
					std::ostringstream stream;
					stream << "Newton steps stopped; continuing with coordinate descent";
					logger->writeLine(stream);
				}
			}
			continue;
		}

		shadowStepDecrease = 0.0;
		cycle(); // advances iteration
		++cycles;
		// steps cut short by their bounds (far from the optimum, or rows saturating): the gap estimate means nothing yet,
		// so no certification attempt, as coordinate descent's own rule declines convergence. (Newton steps may still
		// start: with thousands of columns some step is bound-limited in most early cycles)
		const bool stepsLimited = boundLimitedStep;
		if (context.anyLaplace && (cycles == stopFrom || cycles == warmStart || cycles % 16 == 0)) {
			newtonCost = proxCost();
			convergenceReport.costRatio = newtonCost;
		}
		convergenceReport.cycles = cycles;
		decreases.push_back(shadowStepDecrease);

		// Trigger statistics (2.4 step 4): the rate of the last min(t, 20) step decreases
		double rho = NAN;
		const int m = std::min<int>(static_cast<int>(decreases.size()), 20);
		if (m >= 2) {
			bool positive = true;
			double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
			for (int i = 0; i < m && positive; ++i) {
				const double value = decreases[decreases.size() - m + i];
				positive = value > 0.0;
				const double y = positive ? std::log(value) : 0.0;
				sx += i; sy += y; sxx += static_cast<double>(i) * i; sxy += i * y;
			}
			if (positive) {
				rho = std::exp((m * sxy - sx * sy) / (m * sxx - sx * sx));
			}
		}
		const double decrease = decreases.back();
		gHat = (decrease == 0.0) ? 0.0 : ((rho < 1.0) ? decrease * rho / (1.0 - rho) : std::numeric_limits<double>::infinity());
		const double nHat = (gHat <= context.tauEff) ? 0.0 :
			((rho < 1.0) ? std::log(gHat / context.tauEff) / -std::log(rho) : std::numeric_limits<double>::infinity());

		// Certificate first (2.4 step 3): R when every free coordinate has a positive precision (normal, or elastic net
		// with alpha < 1, where it is an extra certified stop beside the prox-Newton estimate). Its gradient pass is skipped while
		// the step-decrease estimate is still far above tau_eff (R >= gap, so no stop is missed; the
		// estimate reads at most a few times low, E16); every 16th cycle computes it regardless, since the estimate
		// reads far too high where the rate nears 1
		const bool farAway = std::isfinite(gHat) && gHat > 1e3 * context.tauEff && cycles > 1 && cycles % 16 != 0;
		if (context.allStrong && !farAway) {
			std::vector<double> gradient;
			if (modelSpecifics.fullGradient(context.F, gradient)) {
				R = strongConvexityBound(context, gradient);
				convergenceReport.R = R;
				if (!std::isfinite(R)) {
					finish(ILLCONDITIONED, "none", NAN);
					break;
				}
				if (convergenceType == GAP && R <= context.tauEff) {
					finish(SUCCESS, "R", R);
					break;
				}
			}
		}

		// (after a fallback, "newton" waits for suppressUntil as "auto" does, and the CCD-path stops below run meanwhile)
		if (algorithmType == AlgorithmType::NEWTON && cycles >= warmStart && cycles >= suppressUntil && !saturated()) {
			newtonPhase = true;
		} else if (algorithmType == AlgorithmType::AUTO && (cycles >= warmStart || (cycles >= stopFrom && newtonCost < 1.0)) &&
				cycles >= suppressUntil && nHat > 4.0 * newtonCost && !saturated()) {
			// (Laplace fits screen for 10 cycles unless prox-Newton over the working set costs under one cycle)
			newtonPhase = true;
		}
		if (newtonPhase) {
			phaseIterations = 0;
			context.rejections = 0;
			if (convergenceReport.newtonStart < 0) {
				convergenceReport.newtonStart = cycles;
			}
			continue;
		}

		// A curvature certificate on the CCD path: once the estimate is within 100 tau_eff and finishing on CCD
		// costs more than one H (or it is already below tau_eff), with 2^k cycles between failed attempts
		if (convergenceType == GAP && context.conjugateGradient && !context.allNormal && cycles >= stopFrom &&
				gHat <= context.tauEff && !stepsLimited) { // no curvature certificate within the memory budget: stop on the estimate
			finish(SUCCESS, "none", gHat);
			break;
		}
		if (convergenceType == GAP && !context.conjugateGradient && cycles >= stopFrom && cycles >= certifyAfter &&
				!stepsLimited) {
			const double estimate = context.allNormal ? R : gHat;
			const double remaining = context.allNormal ?
				((rho < 1.0 && R > context.tauEff) ? std::log(R / context.tauEff) / -std::log(rho) :
					std::numeric_limits<double>::infinity()) : nHat;
			if (estimate <= 100.0 * context.tauEff && (remaining > newtonCost || estimate <= context.tauEff) && !saturated()) {
				double value = NAN, lambda2Half = NAN;
				Eigen::VectorXd step;
				if (context.anyLaplace && !proxFitsDense(workingSet())) {
					// no prox-Newton over so large a working set: stop on the estimate, as without a curvature
					// certificate under Newton-CG
					if (gHat <= context.tauEff) {
						finish(SUCCESS, "none", gHat);
						break;
					}
				} else if (context.anyLaplace) { // one prox-Newton iteration: its L1 decrease stops the fit, or its step helps
					++iteration;
					++convergenceReport.newtonIterations;
					const NewtonOutcome outcome = proxNewtonIteration(context);
					if (outcome == NewtonOutcome::CERTIFIED) {
						finish(SUCCESS, "R", convergenceReport.R);
						break;
					}
					if (outcome == NewtonOutcome::UNCERTIFIED) {
						finish(SUCCESS, "none", convergenceReport.lambda2Half);
						break;
					}
				} else if (certifyGap(context, &value, &lambda2Half, &step)) {
					convergenceReport.gsc = value;
					convergenceReport.lambda2Half = lambda2Half;
					if (value <= context.tauEff) {
						finishingStep(context, step);
						finish(SUCCESS, "gsc", value);
						break;
					}
				} else if (std::isfinite(lambda2Half)) {
					// H is singular (e.g. columns constant within every stratum of a grouped model): no gsc, and
					// CCD crawls along the flat directions. As on the Newton path, Lambda^2 / 2 <= tau_eff stops
					// uncertified; otherwise "auto" continues with Newton steps
					convergenceReport.lambda2Half = lambda2Half;
					if (lambda2Half <= context.tauEff) {
						finishingStep(context, step);
						finish(SUCCESS, "none", lambda2Half);
						break;
					}
					if (algorithmType == AlgorithmType::AUTO && cycles >= suppressUntil) {
						newtonPhase = true;
						phaseIterations = 0;
						context.rejections = 0;
						if (convergenceReport.newtonStart < 0) {
							convergenceReport.newtonStart = cycles;
						}
						continue;
					}
				}
				certifyAfter = cycles + (1 << std::min(certifyFailures++, 20));
			}
		}
	}
}


// Profile and confint refits (Newton plan 2.9). The reference is the likelihood Hessian H~ at the coefficients the
// profile starts from. A refit takes chord Newton steps d = -(H~ + Lambda)^-1 s on its free columns F (a subset),
// whose first step from the previous solution is the tangent predictor. Its stop allows for the lag: the
// likelihoods here weight rows within a risk set or stratum by exp(eta), and moving every eta by at most delta moves
// each weight by at most exp(2 delta), so H(beta) >= exp(-2 delta) H~ and, with the precision, Lambda^2 <=
// exp(2 delta) Lambda~^2 and nu <= exp(delta) nu~ (the diagonal of H^-1 grows at most by exp(2 delta)).
struct CyclicCoordinateDescent::ProfileReference {
	std::vector<int> F;              // columns of H
	std::vector<int> position;       // column -> position in F (-1 otherwise)
	Eigen::MatrixXd H;               // likelihood Hessian over F at the reference
	std::vector<double> eta;         // x'beta at the reference
	std::vector<double> beta;        // coefficients at the reference
	std::shared_ptr<const std::vector<double>> columnMax; // max_k |x_kj|: sum_j columnMax_j |beta_j - beta~_j| >= delta
};

struct CyclicCoordinateDescent::ProfileFactor {
	const ProfileReference* reference = nullptr;
	std::vector<int> F;
	std::vector<double> precision;
	bool cholesky = false;
	Eigen::LLT<Eigen::MatrixXd> llt;
	Eigen::MatrixXd basis;           // eigen path: eigenvectors with eigenvalues above 1e-10 the largest
	Eigen::VectorXd inverseValues;   // and their inverses
	double nu = NAN;                 // nu~ for the Cholesky path
};

// A reference at the current state over the columns of `like` (or the free Normal / None columns); null when the
// model or the priors do not allow one
std::shared_ptr<const CyclicCoordinateDescent::ProfileReference> CyclicCoordinateDescent::makeProfileReference(
		const ProfileReference* like) {
	auto reference = std::make_shared<ProfileReference>();
	if (like != nullptr) {
		reference->F = like->F;
		reference->position = like->position;
		reference->columnMax = like->columnMax;
	} else {
		for (int j = 0; j < J; ++j) {
			double precision = 0.0, lambda = 0.0;
			if (jointPrior->getLaplaceLambda(j, &lambda)) {
				return nullptr; // prox-Newton refits keep their own path
			}
			if (!fixBeta[j] && jointPrior->getNormalPrecision(j, &precision)) {
				reference->F.push_back(j);
			}
		}
		const double width = static_cast<double>(reference->F.size());
		if (reference->F.empty() || 8.0 * width * width > 64.0 * 1024 * 1024) {
			return nullptr;
		}
		reference->position.assign(J, -1);
		for (size_t i = 0; i < reference->F.size(); ++i) {
			reference->position[reference->F[i]] = static_cast<int>(i);
		}
		auto columnMax = std::make_shared<std::vector<double>>(J, 0.0);
		const auto* data = dynamic_cast<const ModelData<double>*>(&hXI);
		if (data == nullptr) {
			return nullptr;
		}
		for (int j = 0; j < J; ++j) {
			for (GenericIterator<double> it(data->getX(), j); it; ++it) {
				(*columnMax)[j] = std::max((*columnMax)[j], std::abs(it.value()));
			}
		}
		reference->columnMax = columnMax;
	}
	getLogLikelihood(); // brings the state up to date
	std::vector<double> block;
	if (!modelSpecifics.hessianBlock(reference->F, block)) {
		return nullptr;
	}
	const size_t n = reference->F.size();
	reference->H = Eigen::Map<Eigen::MatrixXd>(block.data(), n, n);
	reference->eta = modelSpecifics.getXBeta();
	reference->beta = hBeta;
	return reference;
}

void CyclicCoordinateDescent::beginProfileReference() {
	profileActive = false;
	profileReference.reset();
	profileFactor.reset();
	if (!modelSpecifics.supportsSecondOrder() || usingGPU || usingCUDA || syncCV) {
		return;
	}
	// The mode's reference survives between profile calls on the same fit (the adaptive profile calls in batches),
	// reused while the coefficients there are unchanged
	if (!profilePool.empty()) {
		const std::vector<double>& mode = profilePool[0]->beta;
		bool same = mode.size() == hBeta.size();
		for (size_t j = 0; j < mode.size() && same; ++j) {
			same = std::abs(mode[j] - hBeta[j]) <= 1e-12 * std::max(1.0, std::abs(mode[j]));
		}
		for (int j = 0; j < J && same; ++j) { // and the free columns
			double precision = 0.0;
			const bool free = !fixBeta[j] && jointPrior->getNormalPrecision(j, &precision);
			same = free == (profilePool[0]->position[j] >= 0);
		}
		if (!same) {
			profilePool.clear();
		}
	}
	if (profilePool.empty()) {
		auto reference = makeProfileReference(nullptr);
		if (!reference) {
			return;
		}
		profilePool.push_back(reference);
	}
	profileReference = profilePool[0];
	profileActive = true;
}

void CyclicCoordinateDescent::endProfileReference() {
	profileActive = false;
	profileReference.reset();
	profileFactor.reset();
}

void CyclicCoordinateDescent::shareProfileReference(const CyclicCoordinateDescent& other) {
	profilePool = other.profilePool;
	profileReference = other.profileReference;
	profileActive = other.profileActive;
	profileFactor.reset();
}

std::shared_ptr<const void> CyclicCoordinateDescent::currentProfileReference() const {
	return profileReference;
}

void CyclicCoordinateDescent::useProfileReference(const std::shared_ptr<const void>& reference) {
	if (profileActive && reference) {
		profileReference = std::static_pointer_cast<const ProfileReference>(reference);
	}
}

CyclicCoordinateDescent::NewtonOutcome CyclicCoordinateDescent::chordIteration(NewtonContext& context) {

	const size_t n = context.F.size();

	// The lag delta = max_k |eta_k - eta~_k| from the reference in use (the parent point's: see CcdInterface's profile
	// drivers). Past 0.5 a new reference is taken here, and the point's children inherit it
	std::vector<double> eta = modelSpecifics.getXBeta();
	double delta = 0.0;
	for (size_t k = 0; k < eta.size() && k < profileReference->eta.size(); ++k) {
		delta = std::max(delta, std::abs(eta[k] - profileReference->eta[k]));
	}
	if (!(delta <= 0.5)) {
		auto moved = makeProfileReference(profileReference.get());
		if (!moved) {
			return NewtonOutcome::FAILED;
		}
		++convergenceReport.hessians;
		profileReference = moved;
		delta = 0.0;
	}
	const ProfileReference& reference = *profileReference;
	if (!profileFactor || profileFactor->reference != &reference || profileFactor->F != context.F ||
			profileFactor->precision != context.precision) {
		auto factor = std::make_shared<ProfileFactor>();
		std::vector<int> index(n);
		for (size_t i = 0; i < n; ++i) {
			index[i] = reference.position[context.F[i]];
			if (index[i] < 0) {
				return NewtonOutcome::FAILED; // a column outside the reference
			}
		}
		Eigen::MatrixXd H(n, n);
		for (size_t b = 0; b < n; ++b) {
			for (size_t a = 0; a < n; ++a) {
				H(a, b) = reference.H(index[a], index[b]);
			}
			H(b, b) += context.precision[b];
		}
		factor->llt.compute(H);
		if (factor->llt.info() == Eigen::Success) {
			const Eigen::VectorXd d = factor->llt.matrixLLT().diagonal().array().square();
			factor->cholesky = d.size() == 0 || d.minCoeff() >= 1e-10 * d.maxCoeff();
		}
		if (factor->cholesky) {
			const Eigen::MatrixXd Linverse = factor->llt.matrixL().solve(Eigen::MatrixXd::Identity(n, n));
			const Eigen::VectorXd inverse = Linverse.colwise().squaredNorm().transpose();
			std::vector<double> root(n);
			for (size_t i = 0; i < n; ++i) {
				root[i] = std::sqrt(std::max(inverse(i), 0.0));
			}
			const auto* data = dynamic_cast<const ModelData<double>*>(&hXI);
			auto positive = [this](int k) {
				return !useCrossValidation || static_cast<size_t>(k) >= hWeights.size() || hWeights[k] > 0.0;
			};
			factor->nu = (data != nullptr) ? rowMaximum(*data, context.F, root, 1, K, positive) : NAN;
			if (context.allNormal) {
				factor->nu = std::min(factor->nu, context.nuNormal);
			}
		} else {
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(H);
			const Eigen::VectorXd values = eigen.eigenvalues();
			const double tolerance = 1e-10 * values.cwiseAbs().maxCoeff();
			int kept = 0;
			for (Eigen::Index i = 0; i < values.size(); ++i) {
				kept += values(i) > tolerance;
			}
			factor->basis.resize(n, kept);
			factor->inverseValues.resize(kept);
			for (Eigen::Index i = 0, c = 0; i < values.size(); ++i) {
				if (values(i) > tolerance) {
					factor->basis.col(c) = eigen.eigenvectors().col(i);
					factor->inverseValues(c++) = 1.0 / values(i);
				}
			}
		}
		factor->reference = &reference;
		factor->F = context.F;
		factor->precision = context.precision;
		profileFactor = factor;
	}
	const ProfileFactor& factor = *profileFactor;

	std::vector<double> gradient;
	if (!modelSpecifics.fullGradient(context.F, gradient)) {
		return NewtonOutcome::FAILED;
	}
	Eigen::VectorXd s(n);
	for (size_t i = 0; i < n; ++i) {
		s(i) = gradient[i] + context.precision[i] * hBeta[context.F[i]];
	}
	if (!s.allFinite()) {
		return NewtonOutcome::FAILED;
	}
	const Eigen::VectorXd x = factor.cholesky ? Eigen::VectorXd(factor.llt.solve(s)) :
		Eigen::VectorXd(factor.basis * (factor.inverseValues.asDiagonal() * (factor.basis.transpose() * s)));
	const double lambda2 = std::exp(2.0 * delta) * s.dot(x);
	convergenceReport.lambda2Half = 0.5 * lambda2;

	// Stop at the current point: gsc with the lagged bounds certifies it; a singular H~ stops uncertified on
	// Lambda^2 / 2, as the Newton path does
	if (factor.cholesky) {
		const double value = newton::gsc(2.0 * std::exp(delta) * factor.nu, std::sqrt(lambda2));
		convergenceReport.gsc = value;
		if (value <= context.tauEff) {
			return NewtonOutcome::CERTIFIED;
		}
	} else if (0.5 * lambda2 <= context.tauEff) {
		return NewtonOutcome::UNCERTIFIED;
	}

	// Chord step with Armijo backtracking inside the eta trust region
	const Eigen::VectorXd d = -x;
	const std::vector<double> direction(d.data(), d.data() + n);
	if (!modelSpecifics.setDirection(context.F, direction)) {
		return NewtonOutcome::FAILED;
	}
	auto objective = [&](double t) {
		double penalty = 0.0;
		for (size_t i = 0; i < n; ++i) {
			const double b = hBeta[context.F[i]] + t * d(i);
			penalty += 0.5 * context.precision[i] * b * b;
		}
		return -modelSpecifics.trialLogLikelihood(t) + penalty;
	};
	const double largest = modelSpecifics.directionMaxAbs();
	const double tStart = (largest > context.eta) ? context.eta / largest : 1.0;
	const double f0 = objective(0.0);
	const double slope = s.dot(d);
	double t = tStart;
	bool accepted = false;
	for (int halving = 0; halving < 30 && !accepted; ++halving) {
		const double ft = objective(t);
		accepted = std::isfinite(ft) && ft <= f0 + 1e-4 * t * slope;
		if (!accepted) {
			t *= 0.5;
		}
	}
	if (!accepted || !modelSpecifics.commitStep(t)) {
		return NewtonOutcome::FAILED;
	}
	for (size_t i = 0; i < n; ++i) {
		const int j = context.F[i];
		const double change = t * d(i);
		hBeta[j] += change;
		hDelta[j] = std::max(hDelta[j], 2.0 * std::abs(change));
	}
	kktConsecutive = 0;
	xBetaKnown = true;
	sufficientStatisticsKnown = true;
	convergenceReport.algorithm = "newton";
	return NewtonOutcome::ACCEPTED;
}

} // namespace bsccs
