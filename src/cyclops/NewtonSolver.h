/*
 * NewtonSolver.h
 *
 * Numerics shared by the certified "gap" stops and the dense Newton moves of CyclicCoordinateDescent::findMode
 * (Newton plan sections 2.4-2.6 and 3; the moves themselves are CyclicCoordinateDescent members, defined in
 * NewtonSolver.cpp).
 */

#ifndef NEWTONSOLVER_H_
#define NEWTONSOLVER_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace bsccs {
namespace newton {

// omega(z) / c^2 with omega(z) = z + (1 - z) log(1 - z): the self-concordant gap bound at z = c Lambda < 1, else Inf
inline double gsc(double c, double lambda) {
	const double z = c * lambda;
	if (!(z < 1.0)) {
		return std::numeric_limits<double>::infinity();
	}
	return (z + (1.0 - z) * std::log1p(-z)) / (c * c);
}

enum class SolveMethod { CHOLESKY, LDLT, RANK_DEFICIENT, EIGEN };

// The conditioning test of a Cholesky factor L L' = A, from its squared pivots d: min d >= 1e-10 max d, and every pivot a
// real share of its diagonal, d_k > 1e-10 A_kk (the part of column k not explained by the columns before it; a scaled
// copy of an earlier column leaves rounding noise there that the first test can accept)
inline bool choleskyConditioned(const Eigen::VectorXd& d, const Eigen::VectorXd& diagonal) {
	return d.size() == 0 || (d.allFinite() && d.minCoeff() >= 1e-10 * d.maxCoeff() &&
	                         (d.array() > 1e-10 * diagonal.array()).all());
}

// x = A^-1 b for symmetric A. Cholesky only when it passes choleskyConditioned(). Otherwise a positive
// semidefinite A of lower rank gets A^+ b, the minimum-norm solve, from Cholesky with complete pivoting, and an orthonormal
// basis of the null space it drops (Eigen reports success on collinear designs: plan 2.6, E17). Any other A gets pivoted
// LDLT when min |D_ii| >= 1e-10 max |D_ii|, else the same A^+ b from an eigendecomposition. The Cholesky factor is kept for the inverse
// diagonal of the gsc certificate.
struct SymmetricSolve {
	Eigen::VectorXd x;
	SolveMethod method = SolveMethod::EIGEN;
	int dropped = 0;
	Eigen::MatrixXd nullBasis;          // when dropped > 0: the dropped (orthonormal) directions
	Eigen::LLT<Eigen::MatrixXd> cholesky;
	Eigen::MatrixXd semidefiniteFactor; // lower triangle of top-left (pivotRank x pivotRank) holds L11 (upper triangle unused)
	Eigen::VectorXi pivotOrder;         // permutation of solveSemidefinite(): leading pivotRank entries form the basis
	Eigen::Index pivotRank = 0;
	bool semidefiniteWellScaled = false; // max |-L11^-T L21'| <= 1e4 (safe to seat in FaceFactor::solveWithDependent)

	SymmetricSolve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
		cholesky.compute(A);
		if (cholesky.info() == Eigen::Success) {
			const Eigen::VectorXd d = cholesky.matrixLLT().diagonal().array().square();
			if (choleskyConditioned(d, A.diagonal())) {
				method = SolveMethod::CHOLESKY;
				x = cholesky.solve(b);
				return;
			}
		}
		cholesky = Eigen::LLT<Eigen::MatrixXd>(); // no certificate from here on: release the factor
		if (solveSemidefinite(A, b)) {
			return;
		}
		Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
		if (ldlt.info() == Eigen::Success) {
			const Eigen::VectorXd d = ldlt.vectorD().cwiseAbs();
			if (d.minCoeff() >= 1e-10 * d.maxCoeff()) {
				method = SolveMethod::LDLT;
				x = ldlt.solve(b);
				return;
			}
		}
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(A);
		const Eigen::VectorXd values = eigen.eigenvalues();
		const double tolerance = 1e-10 * values.cwiseAbs().maxCoeff();
		Eigen::VectorXd projected = eigen.eigenvectors().transpose() * b;
		for (Eigen::Index i = 0; i < values.size(); ++i) {
			if (values(i) > tolerance) {
				projected(i) /= values(i);
			} else {
				projected(i) = 0.0;
				++dropped;
			}
		}
		nullBasis.resize(values.size(), dropped);
		for (Eigen::Index i = 0, c = 0; i < values.size(); ++i) {
			if (!(values(i) > tolerance)) {
				nullBasis.col(c++) = eigen.eigenvectors().col(i);
			}
		}
		method = SolveMethod::EIGEN;
		x = eigen.eigenvectors() * projected;
	}

private:

	// Blocked Cholesky with complete pivoting (LAPACK's dpstrf). M holds L in its leading lower-triangular columns
	// and the Schur complement of the pivots so far in its trailing lower-triangular block; each step takes the
	// largest diagonal of that complement, and each panel of nb columns updates the trailing block with one BLAS-3
	// symmetric rank-k update. A has rank-deficiency d when the trailing d x d complement is negligible.
	// P A P' = [L11; L21] [L11' L21'] gives the null space, P' [-L11^-T L21'; I], and A^+ b.
	// false when A has full rank at the threshold or is not positive semidefinite
	bool solveSemidefinite(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
		const Eigen::Index n = A.rows();
		if (n == 0) {
			return false;
		}
		Eigen::MatrixXd M = A;
		Eigen::VectorXi order = Eigen::VectorXi::LinSpaced(n, 0, static_cast<int>(n) - 1);
		if (!A.diagonal().allFinite()) {
			return false;
		}
		const double threshold = 1e-10 * A.diagonal().cwiseAbs().maxCoeff();
		if (!std::isfinite(threshold)) {
			return false;
		}
		const Eigen::Index nb = 64;
		Eigen::VectorXd diag(n);
		Eigen::Index rank = 0;
		bool done = false;
		for (Eigen::Index b0 = 0; b0 < n && !done; b0 += nb) {
			diag.tail(n - b0) = M.diagonal().tail(n - b0);
			const Eigen::Index bEnd = std::min(n, b0 + nb);
			Eigen::Index k = b0;
			for (; k < bEnd; ++k) {
				if (diag.tail(n - k).cwiseAbs().sum() <= threshold) {
					done = true;
					break;
				}
				Eigen::Index largest = 0;
				const double pivot = diag.tail(n - k).maxCoeff(&largest);
				largest += k;
				if (!(pivot > 0.0)) {
					return false;
				}
				if (largest != k) {
					// Symmetric swap of rows/columns k and largest in the lower triangle (including leading cols 0..k-1)
					if (k > 0) {
						M.row(k).head(k).swap(M.row(largest).head(k));
					}
					std::swap(M(k, k), M(largest, largest));
					for (Eigen::Index i = k + 1; i < largest; ++i) {
						std::swap(M(i, k), M(largest, i));
					}
					if (largest + 1 < n) {
						M.col(k).tail(n - largest - 1).swap(M.col(largest).tail(n - largest - 1));
					}
					std::swap(diag(k), diag(largest));
					std::swap(order(k), order(largest));
				}
				const double root = std::sqrt(pivot);
				M(k, k) = root;
				const Eigen::Index below = n - k - 1;
				if (below > 0) {
					if (k > b0) {
						M.col(k).tail(below).noalias() -=
							M.block(k + 1, b0, below, k - b0) * M.block(k, b0, 1, k - b0).transpose();
					}
					M.col(k).tail(below) /= root;
					diag.tail(below).noalias() -= M.col(k).tail(below).cwiseAbs2();
				}
			}
			rank = k;
			const Eigen::Index bw = k - b0;
			const Eigen::Index rem = n - k;
			if (bw > 0 && rem > 0) {
				M.bottomRightCorner(rem, rem).selfadjointView<Eigen::Lower>().rankUpdate(M.block(k, b0, rem, bw), -1.0);
			}
		}
		const Eigen::Index deficiency = n - rank;
		if (deficiency == 0) {
			return false;
		}
		// Scaled Frobenius norm of the symmetric trailing deficiency x deficiency Schur complement from its lower
		// triangle (avoids underflow/overflow at extreme scales) plus explicit finiteness check on all kept entries
		double tailMax = 0.0;
		bool finite = true;
		for (Eigen::Index j = rank; j < n && finite; ++j) {
			const double d = std::abs(M(j, j));
			finite = std::isfinite(d);
			if (d > tailMax) tailMax = d;
			for (Eigen::Index i = j + 1; i < n && finite; ++i) {
				const double v = std::abs(M(i, j));
				finite = std::isfinite(v);
				if (v > tailMax) tailMax = v;
			}
		}
		double tailNorm = 0.0;
		if (finite && tailMax > 0.0) {
			double scaledSq = 0.0;
			for (Eigen::Index j = rank; j < n; ++j) {
				const double d = M(j, j) / tailMax;
				scaledSq += d * d;
				for (Eigen::Index i = j + 1; i < n; ++i) {
					const double v = M(i, j) / tailMax;
					scaledSq += 2.0 * v * v;
				}
			}
			tailNorm = tailMax * std::sqrt(scaledSq);
			finite = std::isfinite(tailNorm);
		}
		for (Eigen::Index j = 0; j < rank && finite; ++j) {
			finite = M.col(j).tail(n - j).allFinite();
		}
		if (!finite || !(tailNorm <= threshold)) {
			return false;
		}
		Eigen::MatrixXd localNullBasis;
		Eigen::VectorXd localX;
		bool wellScaled = true;
		{
			const auto L11 = M.topLeftCorner(rank, rank).triangularView<Eigen::Lower>();
			Eigen::MatrixXd pivoted(n, deficiency);
			if (rank > 0) {
				pivoted.topRows(rank) = -L11.transpose().solve(M.bottomLeftCorner(deficiency, rank).transpose());
				if (!pivoted.topRows(rank).allFinite()) {
					return false;
				}
				wellScaled = pivoted.topRows(rank).cwiseAbs().maxCoeff() <= 1e4;
			}
			pivoted.bottomRows(deficiency).setIdentity();
			Eigen::MatrixXd basis(n, deficiency);
			for (Eigen::Index i = 0; i < n; ++i) {
				basis.row(order(i)) = pivoted.row(i);
			}
			const Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
			localNullBasis = qr.householderQ() * Eigen::MatrixXd::Identity(n, deficiency);
			if (!localNullBasis.allFinite()) {
				return false;
			}

			const Eigen::VectorXd range = b - localNullBasis * (localNullBasis.transpose() * b);
			Eigen::VectorXd full = Eigen::VectorXd::Zero(n);
			if (rank > 0) {
				Eigen::VectorXd leading(rank);
				for (Eigen::Index i = 0; i < rank; ++i) {
					leading(i) = range(order(i));
				}
				L11.solveInPlace(leading);
				L11.transpose().solveInPlace(leading);
				for (Eigen::Index i = 0; i < rank; ++i) {
					full(order(i)) = leading(i);
				}
			}
			localX = full - localNullBasis * (localNullBasis.transpose() * full);
			if (!localX.allFinite()) {
				return false;
			}
		}
		nullBasis = std::move(localNullBasis);
		x = std::move(localX);
		dropped = static_cast<int>(deficiency);
		pivotRank = rank;
		pivotOrder = std::move(order);
		semidefiniteFactor = std::move(M);
		semidefiniteWellScaled = wellScaled;
		method = SolveMethod::RANK_DEFICIENT;
		return true;
	}

public:
	// diag(A^-1) from the Cholesky factor (only when method == CHOLESKY)
	Eigen::VectorXd inverseDiagonal() const {
		const Eigen::Index n = cholesky.matrixLLT().rows();
		const Eigen::MatrixXd Linverse = cholesky.matrixL().solve(Eigen::MatrixXd::Identity(n, n));
		return Linverse.colwise().squaredNorm().transpose();
	}
};

// Rank-revealing Cholesky factor of a face matrix H_S, kept across the rounds of prox-Newton's face solve:
//   - Independent basis columns B (members()) are factored in the lower triangle of L (m x m, conditioned).
//   - Any linearly dependent columns D (dependent()) are kept alongside B so entering/leaving columns update L in
//     O(m^2) instead of refactorizing O(m^3) from scratch on every round of a singular face.
//   - When D is empty (full-rank faces), solve() is a plain triangular solve; when D is non-empty,
//     solveWithDependent() computes the exact minimum-norm solution and null basis in O(|D| m^2), checking the
//     |D| x |D| Schur complement residual and returning false (falling back to SymmetricSolve) if conditioning degrades.
class FaceFactor {
public:
	explicit FaceFactor(Eigen::Index cap) : capacity(cap) {}

	const std::vector<int>& members() const { return order; }
	const std::vector<int>& dependent() const { return dep; }

	// Release L before calling SymmetricSolve so peak memory stays within proxDenseBytes (24 w^2)
	void clear() {
		L.resize(0, 0);
		m = 0;
		order.clear();
		dep.clear();
	}

	// the factor of H_S that SymmetricSolve computed and accepted (same matrix, same order, same test): no second
	// factorization, and L grows lazily only if later rounds add columns past its current size
	void seat(const Eigen::LLT<Eigen::MatrixXd>& llt, const std::vector<int>& S) {
		m = static_cast<Eigen::Index>(S.size());
		order = S;
		dep.clear();
		if (m > 0) {
			L = llt.matrixL();
		} else {
			L.resize(0, 0);
		}
	}

	// Seat from a rank-revealing solveSemidefinite() result by moving M in place (zero copy): its top-left
	// basis.size() x basis.size() lower triangle already holds L11
	bool seatRankDeficient(Eigen::MatrixXd&& M, const std::vector<int>& basis, const std::vector<int>& dependentCols) {
		const Eigen::Index rank = static_cast<Eigen::Index>(basis.size());
		if (M.rows() < rank || M.cols() < rank) {
			clear();
			return false;
		}
		m = rank;
		order = basis;
		dep = dependentCols;
		L = std::move(M);
		if (!conditioned()) {
			clear();
			return false;
		}
		return true;
	}

	// column a of H enters last; false (and the factor cleared) when it fails the same test
	bool add(const Eigen::MatrixXd& H, int a) {
		Eigen::VectorXd l(m);
		for (Eigen::Index k = 0; k < m; ++k) {
			l(k) = H(order[static_cast<size_t>(k)], a);
		}
		if (m > 0) {
			L.topLeftCorner(m, m).triangularView<Eigen::Lower>().solveInPlace(l);
		}
		// the part of column a that the face does not explain must be a real share of its curvature, whatever the order
		// of entry (a scaled duplicate leaves rounding noise here)
		const double d2 = H(a, a) - l.squaredNorm();
		if (!(d2 > 0.0) || !(d2 > 1e-10 * H(a, a)) || !std::isfinite(d2)) {
			clear();
			return false;
		}
		ensureCapacity(m + 1);
		if (m > 0) {
			L.block(m, 0, 1, m) = l.transpose();
		}
		L(m, m) = std::sqrt(d2);
		order.push_back(a);
		++m;
		if (!conditioned()) {
			clear();
			return false;
		}
		return true;
	}

	// Rank-revealing entry: appends a to basis B when independent and well-scaled relative to B,
	// or to dependent list D when explained by B (or tiny relative to B's largest pivot).
	bool addRankRevealing(const Eigen::MatrixXd& H, int a, size_t maxDependent) {
		Eigen::VectorXd l(m);
		for (Eigen::Index k = 0; k < m; ++k) {
			l(k) = H(order[static_cast<size_t>(k)], a);
		}
		if (m > 0) {
			L.topLeftCorner(m, m).triangularView<Eigen::Lower>().solveInPlace(l);
		}
		const double d2 = H(a, a) - l.squaredNorm();
		if (!std::isfinite(d2)) {
			clear();
			return false;
		}
		const double maxBasisDiag = (m > 0) ? L.topLeftCorner(m, m).diagonal().cwiseAbs2().maxCoeff() : 0.0;
		if (d2 > 0.0 && d2 > 1e-10 * H(a, a) && (m == 0 || d2 >= 1e-10 * maxBasisDiag)) {
			ensureCapacity(m + 1);
			if (m > 0) {
				L.block(m, 0, 1, m) = l.transpose();
			}
			L(m, m) = std::sqrt(d2);
			order.push_back(a);
			++m;
			if (!conditioned()) {
				clear();
				return false;
			}
			return true;
		}
		if (dep.size() >= maxDependent) {
			clear();
			return false;
		}
		dep.push_back(a);
		return true;
	}

	bool removeDependent(int a) {
		for (size_t i = 0; i < dep.size(); ++i) {
			if (dep[i] == a) {
				dep.erase(dep.begin() + static_cast<std::ptrdiff_t>(i));
				return true;
			}
		}
		return false;
	}

	// After one or more basis columns leave, try promoting columns of D that became independent into B
	// in descending order of curvature H(a, a) to keep pivot scaling well-conditioned.
	bool promoteDependent(const Eigen::MatrixXd& H) {
		std::vector<int> candidates = dep;
		std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
			return H(a, a) > H(b, b);
		});
		std::vector<int> remaining;
		remaining.reserve(candidates.size());
		double maxBasisDiag = (m > 0) ? L.topLeftCorner(m, m).diagonal().cwiseAbs2().maxCoeff() : 0.0;
		for (int a : candidates) {
			Eigen::VectorXd l(m);
			for (Eigen::Index k = 0; k < m; ++k) {
				l(k) = H(order[static_cast<size_t>(k)], a);
			}
			if (m > 0) {
				L.topLeftCorner(m, m).triangularView<Eigen::Lower>().solveInPlace(l);
			}
			const double d2 = H(a, a) - l.squaredNorm();
			if (!std::isfinite(d2)) {
				clear();
				return false;
			}
			if (d2 > 0.0 && d2 > 1e-10 * H(a, a) && (m == 0 || d2 >= 1e-10 * maxBasisDiag)) {
				ensureCapacity(m + 1);
				if (m > 0) {
					L.block(m, 0, 1, m) = l.transpose();
				}
				L(m, m) = std::sqrt(d2);
				order.push_back(a);
				++m;
				maxBasisDiag = std::max(maxBasisDiag, d2);
				if (!conditioned()) {
					clear();
					return false;
				}
			} else {
				remaining.push_back(a);
			}
		}
		dep = std::move(remaining);
		return true;
	}

	// the factor's k-th column leaves: the trailing block takes a rank-one update with the column below L_kk
	// (L33' L33'^T = L33 L33^T + l l^T) by Givens rotations, which never divide by L_jj; then row and column k are
	// deleted. False (and factor cleared) when the factor left fails the conditioning test
	bool remove(Eigen::Index k) {
		const Eigen::Index r = m - k - 1;
		Eigen::VectorXd v = L.block(k + 1, k, r, 1);
		for (Eigen::Index j = 0; j < r; ++j) {
			const Eigen::Index jj = k + 1 + j;
			const double ljj = L(jj, jj);
			const double rr = std::hypot(ljj, v(j));
			const double c = ljj / rr;
			const double s = v(j) / rr;
			L(jj, jj) = rr;
			for (Eigen::Index i = j + 1; i < r; ++i) {
				const Eigen::Index ii = k + 1 + i;
				const double lij = L(ii, jj);
				L(ii, jj) = c * lij + s * v(i);
				v(i) = c * v(i) - s * lij;
			}
		}
		for (Eigen::Index j = 0; j < k; ++j) {
			for (Eigen::Index i = k; i + 1 < m; ++i) {
				L(i, j) = L(i + 1, j);
			}
		}
		for (Eigen::Index j = k; j + 1 < m; ++j) {
			for (Eigen::Index i = j; i + 1 < m; ++i) {
				L(i, j) = L(i + 1, j + 1);
			}
		}
		order.erase(order.begin() + k);
		--m;
		if (!conditioned()) {
			clear();
			return false;
		}
		return true;
	}

	// x = H_S^-1 r, in the factor's order (when dep is empty)
	Eigen::VectorXd solve(const Eigen::VectorXd& r) const {
		Eigen::VectorXd x = r;
		if (m > 0) {
			L.topLeftCorner(m, m).triangularView<Eigen::Lower>().solveInPlace(x);
			L.topLeftCorner(m, m).transpose().triangularView<Eigen::Upper>().solveInPlace(x);
		}
		return x;
	}

	// Exact minimum-norm solve and orthonormal null basis over S = (B, D) using the kept factor L11 of B,
	// in O(|D| m^2 + |D|^2 (m + |D|)). Verifies that H_DD - L21 L21' is negligible and the basis is well-scaled;
	// returns false if the caller should re-factor with SymmetricSolve.
	bool solveWithDependent(const Eigen::MatrixXd& H, const Eigen::VectorXd& rB, const Eigen::VectorXd& rD,
	                        Eigen::VectorXd& xB, Eigen::VectorXd& xD, Eigen::MatrixXd& nullBasis) const {
		const Eigen::Index r = m;
		const Eigen::Index d = static_cast<Eigen::Index>(dep.size());
		const Eigen::Index n = r + d;
		if (d == 0) {
			return false;
		}
		const auto L11 = L.topLeftCorner(r, r).triangularView<Eigen::Lower>();
		Eigen::MatrixXd L21T(r, d);
		for (Eigen::Index j = 0; j < d; ++j) {
			for (Eigen::Index i = 0; i < r; ++i) {
				L21T(i, j) = H(order[static_cast<size_t>(i)], dep[static_cast<size_t>(j)]);
			}
		}
		if (r > 0) {
			L11.solveInPlace(L21T);
		}
		double maxDiag = 0.0;
		for (Eigen::Index i = 0; i < r; ++i) {
			const int bi = order[static_cast<size_t>(i)];
			maxDiag = std::max(maxDiag, std::abs(H(bi, bi)));
		}
		Eigen::MatrixXd Sdd(d, d);
		for (Eigen::Index j = 0; j < d; ++j) {
			const int dj = dep[static_cast<size_t>(j)];
			maxDiag = std::max(maxDiag, std::abs(H(dj, dj)));
			for (Eigen::Index i = 0; i < d; ++i) {
				Sdd(i, j) = H(dep[static_cast<size_t>(i)], dj);
			}
		}
		if (r > 0) {
			Sdd.noalias() -= L21T.transpose() * L21T;
		}
		if (!Sdd.allFinite() || !(Sdd.stableNorm() <= 1e-10 * maxDiag)) {
			return false;
		}
		Eigen::MatrixXd basis(n, d);
		if (r > 0) {
			basis.topRows(r) = -L11.transpose().solve(L21T);
			if (!basis.topRows(r).allFinite() || basis.topRows(r).cwiseAbs().maxCoeff() > 1e4) {
				return false; // poorly scaled basis (e.g. a much larger scaled duplicate entered after B): re-pivot
			}
		}
		basis.bottomRows(d).setIdentity();
		const Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
		nullBasis = qr.householderQ() * Eigen::MatrixXd::Identity(n, d);

		Eigen::VectorXd b(n);
		b.head(r) = rB;
		b.tail(d) = rD;
		const Eigen::VectorXd range = b - nullBasis * (nullBasis.transpose() * b);
		Eigen::VectorXd full = Eigen::VectorXd::Zero(n);
		if (r > 0) {
			Eigen::VectorXd leading = range.head(r);
			L11.solveInPlace(leading);
			L11.transpose().solveInPlace(leading);
			full.head(r) = leading;
		}
		const Eigen::VectorXd sol = full - nullBasis * (nullBasis.transpose() * full);
		if (!sol.allFinite()) {
			return false;
		}
		xB = sol.head(r);
		xD = sol.tail(d);
		return true;
	}

private:
	void ensureCapacity(Eigen::Index needed) {
		if (L.rows() >= needed && L.cols() >= needed) {
			return;
		}
		const Eigen::Index target = std::min(std::max(capacity, needed), std::max<Eigen::Index>(needed + 64, needed + needed / 4));
		L.conservativeResize(target, target);
	}

	// the first part of choleskyConditioned(); the second holds through removals (a pivot can only grow when a column
	// before it leaves) and add() checks it for the entering column
	bool conditioned() const {
		if (m == 0) {
			return true;
		}
		const Eigen::VectorXd d = L.topLeftCorner(m, m).diagonal().array().square();
		return d.allFinite() && d.minCoeff() >= 1e-10 * d.maxCoeff();
	}

	Eigen::Index capacity = 0;
	Eigen::MatrixXd L;
	Eigen::Index m = 0;
	std::vector<int> order;
	std::vector<int> dep;
};

} // namespace newton
} // namespace bsccs

#endif /* NEWTONSOLVER_H_ */
