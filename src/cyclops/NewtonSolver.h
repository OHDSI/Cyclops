/*
 * NewtonSolver.h
 *
 * Numerics shared by the certified "gap" stops and the dense Newton moves of CyclicCoordinateDescent::findMode
 * (Newton plan sections 2.4-2.6 and 3; the moves themselves are CyclicCoordinateDescent members, defined in
 * NewtonSolver.cpp).
 */

#ifndef NEWTONSOLVER_H_
#define NEWTONSOLVER_H_

#include <cmath>
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
	Eigen::MatrixXd nullBasis;      // when dropped > 0: the dropped (orthonormal) directions
	Eigen::LLT<Eigen::MatrixXd> cholesky;

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
	// Cholesky with complete pivoting (LAPACK's dpstrf). M holds L in its leading columns and the Schur complement of the
	// pivots so far in its trailing block; each step takes the largest diagonal of that complement. A has rank-deficiency d
	// when the trailing d x d complement is negligible: its trace bounds its eigenvalues if it is positive semidefinite, and
	// its norm bounds them if it is not. Eigen's LDLT is not rank-revealing, as it picks each pivot before the
	// diagonal is updated. P A P' = [L11; L21] [L11' L21'] gives the null space, P' [-L11^-T L21'; I], and A^+ b.
	// false when A has full rank at the threshold or is not positive semidefinite
	bool solveSemidefinite(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
		const Eigen::Index n = A.rows();
		Eigen::MatrixXd M = A;
		Eigen::VectorXi order = Eigen::VectorXi::LinSpaced(n, 0, static_cast<int>(n) - 1);
		const double threshold = 1e-10 * A.diagonal().cwiseAbs().maxCoeff();
		if (!std::isfinite(threshold)) {
			return false;
		}
		Eigen::Index rank = 0;
		for (; rank < n; ++rank) {
			if (M.diagonal().tail(n - rank).cwiseAbs().sum() <= threshold) {
				break;
			}
			Eigen::Index largest;
			const double pivot = M.diagonal().tail(n - rank).maxCoeff(&largest);
			largest += rank;
			if (!(pivot > 0.0)) {
				return false;
			}
			if (largest != rank) {
				M.row(rank).swap(M.row(largest));
				M.col(rank).swap(M.col(largest));
				std::swap(order(rank), order(largest));
			}
			const double root = std::sqrt(pivot);
			M(rank, rank) = root;
			const Eigen::Index below = n - rank - 1;
			if (below > 0) {
				M.col(rank).tail(below) /= root;
				M.bottomRightCorner(below, below).noalias() -= M.col(rank).tail(below) * M.col(rank).tail(below).transpose();
			}
		}
		const Eigen::Index deficiency = n - rank;
		if (deficiency == 0 || !(M.bottomRightCorner(deficiency, deficiency).norm() <= threshold) ||
				!M.leftCols(rank).allFinite()) {
			return false;
		}
		const auto L11 = M.topLeftCorner(rank, rank).triangularView<Eigen::Lower>();
		Eigen::MatrixXd pivoted(n, deficiency);
		pivoted.topRows(rank) = -L11.transpose().solve(M.bottomLeftCorner(deficiency, rank).transpose());
		pivoted.bottomRows(deficiency).setIdentity();
		Eigen::MatrixXd basis(n, deficiency);
		for (Eigen::Index i = 0; i < n; ++i) {
			basis.row(order(i)) = pivoted.row(i);
		}
		const Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
		nullBasis = qr.householderQ() * Eigen::MatrixXd::Identity(n, deficiency);

		const Eigen::VectorXd range = b - nullBasis * (nullBasis.transpose() * b);
		Eigen::VectorXd leading(rank);
		for (Eigen::Index i = 0; i < rank; ++i) {
			leading(i) = range(order(i));
		}
		L11.solveInPlace(leading);
		L11.transpose().solveInPlace(leading);
		Eigen::VectorXd full = Eigen::VectorXd::Zero(n);
		for (Eigen::Index i = 0; i < rank; ++i) {
			full(order(i)) = leading(i);
		}
		x = full - nullBasis * (nullBasis.transpose() * full);
		dropped = static_cast<int>(deficiency);
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

// Cholesky factor of a face matrix H_S, kept across the rounds of prox-Newton's face solve: a column enters by a
// triangular solve and leaves by a rank-one update of the trailing block, O(m^2) each instead of a new O(m^3)
// factorization. members()[k] is the position in H of the factor's k-th column, in the order of entry. A face that
// fails SymmetricSolve's conditioning test (choleskyConditioned()) is not factored here: the caller solves it
// with SymmetricSolve, which handles singular faces. The test sees the pivots in the order of entry, SymmetricSolve in
// the order of W, so for faces near the threshold (condition number about 1e10) the two can decide differently
class FaceFactor {
public:
	explicit FaceFactor(Eigen::Index capacity) : L(capacity, capacity) {}

	const std::vector<int>& members() const { return order; }

	// the factor of H_S that SymmetricSolve computed and accepted (same matrix, same order, same test): no second
	// factorization, and no second m x m matrix at the peak
	void seat(const Eigen::LLT<Eigen::MatrixXd>& llt, const std::vector<int>& S) {
		m = static_cast<Eigen::Index>(S.size());
		order = S;
		if (m > 0) {
			L.topLeftCorner(m, m) = llt.matrixL();
		}
	}

	// column a of H enters last; false (and the factor unusable) when it fails the same test
	bool add(const Eigen::MatrixXd& H, int a) {
		Eigen::VectorXd l(m);
		for (Eigen::Index k = 0; k < m; ++k) {
			l(k) = H(order[k], a);
		}
		if (m > 0) {
			L.topLeftCorner(m, m).triangularView<Eigen::Lower>().solveInPlace(l);
		}
		// the part of column a that the face does not explain must be a real share of its curvature, whatever the order
		// of entry (a scaled duplicate leaves rounding noise here)
		const double d2 = H(a, a) - l.squaredNorm();
		if (!(d2 > 1e-10 * H(a, a)) || !std::isfinite(d2)) {
			return false;
		}
		L.block(m, 0, 1, m) = l.transpose();
		L(m, m) = std::sqrt(d2);
		order.push_back(a);
		++m;
		return conditioned();
	}

	// the factor's k-th column leaves: the trailing block takes a rank-one update with the column below L_kk
	// (L33' L33'^T = L33 L33^T + l l^T) by Givens rotations, which never divide by L_jj; then row and column k are
	// deleted. False when the factor left fails the conditioning test (deleting a column can raise max L_ii^2)
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
		return conditioned();
	}

	// x = H_S^-1 r, in the factor's order
	Eigen::VectorXd solve(const Eigen::VectorXd& r) const {
		Eigen::VectorXd x = r;
		if (m > 0) {
			L.topLeftCorner(m, m).triangularView<Eigen::Lower>().solveInPlace(x);
			L.topLeftCorner(m, m).transpose().triangularView<Eigen::Upper>().solveInPlace(x);
		}
		return x;
	}

private:
	// the first part of choleskyConditioned(); the second holds through removals (a pivot can only grow when a column
	// before it leaves) and add() checks it for the entering column
	bool conditioned() const {
		if (m == 0) {
			return true;
		}
		const Eigen::VectorXd d = L.topLeftCorner(m, m).diagonal().array().square();
		return d.allFinite() && d.minCoeff() >= 1e-10 * d.maxCoeff();
	}

	Eigen::MatrixXd L;
	Eigen::Index m = 0;
	std::vector<int> order;
};

} // namespace newton
} // namespace bsccs

#endif /* NEWTONSOLVER_H_ */
