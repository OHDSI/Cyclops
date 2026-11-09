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

// x = A^-1 b for symmetric A. Cholesky only when min L_ii^2 >= 1e-10 max L_ii^2. Otherwise a positive
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
			if (d.size() == 0 || d.minCoeff() >= 1e-10 * d.maxCoeff()) {
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

} // namespace newton
} // namespace bsccs

#endif /* NEWTONSOLVER_H_ */
