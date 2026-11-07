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

enum class SolveMethod { CHOLESKY, LDLT, EIGEN };

// x = A^-1 b for symmetric A. Cholesky only when min L_ii^2 >= 1e-10 max L_ii^2, pivoted LDLT only when
// min |D_ii| >= 1e-10 max |D_ii|, else a minimum-norm eigen solve that drops directions with eigenvalues below
// 1e-10 times the largest (Eigen reports success on collinear designs: plan 2.6, E17). The Cholesky factor is
// kept for the inverse diagonal of the gsc certificate.
struct SymmetricSolve {
	Eigen::VectorXd x;
	SolveMethod method = SolveMethod::EIGEN;
	int dropped = 0;
	Eigen::MatrixXd nullBasis;      // eigen path: the dropped (orthonormal) directions
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
