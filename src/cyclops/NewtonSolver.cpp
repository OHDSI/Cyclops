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
#include <sstream>

#include "CyclicCoordinateDescent.h"
#include "Iterators.h"
#include "ModelData.h"
#include "NewtonSolver.h"

namespace bsccs {

struct CyclicCoordinateDescent::NewtonContext {
	std::vector<int> F;             // free coordinates (Normal or None priors)
	std::vector<double> precision;  // prior precision 1 / sigma_j^2 (0 for None)
	bool allNormal = true;
	double tauEff = 0.0;
	double nuNormal = NAN;          // all-Normal form of nu, fixed for the fit
	double eta = 5.0;               // eta trust region: t max_k |x_k'd| <= eta (2.6)
	int rejections = 0;             // consecutive rejected line searches
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

} // namespace

std::string CyclicCoordinateDescent::secondOrderIneligibility() const {
	if (usingGPU || usingCUDA) {
		return "it runs on a GPU";
	}
	if (syncCV) {
		return "cross-validation folds are synchronized";
	}
	if (!modelSpecifics.supportsSecondOrder()) {
		return "the model has no second-order kernels (so far cox and cox_counting in double precision on the CPU)";
	}
	for (int j = 0; j < J; ++j) {
		double precision = 0.0;
		if (!fixBeta[j] && !jointPrior->getNormalPrecision(j, &precision)) {
			return "a free coefficient has a prior other than normal or none";
		}
	}
	return "";
}

// One H at the current point: Lambda^2 / 2 and, when H is well conditioned, gsc (3.1)
bool CyclicCoordinateDescent::certifyGap(NewtonContext& context, double* gscValue, double* lambda2Half) {
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

CyclicCoordinateDescent::NewtonOutcome CyclicCoordinateDescent::newtonIteration(NewtonContext& context) {

	const size_t n = context.F.size();
	std::vector<double> gradient, block;
	if (!modelSpecifics.fullGradient(context.F, gradient) || !modelSpecifics.hessianBlock(context.F, block)) {
		return NewtonOutcome::FAILED;
	}
	++convergenceReport.hessians;
	Eigen::VectorXd s(n);
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
		return NewtonOutcome::FAILED;
	}
	const newton::SymmetricSolve solve(H, s);
	Eigen::VectorXd d = -solve.x;
	const double lambda2 = -s.dot(d);
	convergenceReport.droppedDirections = solve.dropped;
	if (!(lambda2 >= 0.0) || !d.allFinite()) {
		return NewtonOutcome::FAILED;
	}
	convergenceReport.lambda2Half = 0.5 * lambda2;

	// Stop test at the current point (3.2): gsc <= tau_eff certifies it, and its last step is still taken; while
	// gsc is unavailable (c Lambda >= 1, or H not positive definite) Lambda^2 / 2 <= tau_eff stops uncertified.
	// gsc >= Lambda^2 / 2, so nu is only computed when that can pass.
	bool stop = false;
	bool certified = false;
	if (context.stopType == GAP) {
		if (0.5 * lambda2 <= context.tauEff) {
			stop = true;
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
					certified = value <= context.tauEff;
				}
			}
		}
	} else if (context.stopType == KKT) {
		stop = kkt < context.tolerance;
	}

	// None coordinates get |delta_j| <= hDelta_j when the solve dropped or damped directions: the eta trust region
	// cannot see directions in the null space of X (2.6)
	if (solve.method != newton::SolveMethod::CHOLESKY) {
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
		if (lastStep.size() == hDelta.size()) {
			lastStep[j] = 0.0;
			flipCount[j] = 0;
		}
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
		if (!fixBeta[j] && jointPrior->getNormalPrecision(j, &precision)) {
			context.F.push_back(j);
			context.precision.push_back(precision);
			context.allNormal = context.allNormal && precision > 0.0;
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
	const double costRatio = modelSpecifics.newtonCostRatio(context.F);
	convergenceReport.costRatio = costRatio;

	const int warmStart = 3; // 2.4 step 2
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
			const NewtonOutcome outcome = newtonIteration(context);
			if (outcome == NewtonOutcome::CERTIFIED) {
				finish(SUCCESS, "gsc", convergenceReport.gsc);
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

		// Certificate first (2.4 step 3): R when every free coordinate is Normal. Its gradient pass is skipped while
		// the step-decrease estimate is still far above tau_eff (R >= gap, so no stop is missed; the
		// estimate reads at most a few times low, E16); every 16th cycle computes it regardless, since the estimate
		// reads far too high where the rate nears 1
		const bool farAway = std::isfinite(gHat) && gHat > 1e3 * context.tauEff && cycles > 1 && cycles % 16 != 0;
		if (context.allNormal && !farAway) {
			std::vector<double> gradient;
			if (modelSpecifics.fullGradient(context.F, gradient)) {
				double sum = 0.0;
				for (size_t i = 0; i < context.F.size(); ++i) {
					const double s = gradient[i] + context.precision[i] * hBeta[context.F[i]];
					sum += s * s / context.precision[i];
				}
				R = 0.5 * sum;
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

		if (algorithmType == AlgorithmType::NEWTON && cycles >= warmStart) {
			newtonPhase = true;
		} else if (algorithmType == AlgorithmType::AUTO && cycles >= warmStart && cycles >= suppressUntil &&
				nHat > 4.0 * costRatio) {
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
		if (convergenceType == GAP && cycles >= warmStart && cycles >= certifyAfter) {
			const double estimate = context.allNormal ? R : gHat;
			const double remaining = context.allNormal ?
				((rho < 1.0 && R > context.tauEff) ? std::log(R / context.tauEff) / -std::log(rho) :
					std::numeric_limits<double>::infinity()) : nHat;
			if (estimate <= 100.0 * context.tauEff && (remaining > costRatio || estimate <= context.tauEff)) {
				double value = NAN, lambda2Half = NAN;
				if (certifyGap(context, &value, &lambda2Half)) {
					convergenceReport.gsc = value;
					convergenceReport.lambda2Half = lambda2Half;
					if (value <= context.tauEff) {
						finish(SUCCESS, "gsc", value);
						break;
					}
				}
				certifyAfter = cycles + (1 << std::min(certifyFailures++, 20));
			}
		}
	}
}

} // namespace bsccs
