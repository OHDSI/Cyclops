/*
 * FitterPresets.cpp
 *
 * See FitterPresets.h. This was R's .fitterSettings(), .resolveAlgorithm() and .resolveControl(); R now calls it.
 */

#include "FitterPresets.h"

#include <stdexcept>

namespace bsccs {

namespace {

// Active-set cycling pays off only with many covariates (it lost with a support of about 100)
const int activeSetMinCovariates = 1000;

bool secondOrderModel(const std::optional<std::string>& modelType) { // second-order kernels (Newton plan 2.2)
	if (!modelType) return false;
	for (const char* m : {"cox", "cox_counting", "cpr", "sccs", "clr", "lr"}) {
		if (*modelType == m) return true;
	}
	return false;
}

bool is(const std::optional<std::string>& value, const char* text) {
	return value && *value == text;
}

// Settings that createControl() leaves to the fitter. fitter = "super" changes logistic regression: the "kkt"
// convergence criterion, cross-validation fold fits by "kkt" at a looser tolerance, a search that steps down first and
// refines its result, per-fold warm starts, active-set cycling for models with many covariates and adaptive intercept
// steps. Least squares uses "kkt", active-set cycling, adaptive intercept steps and a search that steps up from the
// null model and refines its result under "default" too, as the earlier criterion could stop far from the optimum and
// the earlier start and first step depended on the units of y; "super" adds fold fits to a looser tolerance, which
// changes the selected variance slightly. A setting given to createControl() wins. The fitter leaves
// cvConvergenceType and cvTolerance unset where its fold fits follow the fit.
ControlRequest fitterSettings(const ControlRequest& control) {
	const int activeSetEvery = (control.nCovariates && *control.nCovariates >= activeSetMinCovariates) ? 10 : 0;
	ControlRequest s;
	s.cvMinIterations = 5;
	s.cvRefine = s.cvNullModelBound = false;
	s.activeSetEvery = 0;
	s.activeSetThreshold = 0.01;
	s.adaptiveIntercept = false;
	if (control.fitter == "super" && is(control.modelType, "lr")) {
		s.convergenceType = "kkt";
		s.cvConvergenceType = "kkt";
		s.cvSearchDirection = "down";
		s.cvRefine = s.cvNullModelBound = s.cvWarmStart = true;
		s.activeSetEvery = activeSetEvery;
		s.adaptiveIntercept = true;
	} else if (control.fitter == "super" && is(control.modelType, "ls")) {
		s.convergenceType = "kkt";
		s.cvConvergenceType = "kkt";
		s.cvTolerance = 0.1;
		s.cvSearchDirection = "up";
		s.cvRefine = s.cvNullModelBound = s.cvWarmStart = true;
		s.activeSetEvery = activeSetEvery;
		s.adaptiveIntercept = true;
	} else if (control.fitter == "default" && is(control.modelType, "ls")) {
		s.convergenceType = "kkt";
		s.cvSearchDirection = "up";
		s.cvRefine = s.cvNullModelBound = true;
		s.cvWarmStart = !control.resetCoefficients;
		s.activeSetEvery = activeSetEvery;
		s.adaptiveIntercept = true;
	} else { // warm starts make fold fits without resetCoefficients independent of threads
		s.convergenceType = "gradient";
		s.cvSearchDirection = "auto";
		s.cvWarmStart = !control.resetCoefficients && control.fitter != "legacy";
	}
	return s;
}

struct Working { // the control during resolution
	ControlRequest control;
	std::string fallbackConvergenceType;
	std::optional<double> fallbackTolerance;
	std::optional<int> fallbackMaxIterations;
};

ResolvedControl resolve(Working w, std::vector<std::string>& warnings);

// algorithm and the "gap" criterion (Newton plan 2.10). Unset means the fitter's choice: "auto" for models with
// second-order kernels unless the fitter is "legacy", else "ccd". An explicit criterion other than "gap" under "auto"
// runs "ccd", bitwise as before. "auto" or "newton" without a criterion means "gap"; for "auto", the criterion the
// fitter would have used on CCD is kept as the fallback for a fit that cannot take Newton steps (a tolerance given
// without a criterion applies only there).
void resolveAlgorithm(Working& w, std::vector<std::string>& warnings) {
	ControlRequest& c = w.control;
	if (!c.algorithm) {
		c.algorithm = (c.fitter != "legacy" && secondOrderModel(c.modelType)) ? "auto" : "ccd";
	}
	const bool explicitType = c.convergenceType.has_value();
	if (*c.algorithm == "auto" && explicitType && *c.convergenceType != "gap") {
		c.algorithm = "ccd";
	}
	if ((*c.algorithm == "auto" || *c.algorithm == "newton") && !explicitType) {
		if (*c.algorithm == "auto") {
			Working ccd = w;
			ccd.control.algorithm = "ccd";
			const ResolvedControl fallback = resolve(ccd, warnings);
			w.fallbackConvergenceType = fallback.convergenceType;
			w.fallbackTolerance = fallback.tolerance;
			w.fallbackMaxIterations = fallback.maxIterations;
		}
		c.convergenceType = "gap";
		c.tolerance.reset();
	}
}

ResolvedControl resolve(Working w, std::vector<std::string>& warnings) {
	resolveAlgorithm(w, warnings);
	ControlRequest& c = w.control;
	const ControlRequest s = fitterSettings(c);
	if (c.fitter == "super" && is(c.modelType, "lr") && c.tolerance && !c.convergenceType) {
		warnings.push_back("tolerance is ignored without convergenceType under fitter = \"super\": its \"kkt\" "
		                   "criterion measures the tolerance in standard-error units");
		c.tolerance.reset();
	}
	// A fitter's fold criterion and tolerance are relative to its own criterion (super: "kkt" folds looser than its
	// "kkt" fit, for coordinate descent). When the fit uses another criterion (the "gap" of "auto" or "newton", or a
	// given one), they do not apply and the fold fits follow the fit; a given cvConvergenceType always applies
	const bool ownCriterion = !c.convergenceType || c.convergenceType == s.convergenceType;
	if (!c.convergenceType) c.convergenceType = s.convergenceType;
	if (ownCriterion && !c.cvConvergenceType) c.cvConvergenceType = s.cvConvergenceType;
	if (ownCriterion && !c.cvTolerance) c.cvTolerance = s.cvTolerance;
	if (!c.cvMinIterations) c.cvMinIterations = s.cvMinIterations;
	if (!c.cvSearchDirection) c.cvSearchDirection = s.cvSearchDirection;
	if (!c.cvRefine) c.cvRefine = s.cvRefine;
	if (!c.cvNullModelBound) c.cvNullModelBound = s.cvNullModelBound;
	if (!c.cvWarmStart) c.cvWarmStart = s.cvWarmStart;
	const bool filledActiveSet = !c.activeSetEvery;
	if (filledActiveSet) c.activeSetEvery = s.activeSetEvery;
	if (!c.activeSetThreshold) c.activeSetThreshold = s.activeSetThreshold;
	const bool filledAdaptive = !c.adaptiveIntercept;
	if (filledAdaptive) c.adaptiveIntercept = s.adaptiveIntercept;

	const bool kkt = *c.convergenceType == "kkt"; // in standard-error units; an honest rule needs more cycles
	const bool gap = *c.convergenceType == "gap"; // in log-posterior units (Newton plan 3.3, D3)
	if (!c.tolerance) c.tolerance = gap ? 1E-6 : (kkt ? 1E-4 : 1E-6); // gap: certified bounds are conservative
	if (!c.maxIterations) c.maxIterations = (kkt || gap) ? 5000 : 1000;
	if (gap && !c.cvConvergenceType) c.cvConvergenceType = "gap"; // fold fits stop at a looser gap
	if (c.cvConvergenceType) { // fold fits with their own rule get its defaults; kkt folds stop earlier
		const bool cvKkt = *c.cvConvergenceType == "kkt";
		const bool cvGap = *c.cvConvergenceType == "gap";
		if (!c.cvTolerance) c.cvTolerance = cvGap ? 1E-5 : (cvKkt ? 3E-3 : 1E-6);
		if (!c.cvMaxIterations) c.cvMaxIterations = (cvKkt || cvGap) ? 5000 : 1000;
	}
	if (!kkt && !is(c.cvConvergenceType, "kkt")) { // the fitter's settings that need "kkt" lapse; given ones are errors
		if (filledActiveSet) c.activeSetEvery = 0;
		if (filledAdaptive) c.adaptiveIntercept = false;
		if (*c.activeSetEvery != 0) {
			throw std::invalid_argument("activeSetEvery needs convergenceType = \"kkt\" or cvConvergenceType = \"kkt\"");
		}
		if (*c.adaptiveIntercept) {
			throw std::invalid_argument("adaptiveIntercept needs convergenceType = \"kkt\" or cvConvergenceType = \"kkt\"");
		}
	}

	ResolvedControl r;
	r.algorithm = *c.algorithm;
	r.convergenceType = *c.convergenceType;
	r.tolerance = *c.tolerance;
	r.maxIterations = *c.maxIterations;
	r.cvConvergenceType = c.cvConvergenceType;
	r.cvTolerance = c.cvTolerance;
	r.cvMaxIterations = c.cvMaxIterations;
	r.cvMinIterations = *c.cvMinIterations;
	r.cvSearchDirection = *c.cvSearchDirection;
	r.cvRefine = *c.cvRefine;
	r.cvNullModelBound = *c.cvNullModelBound;
	r.cvWarmStart = *c.cvWarmStart;
	r.activeSetEvery = *c.activeSetEvery;
	r.activeSetThreshold = *c.activeSetThreshold;
	r.adaptiveIntercept = *c.adaptiveIntercept;
	r.fallbackConvergenceType = w.fallbackConvergenceType;
	r.fallbackTolerance = w.fallbackTolerance;
	r.fallbackMaxIterations = w.fallbackMaxIterations;
	return r;
}

} // namespace

ResolvedControl resolveControl(const ControlRequest& request) {
	std::vector<std::string> warnings;
	Working w;
	w.control = request;
	ResolvedControl r;
	try {
		r = resolve(w, warnings);
	} catch (const std::invalid_argument& e) {
		r.error = e.what();
	}
	r.warnings = warnings;
	return r;
}

} // namespace bsccs
