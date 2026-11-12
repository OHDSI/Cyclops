/*
 * FitterPresets.h
 *
 * The settings a fit takes from its fitter ("default", "super", "legacy"), its model and its algorithm when the user
 * leaves them unset: one place for R (createControl() through .resolveControl()) and the Python package. Unset
 * settings are empty optionals, in the request and where they stay unset in the result (R's NULL).
 */

#ifndef FITTERPRESETS_H_
#define FITTERPRESETS_H_

#include <optional>
#include <string>
#include <vector>

namespace bsccs {

// What the user set; empty = unset
struct ControlRequest {
	std::string fitter = "default";
	std::optional<std::string> modelType; // "lr", "pr", "ls", "cox", ...; unset when no data is at hand
	std::optional<int> nCovariates;
	bool resetCoefficients = false;
	std::optional<std::string> algorithm;       // "ccd", "mm", "auto", "newton"
	std::optional<std::string> convergenceType; // "gradient", "lange", ..., "kkt", "gap"
	std::optional<double> tolerance;
	std::optional<int> maxIterations;
	std::optional<std::string> cvConvergenceType;
	std::optional<double> cvTolerance;
	std::optional<int> cvMaxIterations;
	std::optional<int> cvMinIterations;
	std::optional<std::string> cvSearchDirection; // "auto", "up", "down"
	std::optional<bool> cvRefine;
	std::optional<bool> cvNullModelBound;
	std::optional<bool> cvWarmStart;
	std::optional<int> activeSetEvery;
	std::optional<double> activeSetThreshold;
	std::optional<bool> adaptiveIntercept;
};

// The settings of the fit; the fold settings stay unset where fold fits follow the fit
struct ResolvedControl {
	std::string algorithm;
	std::string convergenceType;
	double tolerance = 0.0;
	int maxIterations = 0;
	std::optional<std::string> cvConvergenceType;
	std::optional<double> cvTolerance;
	std::optional<int> cvMaxIterations;
	int cvMinIterations = 5;
	std::string cvSearchDirection;
	bool cvRefine = false;
	bool cvNullModelBound = false;
	bool cvWarmStart = false;
	int activeSetEvery = 0;
	double activeSetThreshold = 0.01;
	bool adaptiveIntercept = false;
	std::string fallbackConvergenceType;    // "auto" with the "gap" criterion: the criterion of a CCD fallback
	std::optional<double> fallbackTolerance;
	std::optional<int> fallbackMaxIterations;
	std::vector<std::string> warnings;      // for the caller to issue, in order
	std::string error;                      // settings that contradict each other (after the warnings); else empty
};

ResolvedControl resolveControl(const ControlRequest& request);

} // namespace bsccs

#endif /* FITTERPRESETS_H_ */
