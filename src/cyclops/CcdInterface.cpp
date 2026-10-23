/*
 * ccd.cpp
 *
 *  Created on: July, 2010
 *      Author: msuchard
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>

#include <iostream>
#include <time.h>

#ifndef _MSC_VER
	#include <sys/time.h>
#endif

#include <math.h>

// #include "Types.h"
#include "CcdInterface.h"
#include "CyclicCoordinateDescent.h"
#include "ModelData.h"

//#include "boost/iterator/counting_iterator.hpp"
#include "Thread.h"

// #include "io/InputReader.h"
// #include "io/HierarchyReader.h"
// #include "io/CLRInputReader.h"
// #include "io/RTestInputReader.h"
// #include "io/CCTestInputReader.h"
// #include "io/CoxInputReader.h"
// #include "io/NewCLRInputReader.h"
// #include "io/NewSCCSInputReader.h"
// #include "io/NewCoxInputReader.h"
// #include "io/NewGenericInputReader.h"
// #include "io/BBRInputReader.h"
// #include "io/OutputWriter.h"
#include "drivers/CrossValidationSelector.h"
#include "drivers/WeightBasedSelector.h"
#include "drivers/GridSearchCrossValidationDriver.h"
#include "drivers/HierarchyGridSearchCrossValidationDriver.h"
#include "drivers/AutoSearchCrossValidationDriver.h"
#include "drivers/HierarchyAutoSearchCrossValidationDriver.h"
#include "drivers/BootstrapSelector.h"
#include "drivers/ProportionSelector.h"
#include "drivers/BootstrapDriver.h"
//#include "engine/ModelSpecifics.h"

// #include "tclap/CmdLine.h"
#include "utils/RZeroIn.h"

//#include <R.h>

#ifdef CUDA
	#include "GPUCyclicCoordinateDescent.h"
//	#include "BetterGPU.h"
#endif


#define NEW

namespace bsccs {

// using namespace TCLAP;
using namespace std;

#ifdef _MSC_VER
	#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
		#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
	#else
		#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
	#endif
	struct timezone
	{
		int  tz_minuteswest; /* minutes W of Greenwich */
		int  tz_dsttime;     /* type of dst correction */
	};

	// Definition of a gettimeofday function

	int gettimeofday(struct timeval *tv, struct timezone *tz)
	{
		// Define a structure to receive the current Windows filetime
		FILETIME ft;

		// Initialize the present time to 0 and the timezone to UTC
		unsigned __int64 tmpres = 0;
		static int tzflag = 0;

		if (NULL != tv)
		{
			GetSystemTimeAsFileTime(&ft);

			// The GetSystemTimeAsFileTime returns the number of 100 nanosecond
			// intervals since Jan 1, 1601 in a structure. Copy the high bits to
			// the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
			tmpres |= ft.dwHighDateTime;
			tmpres <<= 32;
			tmpres |= ft.dwLowDateTime;

			// Convert to microseconds by dividing by 10
			tmpres /= 10;

			// The Unix epoch starts on Jan 1 1970.  Need to subtract the difference
			// in seconds from Jan 1 1601.
			tmpres -= DELTA_EPOCH_IN_MICROSECS;

			// Finally change microseconds to seconds and place in the seconds value.
			// The modulus picks up the microseconds.
			tv->tv_sec = (long)(tmpres / 1000000UL);
			tv->tv_usec = (long)(tmpres % 1000000UL);
		}

		if (NULL != tz)
		{
			if (!tzflag)
			{
				_tzset();
				tzflag++;
			}

			// Adjust for the timezone west of Greenwich
			tz->tz_minuteswest = _timezone / 60;
			tz->tz_dsttime = _daylight;
		}

		return 0;
	}
#endif

CcdInterface::CcdInterface(void) {
    setDefaultArguments();
}

CcdInterface::~CcdInterface(void) {
    // Do nothing
}

double CcdInterface::calculateSeconds(const timeval &time1, const timeval &time2) {
	return time2.tv_sec - time1.tv_sec +
			(double)(time2.tv_usec - time1.tv_usec) / 1000000.0;
}

void CcdInterface::setDefaultArguments(void) {
	arguments.useGPU = false;
// 	arguments.maxIterations = 1000;
	arguments.inFileName = "default_in";
	arguments.outFileName = "default_out";
	arguments.outDirectoryName = "";
	arguments.hyperPriorSet = false;
	arguments.hyperprior = 1.0;
// 	arguments.tolerance = 1E-6; //5E-4;
	arguments.seed = -99;
// 	arguments.doCrossValidation = false;
// 	arguments.useAutoSearchCV = false;
// 	arguments.lowerLimit = 0.01;
// 	arguments.upperLimit = 20.0;
// 	arguments.fold = 10;
// 	arguments.gridSteps = 10;
// 	arguments.cvFileName = "cv.txt";
	arguments.useHierarchy = false;
	arguments.doBootstrap = false;
	arguments.replicates = 100;
	arguments.reportRawEstimates = false;
	arguments.reportDifference = false;
	arguments.modelName = "sccs";
	arguments.fitter = "default";
	arguments.fileFormat = "generic";
	//arguments.outputFormat = "estimates";
	arguments.computeMLE = false;
	arguments.fitMLEAtMode = false;
	arguments.reportASE = false;
	arguments.useNormalPrior = false;
// 	arguments.convergenceType = GRADIENT;
// 	arguments.convergenceTypeString = "gradient";
	arguments.doPartial = false;
	arguments.noiseLevel = NOISY;
	arguments.threads = -1;
	arguments.resetCoefficients = false;
}

double CcdInterface::initializeModel(
		AbstractModelData** modelData,
		CyclicCoordinateDescent** ccd,
		AbstractModelSpecifics** model) {

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

    initializeModelImpl(modelData, ccd, model);

	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

std::string CcdInterface::getPathAndFileName(const CCDArguments& arguments, std::string stem) {
	string fileName;
	if (arguments.outputFormat.size() == 1) {
		fileName = arguments.outDirectoryName + arguments.outFileName;
	} else {
		fileName = arguments.outDirectoryName + stem + arguments.outFileName;
	}
	return fileName;
}

double CcdInterface::predictModel(CyclicCoordinateDescent *ccd, AbstractModelData *modelData) {

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

    predictModelImpl(ccd, modelData);

    gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

struct OptimizationProfile {

	CyclicCoordinateDescent& ccd;
	CCDArguments& arguments;

	OptimizationProfile(CyclicCoordinateDescent& _ccd, CCDArguments& _arguments, int _index, double _max,
			double _threshold = 1.920729, bool _includePenalty = false) :
			ccd(_ccd), arguments(_arguments), index(_index), max(_max), threshold(_threshold),
			nEvals(0), includePenalty(_includePenalty) {
	}

	int getEvaluations() {
		return nEvals;
	}

	double objective(double x) {
		++nEvals;
		ccd.setBeta(index, x);
		ccd.setFixedBeta(index, true);
		ccd.update(arguments.modeFinding);
		ccd.setFixedBeta(index, false);
		double y = ccd.getLogLikelihood() + threshold - max;
		if (includePenalty) {
			y += ccd.getLogPrior();
		}
		return y;
	}

	double derivative() {
	    double d = ccd.getLogLikelihoodGradient(index);
	    if (includePenalty) {
	        d += ccd.getLogPriorGradient(index);
	    }
	    return d;
	}

	double getMaximum() {
		return threshold;
	}

	int index;
	double max;
	double threshold;
	int nEvals;
	bool includePenalty;
};

// Safeguarded Newton iteration for the profile bound in `direction` from the mode x0. It starts at the
// Wald bound for `curvature` (the other coefficients fixed, so inside the profile interval for a
// quadratic log likelihood) and uses the derivative of the profile, which is the partial derivative at
// the profile fit. Returns NAN when it fails; the caller then brackets a sign change instead.
static double getNewtonBound(OptimizationProfile& eval, double x0, double direction,
                             double curvature, double tolerance = 1E-3, int maxSteps = 20) {
	// A nearly flat profile (an estimate at infinity) puts the Wald bound so far out that the fits there
	// can overflow; leave it to bracketing
	const double halfWidth = std::sqrt(2.0 * eval.getMaximum() / curvature);
	if (!(halfWidth <= 10.0 * std::max(std::abs(x0), 1.0))) {
		return NAN;
	}
	double inside = x0;   // objective > 0
	double outside = NAN; // objective < 0
	double x = x0 + direction * halfWidth;
	for (int step = 0; step < maxSteps; ++step) {
		const double y = eval.objective(x);
		if (std::isnan(y)) {
			return NAN;
		}
		if (y == 0.0) {
			return x;
		}
		const double d = eval.derivative();
		if (y > 0.0) {
			inside = x;
		} else {
			outside = x;
		}
		double next = x - y / d;
		const bool descending = direction * d < 0.0 && std::isfinite(next);
		if (std::isnan(outside)) {
			if (!descending || (next - inside) * direction <= 0.0) {
				return NAN;
			}
		} else if (!descending || (next - inside) * direction <= 0.0 || (next - outside) * direction >= 0.0) {
			next = 0.5 * (inside + outside); // bisect the bracket
		}
		if (std::abs(next - x) < tolerance) {
			return next;
		}
		x = next;
	}
	return NAN;
}

double CcdInterface::profileModel(CyclicCoordinateDescent *ccd, AbstractModelData *modelData,
		const ProfileVector& profileCI, ProfileInformationMap& profileMap,
		int inThreads, double threshold,
		bool overrideNoRegularization, bool includePenalty) {

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	double mode = ccd->getLogLikelihood();
	if (includePenalty) {
	    mode += ccd->getLogPrior();
	}
	int J = ccd->getBetaSize();
	std::vector<double> x0s(J);
	for (int j = 0; j < J; ++j) {
	    x0s[j] = ccd->getBeta(j);
	}

#define NEW_PROFILE
#ifdef NEW_PROFILE
	typedef std::tuple<int,double> BoundType;
	typedef std::vector<BoundType> BoundVector;
	BoundVector bounds;
	std::vector<int> indices;

	int id = 0;

	for (ProfileVector::const_iterator it = profileCI.begin();
	        it != profileCI.end(); ++it, ++id) {
		int index = modelData->getColumnIndexByName(*it);

		if (index == -1) {
		    std::ostringstream stream;
			stream << "Variable " << *it << " not found.";
			error->throwError(stream);
		} else {
			// TODO Check prior on covariate
			if (!overrideNoRegularization && ccd->getIsRegularized(index)) {
			    std::ostringstream stream;
			    stream << "Variable " << *it << " is regularized.";
			    error->throwError(stream);
			}
			indices.push_back(index);
		    bounds.push_back(std::make_tuple(id, +1.0));
			bounds.push_back(std::make_tuple(id, -1.0));
		}
	}

	// Newton bounds need the derivative of the profile, which is the partial derivative at the profile fit
	// when the objective includes the penalty of every free coefficient that moves. Without includePenalty,
	// that holds only if no other free coefficient is regularized; with it, regularized covariates (whose
	// derivative needs the prior gradient) are left to bracketing too. The curvature at the mode seeds them.
	int nRegularized = 0;
	for (int j = 0; j < J; ++j) {
		if (ccd->getIsRegularized(j) && !ccd->getFixedBeta(j)) {
			++nRegularized;
		}
	}
	std::vector<double> curvatures(indices.size(), NAN);
	for (size_t id = 0; id < indices.size(); ++id) {
		const bool regularized = ccd->getIsRegularized(indices[id]);
		const bool exact = includePenalty ? !regularized : nRegularized == (regularized ? 1 : 0);
		if (exact) {
			curvatures[id] = ccd->getHessianDiagonal(indices[id]);
		}
	}

	// Parallelize across columns and lower/upper bound
	int nThreads = (inThreads == -1) ?
	    bsccs::thread::hardware_concurrency() : inThreads;

	std::ostringstream stream2;
	stream2 << "Using " << nThreads << " thread(s)";
	logger->writeLine(stream2);

	std::vector<CyclicCoordinateDescent*> ccdPool;

	ccdPool.push_back(ccd);

	for (int i = 1; i < nThreads; ++i) {
	    CyclicCoordinateDescent* copy = ccd->clone(arguments.computeDevice);
	    for (int j = 0; j < J; ++j) {
	        copy->setFixedBeta(j, ccd->getFixedBeta(j));
	    }
	    ccdPool.push_back(copy);
	}

    std::vector<double> lowerPts(indices.size());
	std::vector<double> upperPts(indices.size());
	std::vector<int> lowerCnts(indices.size());
	std::vector<int> upperCnts(indices.size());

	auto getBound = [this,
	            &x0s, &curvatures,
	            &indices, &lowerPts, &upperPts,
                &lowerCnts, &upperCnts, includePenalty, mode, threshold
            ](const BoundType bound, CyclicCoordinateDescent* ccd) {
	    const auto id = std::get<0>(bound);
	    const auto direction = std::get<1>(bound);
	    const auto index = indices[id];

	    double x0 = x0s[index];

	    // Bound edge, starting at the mode so that bounds do not depend on their order or the threads
	    ccd->setBeta(x0s);
	    OptimizationProfile eval(*ccd, arguments, index, mode, threshold, includePenalty);

	    double pt = curvatures[id] > 0.0 ?
	        getNewtonBound(eval, x0, direction, curvatures[id]) : NAN;

	    if (std::isnan(pt)) {
	        ccd->setBeta(x0s);
	        RZeroIn<OptimizationProfile> zeroIn(eval, 1E-3);

	        double obj0 = eval.getMaximum();

	        //		std::cout << "BEGIN UP bracket" << std::endl;
	        RZeroIn<OptimizationProfile>::Coordinate bracket =
	            zeroIn.bracketSignChange(x0, obj0, direction);
	        //		std::cout << "END UP bracket " << bracket.first << " " << bracket.second << std::endl;

	        pt = std::isnan(bracket.second) ? NAN : zeroIn.getRoot(x0, bracket.first, obj0, bracket.second);
	    }

	    if (direction == 1.0) {
	        upperPts[id] = pt;
	        upperCnts[id] = eval.getEvaluations();
	    } else { // direction == -1.0
	        lowerPts[id] = pt;
	        lowerCnts[id] = eval.getEvaluations();
	    }
	};

    ccd->getHBeta();

    if (nThreads == 1) {
        std::for_each(std::begin(bounds), std::end(bounds),
                      [&getBound, ccd](const BoundType bound) {
                        getBound(bound, ccd); // Use main CCD object
                      }
                    );
    } else {
        auto scheduler = TaskScheduler<IncrementableIterator<size_t>>(
            IncrementableIterator<size_t>(0), // boost::make_counting_iterator(0),
            IncrementableIterator<size_t>(bounds.size()), //boost::make_counting_iterator(static_cast<int>(bounds.size())),
            nThreads);

        auto oneTask = [&getBound, &scheduler, &ccdPool, &bounds](unsigned long task) {
            getBound(bounds[task], ccdPool[scheduler.getThreadIndex(task)]);
        };

        // Run all tasks in parallel
        ccd->getProgressLogger().setConcurrent(true);
        ccd->getErrorHandler().setConcurrent(true);
        scheduler.execute(oneTask);
        ccd->getProgressLogger().setConcurrent(false);
        ccd->getErrorHandler().setConcurrent(false);
        ccd->getProgressLogger().flush();
        ccd->getErrorHandler().flush();
    }

    // Clean up copies
    for (int i = 1; i < nThreads; ++i) {
        delete ccdPool[i]; // TODO use unique_ptr
    }

        // Build result serially
        auto itLowerPt = std::begin(lowerPts);
        auto itUpperPt = std::begin(upperPts);
        auto itLowerCnt = std::begin(lowerCnts);
        auto itUpperCnt = std::begin(upperCnts);
        for (auto itIndex = std::begin(indices); itIndex != std::end(indices);
            ++itIndex, ++itLowerPt, ++itUpperPt, ++itLowerCnt, ++itUpperCnt) {

            const auto index = *itIndex;
            const auto lowerPt = *itLowerPt;
            const auto upperPt = *itUpperPt;
            const auto evals = *itLowerCnt + *itUpperCnt;

			if (arguments.noiseLevel >= NOISY) {
	            std::ostringstream stream;
				stream << "Profile: " << modelData->getColumnLabel(index) << " (" << lowerPt << ", "
						<< upperPt << ")  in " << evals;
				logger->writeLine(stream);
			}

			ProfileInformation profile(lowerPt, upperPt, evals);
			profileMap.insert(std::pair<IdType, ProfileInformation>(modelData->getColumnNumericalLabel(index), profile));

			// std::cerr << "Placing " << profileCI[index] << " with " << profile.lower95Bound << std::endl;

		}

		// Reset to mode
		if (indices.size() > 0) {
		    // Reset exactly, and recompute X beta now, as prediction reads it directly
		    ccd->setBeta(x0s);
		    ccd->getLogLikelihood();
		    ccd->setHXBeta();
		    // DEBUG, TODO Remove?
// 		    double testMode = ccd->getLogLikelihood();
// 		    std::ostringstream stream;
// 		    stream << "Difference after profile: " << testMode << " " << mode;
// 		    logger->writeLine(stream);
		}
#else
		std::vector<int> columns;
		for (ProfileVector::const_iterator it = profileCI.begin();
       it != profileCI.end(); ++it) {
		    int index = modelData->getColumnIndexByName(*it);

		    //		std::cerr << "Matched: " << *it << " at " << index << std::endl;

		    if (index == -1) {
		        std::ostringstream stream;
		        stream << "Variable " << *it << " not found.";
		        error->throwError(stream);
		    } else {
		        // TODO Minor bug, order of column evaluation yields different estimates
		        // TODO Check prior on covariate
		        if (!overrideNoRegularization && ccd->getIsRegularized(index)) {
		            std::ostringstream stream;
		            stream << "Variable " << *it << " is regularized.";
		            error->throwError(stream);
		        }
		        columns.push_back(index);
		    }
		}

		for (std::vector<int>::const_iterator it = columns.begin();
       it != columns.end(); ++it) {

		    int index = *it;
		    double x0 = x0s[index];

		    // Bound edge
		    OptimizationProfile upEval(*ccd, arguments, index, mode, threshold, includePenalty);
		    RZeroIn<OptimizationProfile> zeroInUp(upEval, 1E-3);
		    RZeroIn<OptimizationProfile> zeroInDn(upEval, 1E-3);

		    double obj0 = upEval.getMaximum();

		    //		std::cout << "BEGIN UP bracket" << std::endl;
		    RZeroIn<OptimizationProfile>::Coordinate upperBracket =
		        zeroInUp.bracketSignChange(x0, obj0, 1.0);
		    // 		std::cout << "END UP bracket " << upperBracket.first << " " << upperBracket.second << std::endl;
		    if (std::isnan(upperBracket.second) == true) {
		        std::ostringstream stream;
		        stream << "Unable to bracket sign change in profile.";
		        error->throwError(stream);
		    }

		    double upperPt = zeroInUp.getRoot(x0, upperBracket.first, obj0, upperBracket.second);
		    // 		std::cout << "BEGIN DN bracket" << std::endl;
		    RZeroIn<OptimizationProfile>::Coordinate lowerBracket =
		        zeroInDn.bracketSignChange(x0, obj0, -1.0);
		    // 		std::cout << "END DN bracket " << lowerBracket.first << " " << lowerBracket.second << std::endl;
		    if (std::isnan(lowerBracket.second == true)) {
		        std::ostringstream stream;
		        stream << "Unable to bracket sign change in profile.";
		        error->throwError(stream);
		    }

		    double lowerPt = zeroInDn.getRoot(x0, lowerBracket.first, obj0, lowerBracket.second);

		    if (arguments.noiseLevel >= NOISY) {
		        std::ostringstream stream;
		        stream << "Profile: " << modelData->getColumn(index).getLabel() << " (" << lowerPt << ", "
                 << upperPt << ")  in " << upEval.getEvaluations();
		        logger->writeLine(stream);
		    }

		    ProfileInformation profile(lowerPt, upperPt, upEval.getEvaluations());
		    profileMap.insert(std::pair<IdType, ProfileInformation>(modelData->getColumn(index).getNumericalLabel(), profile));

		    //		std::cerr << "Placing " << profileCI[index] << " with " << profile.lower95Bound << std::endl;

		}

		// Reset to mode
		if (columns.size() > 0) {
		    // Reset
		    for (int j = 0; j < J; ++j) {
		        ccd->setBeta(j, x0s[j]);
		    }
		    // DEBUG, TODO Remove?
		    // 		    double testMode = ccd->getLogLikelihood();
		    // 		    std::ostringstream stream;
		    // 		    stream << "Difference after profile: " << testMode << " " << mode;
		    // 		    logger->writeLine(stream);
		}
#endif // NEW_PROFILE

    gettimeofday(&time2, NULL);
    return calculateSeconds(time1, time2);
}

double CcdInterface::evaluateProfileModel(CyclicCoordinateDescent *ccd, AbstractModelData *modelData,
                                          const IdType covariate,
                                          const std::vector<double>& points,
                                          std::vector<double>& values,
                                          std::vector<double>* derivatives,
                                          int inThreads,
                                          bool includePenalty,
                                          const std::vector<double>& startPoints,
                                          const std::vector<double>& startBetas,
                                          std::vector<double>* betas,
                                          bool chain) {

    struct timeval time1, time2;
    gettimeofday(&time1, NULL);

    int index = modelData->getColumnIndexByName(covariate);

    if (index == -1) {
        std::ostringstream stream;
        stream << "Variable " << covariate << " not found.";
        error->throwError(stream);
    }

    // Parallelize across grid values
    int nThreads = (inThreads == -1) ?
    bsccs::thread::hardware_concurrency() : inThreads;

    ccd->getLogLikelihood(); // TODO Remove

    std::ostringstream stream2;
    stream2 << "Using " << nThreads << " thread(s)";
    logger->writeLine(stream2);

    int J = ccd->getBetaSize();
    std::vector<double> x0s(J);
    for (int j = 0; j < J; ++j) {
        x0s[j] = ccd->getBeta(j);
    }

    bool doDerivative = derivatives != nullptr;

    ccd->makeDirty(); // Reset internal memory

    // Each point starts from the solution at the nearest of: the current coefficients, the supplied
    // starting solutions and (chain) the requested points between it and the current value. The choice
    // depends only on the locations, so the results do not depend on the number of threads.
    const int nPoints = points.size();
    const int nStarts = startPoints.size();
    const double center = x0s[index];

    std::vector<std::vector<double>> starts(nStarts);
    for (int s = 0; s < nStarts; ++s) {
        starts[s].assign(startBetas.begin() + s * J, startBetas.begin() + (s + 1) * J);
    }

    std::vector<double> sortedPoints(points);
    std::sort(sortedPoints.begin(), sortedPoints.end());

    std::vector<int> parent(nPoints); // -1: current; -2 - s: start s; >= 0: requested point
    std::vector<double> chainPoint(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        const double x = points[i];
        double best = std::abs(x - center);
        parent[i] = -1;
        for (int s = 0; s < nStarts; ++s) {
            const double d = std::abs(x - startPoints[s]);
            if (d < best) {
                best = d;
                parent[i] = -2 - s;
            }
        }
        if (chain) { // nearest requested point strictly between x and the current value
            double y = NAN;
            if (x > center) {
                auto it = std::lower_bound(sortedPoints.begin(), sortedPoints.end(), x);
                if (it != sortedPoints.begin() && *(it - 1) > center) y = *(it - 1);
            } else if (x < center) {
                auto it = std::upper_bound(sortedPoints.begin(), sortedPoints.end(), x);
                if (it != sortedPoints.end() && *it < center) y = *it;
            }
            if (!std::isnan(y) && std::abs(x - y) < best) {
                parent[i] = -3 - nStarts; // resolved below
                chainPoint[i] = y;
            }
        }
    }
    if (chain) { // map chain points to (the first) requested index with that location
        std::map<double, int> first;
        for (int i = 0; i < nPoints; ++i) {
            first.emplace(points[i], i);
        }
        for (int i = 0; i < nPoints; ++i) {
            if (parent[i] == -3 - nStarts) parent[i] = first[chainPoint[i]];
        }
    }

    // Evaluate in waves: a point waits for its parent; points within a wave are independent
    std::vector<int> order(nPoints);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&points, center](int a, int b) {
        return std::abs(points[a] - center) < std::abs(points[b] - center);
    });
    std::vector<int> wave(nPoints, 0);
    std::vector<int> children(nPoints, 0);
    int nWaves = 0;
    for (int i : order) { // a requested parent is strictly nearer the current value
        if (parent[i] >= 0) {
            wave[i] = wave[parent[i]] + 1;
            ++children[parent[i]];
        }
        nWaves = std::max(nWaves, wave[i] + 1);
    }
    std::vector<std::vector<int>> tasksByWave(nWaves);
    for (int i = 0; i < nPoints; ++i) {
        tasksByWave[wave[i]].push_back(i);
    }
    size_t maxWidth = 0;
    for (const auto& tasks : tasksByWave) {
        maxWidth = std::max(maxWidth, tasks.size());
    }
    nThreads = std::max(1, std::min(nThreads, static_cast<int>(maxWidth)));

    std::vector<std::vector<double>> solutions(nPoints);
    const bool keepAll = betas != nullptr;

    auto evaluate = [this, index, includePenalty, doDerivative, J, keepAll,
                     &x0s, &starts, &parent, &children, &solutions, &points, &values, &derivatives](
                         const int i, CyclicCoordinateDescent* ccd) {

        const int p = parent[i];
        ccd->setBeta(p == -1 ? x0s : (p < -1 ? starts[-2 - p] : solutions[p]));

        OptimizationProfile eval(*ccd, arguments, index,
                                 0.0, 0.0, includePenalty);

        values[i] = eval.objective(points[i]);
        if (doDerivative) {
            (*derivatives)[i] = eval.derivative();
        }
        if (keepAll || children[i] > 0) {
            solutions[i].resize(J);
            for (int j = 0; j < J; ++j) {
                solutions[i][j] = ccd->getBeta(j);
            }
        }
    };

    std::vector<CyclicCoordinateDescent*> ccdPool;
    ccdPool.push_back(ccd);

    for (int i = 1; i < nThreads; ++i) {
        CyclicCoordinateDescent* copy = ccd->clone(arguments.computeDevice);
        for (int j = 0; j < J; ++j) {
            copy->setFixedBeta(j, ccd->getFixedBeta(j));
        }
        ccdPool.push_back(copy);
    }

    for (const auto& tasks : tasksByWave) {
        if (nThreads == 1 || tasks.size() == 1) {
            for (int i : tasks) {
                evaluate(i, ccd);
            }
        } else {
            auto scheduler = TaskScheduler<IncrementableIterator<size_t>>(
                IncrementableIterator<size_t>(0),
                IncrementableIterator<size_t>(tasks.size()),
                nThreads);

            auto oneTask = [&evaluate, &scheduler, &ccdPool, &tasks](unsigned long task) {
                evaluate(tasks[task], ccdPool[scheduler.getThreadIndex(task)]);
            };

            // Run all tasks in parallel
            ccd->getProgressLogger().setConcurrent(true);
            ccd->getErrorHandler().setConcurrent(true);
            scheduler.execute(oneTask);
            ccd->getProgressLogger().setConcurrent(false);
            ccd->getErrorHandler().setConcurrent(false);
            ccd->getProgressLogger().flush();
            ccd->getErrorHandler().flush();
        }
        if (!keepAll) { // release solutions whose children are done
            for (int i : tasks) {
                if (parent[i] >= 0 && --children[parent[i]] == 0) {
                    std::vector<double>().swap(solutions[parent[i]]);
                }
            }
        }
    }

    if (keepAll) {
        betas->resize(static_cast<size_t>(J) * nPoints);
        for (int i = 0; i < nPoints; ++i) {
            std::copy(solutions[i].begin(), solutions[i].end(), betas->begin() + static_cast<size_t>(i) * J);
        }
    }

    // Reset exactly (incremental updates leave the coefficients off by rounding), and recompute
    // X beta now, as prediction reads it directly
    ccd->setBeta(x0s);
    ccd->getLogLikelihood();
    // DEBUG
    //double testMode = ccd->getLogLikelihood();
    //std::ostringstream stream;
    //stream << "Difference after profile: " << testMode << " " << mode;
    //logger->writeLine(stream);

    // Clean up copies
    for (int i = 1; i < nThreads; ++i) {
        delete ccdPool[i]; // TODO use shared_ptr
    }

    gettimeofday(&time2, NULL);
    return calculateSeconds(time1, time2);
}

double CcdInterface::diagnoseModel(CyclicCoordinateDescent *ccd, AbstractModelData *modelData,
		double loadTime,
		double updateTime) {

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

    diagnoseModelImpl(ccd, modelData, loadTime, updateTime);

	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

double CcdInterface::logModel(CyclicCoordinateDescent *ccd, AbstractModelData *modelData,
	    ProfileInformationMap& profileMap, bool withASE) {

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

    logModelImpl(ccd, modelData, profileMap, withASE);

	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

void CcdInterface::setZeroBetaAsFixed(CyclicCoordinateDescent *ccd) {
	for (int j = 0; j < ccd->getBetaSize(); ++j) {
		if (ccd->getBeta(j) == 0.0) {
			ccd->setFixedBeta(j, true);
		}
	}
}

double CcdInterface::fitModel(CyclicCoordinateDescent *ccd) {
	if (arguments.noiseLevel > SILENT) {
	    std::ostringstream stream;
		stream << "Using prior: " << ccd->getPriorInfo();
		logger->writeLine(stream);
	}

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	//Eric: Commented this out.
	//std::vector<double> weights(7, 1.0);
	//ccd->setWeights(weights.data());
	ccd->setThreads(arguments.threads == -1 ? bsccs::thread::hardware_concurrency() : arguments.threads);
	ccd->update(arguments.modeFinding);

	gettimeofday(&time2, NULL);

	return calculateSeconds(time1, time2);
}


namespace {

// Subject of each row as the exchangeable unit; one subject per row when none were loaded
std::vector<int> getSubjectIds(const AbstractModelData* modelData) {
	std::vector<int> ids = modelData->getSubjectIndexVectorRef();
	if (ids.empty()) {
		ids.resize(modelData->getNumberOfRows());
		std::iota(ids.begin(), ids.end(), 0);
	}
	return ids;
}

} // namespace

SelectorType CcdInterface::getDefaultSelectorTypeOrOverride(SelectorType selectorType, ModelType modelType) {
	if (selectorType == SelectorType::DEFAULT) {
		selectorType = (modelType == ModelType::COX ||
                        modelType == ModelType::COX_RAW ||
                        modelType == ModelType::COX_COUNTING) // TODO Fix for Normal, logistic, POISSON
			? SelectorType::BY_ROW : SelectorType::BY_PID;
// 				NORMAL,
// 				POISSON,
// 				LOGISTIC,
// 				CONDITIONAL_LOGISTIC,
// 				TIED_CONDITIONAL_LOGISTIC,
// 				CONDITIONAL_POISSON,
// 				SELF_CONTROLLED_MODEL,
// 				COX,
// 				COX_RAW,
	}
	return selectorType;
}

double CcdInterface::runBoostrap(
		CyclicCoordinateDescent *ccd,
		AbstractModelData *modelData,
		std::vector<double>& savedBeta,
		std::string& treatmentId) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	auto selectorType = getDefaultSelectorTypeOrOverride(
		arguments.crossValidation.selectorType, modelData->getModelType());

	vector<int> ids;
	if (selectorType == SelectorType::BY_ROW) {
		ids.resize(modelData->getNumberOfRows());
		std::iota(ids.begin(), ids.end(), 0);
	} else if (selectorType == SelectorType::BY_SUBJECT) { // resample subjects with all their rows
		ids = getSubjectIds(modelData);
		selectorType = SelectorType::BY_PID;
	} else {
		ids = modelData->getPidVectorSTL();
	}

	// Get possible weights
	std::vector<double> weights = ccd->getWeights();

	bool useWeights = false;
	for (auto& w : weights) {
		if (w != 1.0) {
			useWeights = true;
			break;
		}
	}

	BootstrapSelector selector(arguments.replicates, ids,
			selectorType, arguments.seed, logger, error,
			nullptr, (useWeights ? &weights : nullptr));
	BootstrapDriver driver(arguments.replicates, modelData, logger, error);

	driver.drive(*ccd, selector, arguments);
	gettimeofday(&time2, NULL);

//	driver.logResults(arguments, savedBeta, ccd->getConditionId());
	driver.logHR(arguments, savedBeta, treatmentId);
	return calculateSeconds(time1, time2);
}


double CcdInterface::runFitMLEAtMode(CyclicCoordinateDescent* ccd) {
    std::ostringstream stream;
	stream << std::endl << "Estimating MLE at posterior mode";
	logger->writeLine(stream);

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	setZeroBetaAsFixed(ccd);
	ccd->setPriorType(priors::NONE);
	fitModel(ccd);

	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

double CcdInterface::runCrossValidation(CyclicCoordinateDescent *ccd, AbstractModelData *modelData) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	auto selectorType = getDefaultSelectorTypeOrOverride(
		arguments.crossValidation.selectorType, modelData->getModelType());

	// Get possible weights
	std::vector<double> weights = ccd->getWeights();

	bool useWeights = false;
	for (auto& w : weights) {
		if (w != 1.0) {
			useWeights = true;
			break;
		}
	}

	// TODO ADD CODE HERE

	//arguments.crossValidation.fold != -1 ?

	std::vector<int> ids = modelData->getPidVectorSTL();
	if (selectorType == SelectorType::BY_SUBJECT) { // keep all rows of a subject in the same fold
		ids = getSubjectIds(modelData);
		selectorType = SelectorType::BY_PID;
	}

	AbstractSelector* ptr;
	if (arguments.crossValidation.fold != -1) {
	ptr = new CrossValidationSelector(arguments.crossValidation.fold,
                             ids,
                             selectorType, arguments.seed, logger, error,
                             nullptr,
                             (useWeights ? &weights : nullptr));
	} else {
	ptr =

     new WeightBasedSelector(1,
                             modelData->getPidVectorSTL(),
                             selectorType, arguments.seed, logger, error,
                             nullptr,
                             &weights);
	    arguments.crossValidation.foldToCompute = 1;
	}

	std::unique_ptr<AbstractSelector> selector =
	    bsccs::unique_ptr<AbstractSelector>(ptr);

	//    std::make_unique_ptr<AbstractSelector>(ptr2);

	AbstractCrossValidationDriver* driver;
	if (arguments.crossValidation.useAutoSearchCV) {
		if (arguments.useHierarchy) {
			driver = new HierarchyAutoSearchCrossValidationDriver(*modelData, arguments, logger, error);
		} else {
			driver = new AutoSearchCrossValidationDriver(*modelData, arguments, logger, error);
		}
	} else {
		if (arguments.useHierarchy) {
			driver = new HierarchyGridSearchCrossValidationDriver(arguments, logger, error);
		} else {
			driver = new GridSearchCrossValidationDriver(arguments, logger, error);
		}
	}

	ccd->setThreads(1); // folds run in parallel instead; fitModel() restores the threads
	driver->drive(*ccd, *selector, arguments);

	gettimeofday(&time2, NULL);

// 	driver->logResults(arguments);

	if (arguments.crossValidation.doFitAtOptimal) {
	    if (arguments.noiseLevel > SILENT) {
	        std::ostringstream stream;
	        stream << "Fitting model at optimal hyperparameter";
	        logger->writeLine(stream);
	    }
		// Do full fit for optimal parameter
		driver->resetForOptimal(*ccd, *selector, arguments);
		fitModel(ccd);
		if (arguments.fitMLEAtMode) {
			runFitMLEAtMode(ccd);
		}
	}
	delete driver;

	return calculateSeconds(time1, time2);
}

bool CcdInterface::includesOption(const string& line, const string& option) {
	size_t found = line.find(option);
	return found != string::npos;
}

// CmdLineCcdInterface::CmdLineCcdInterface(int argc, char* argv[]) {
//     std::vector<std::string> args;
// 	for (int i = 0; i < argc; i++)
// 		args.push_back(argv[i]);
// 	CcdInterface::parseCommandLine(args);
// }

// CmdLineCcdInterface::~CmdLineCcdInterface() {
//     // Do nothing
// }

// void CmdLineCcdInterface::parseCommandLine(int argc, char* argv[],
// 		CCDArguments &arguments) {
// 	std::vector<std::string> args;
// 	for (int i = 0; i < argc; i++)
// 		args.push_back(argv[i]);
// 	CcdInterface::parseCommandLine(args, arguments);
// }

} // namespace

