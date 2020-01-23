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

#include "boost/iterator/counting_iterator.hpp"
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
	arguments.modelName = "sccs";
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

	double getMaximum() {
		return threshold;
	}

	int index;
	double max;
	double threshold;
	int nEvals;
	bool includePenalty;
};

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
			// TODO Minor bug, order of column evaluation yields different estimates
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

	// Parallelize across columns and lower/upper bound
	int nThreads = (inThreads == -1) ?
	    bsccs::thread::hardware_concurrency() : inThreads;

	std::ostringstream stream2;
	stream2 << "Using " << nThreads << " thread(s)";
	logger->writeLine(stream2);

	std::vector<CyclicCoordinateDescent*> ccdPool;

	ccdPool.push_back(ccd);

	for (int i = 1; i < nThreads; ++i) {
	    ccdPool.push_back(ccd->clone());
	}

    std::vector<double> lowerPts(indices.size());
	std::vector<double> upperPts(indices.size());
	std::vector<int> lowerCnts(indices.size());
	std::vector<int> upperCnts(indices.size());

	auto getBound = [this,
	            &x0s,
	            &indices, &lowerPts, &upperPts,
                &lowerCnts, &upperCnts, includePenalty, mode, threshold
            ](const BoundType bound, CyclicCoordinateDescent* ccd) {
	    const auto id = std::get<0>(bound);
	    const auto direction = std::get<1>(bound);
	    const auto index = indices[id];

	    double x0 = x0s[index];

	    // Bound edge
	    OptimizationProfile eval(*ccd, arguments, index, mode, threshold, includePenalty);
	    RZeroIn<OptimizationProfile> zeroIn(eval, 1E-3);

	    double obj0 = eval.getMaximum();

	    //		std::cout << "BEGIN UP bracket" << std::endl;
	    RZeroIn<OptimizationProfile>::Coordinate bracket =
	        zeroIn.bracketSignChange(x0, obj0, direction);
	     //		std::cout << "END UP bracket " << bracket.first << " " << bracket.second << std::endl;

	    double pt = std::isnan(bracket.second) ? NAN : zeroIn.getRoot(x0, bracket.first, obj0, bracket.second);

	    if (direction == 1.0) {
	        upperPts[id] = pt;
	        upperCnts[id] = eval.getEvaluations();
	    } else { // direction == -1.0
	        lowerPts[id] = pt;
	        lowerCnts[id] = eval.getEvaluations();
	    }
	};

    if (nThreads == 1) {
        std::for_each(std::begin(bounds), std::end(bounds),
                      [&getBound, ccd](const BoundType bound) {
                        getBound(bound, ccd); // Use main CCD object
                      }
                    );
    } else {
        auto scheduler = TaskScheduler<boost::counting_iterator<int> >(
            boost::make_counting_iterator(0),
            boost::make_counting_iterator(static_cast<int>(bounds.size())),
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
                                          int inThreads,
                                          bool includePenalty) {

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

    double mode = ccd->getLogLikelihood(); // TODO Remove

    std::ostringstream stream2;
    stream2 << "Using " << nThreads << " thread(s)";
    logger->writeLine(stream2);

    int J = ccd->getBetaSize();
    std::vector<double> x0s(J);
    for (int j = 0; j < J; ++j) {
        x0s[j] = ccd->getBeta(j);
    }

    std::vector<CyclicCoordinateDescent*> ccdPool;
    ccdPool.push_back(ccd);

    for (int i = 1; i < nThreads; ++i) {
        ccdPool.push_back(ccd->clone());
    }

    auto evaluate = [this, index, includePenalty](const double point,
                            CyclicCoordinateDescent* ccd) {

        OptimizationProfile eval(*ccd, arguments, index,
                                 0.0, 0.0, includePenalty);

        return eval.objective(point);
    };

    if (nThreads == 1) {
        for (int i = 0; i < points.size(); ++i) {
            values[i] = evaluate(points[i], ccd);
        }
    } else {
        auto scheduler = TaskScheduler<boost::counting_iterator<int>>(
            boost::make_counting_iterator(0), boost::make_counting_iterator(static_cast<int>(points.size())), nThreads);

        auto oneTask = [&evaluate, &scheduler, &ccdPool, &points, &values](unsigned long task) {
            values[task] = evaluate(points[task], ccdPool[scheduler.getThreadIndex(task)]);
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

    // Reset
    for (int j = 0; j < J; ++j) {
        ccd->setBeta(j, x0s[j]);
    }
    // DEBUG, TODO Remove?
    double testMode = ccd->getLogLikelihood();
    std::ostringstream stream;
    stream << "Difference after profile: " << testMode << " " << mode;
    logger->writeLine(stream);

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

	ccd->update(arguments.modeFinding);

	gettimeofday(&time2, NULL);

	return calculateSeconds(time1, time2);
}


SelectorType CcdInterface::getDefaultSelectorTypeOrOverride(SelectorType selectorType, ModelType modelType) {
	if (selectorType == SelectorType::DEFAULT) {
		selectorType = (modelType == ModelType::COX ||
                        modelType == ModelType::COX_RAW) // TODO Fix for Normal, logistic, POISSON
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
		std::vector<double>& savedBeta) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	auto selectorType = getDefaultSelectorTypeOrOverride(
		arguments.crossValidation.selectorType, modelData->getModelType());

	BootstrapSelector selector(arguments.replicates, modelData->getPidVectorSTL(),
			selectorType, arguments.seed, logger, error);
	BootstrapDriver driver(arguments.replicates, modelData, logger, error);

	driver.drive(*ccd, selector, arguments);
	gettimeofday(&time2, NULL);

	driver.logResults(arguments, savedBeta, ccd->getConditionId());
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

	CrossValidationSelector selector(arguments.crossValidation.fold,
	 		modelData->getPidVectorSTL(),
			selectorType, arguments.seed, logger, error,
			nullptr,
			(useWeights ? &weights : nullptr)
			); // TODO ERROR HERE!  NOT ALL MODELS ARE SUBJECT

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

	driver->drive(*ccd, selector, arguments);

	gettimeofday(&time2, NULL);

// 	driver->logResults(arguments);

	if (arguments.crossValidation.doFitAtOptimal) {
	    if (arguments.noiseLevel > SILENT) {
	        std::ostringstream stream;
	        stream << "Fitting model at optimal hyperparameter";
	        logger->writeLine(stream);
	    }
		// Do full fit for optimal parameter
		driver->resetForOptimal(*ccd, selector, arguments);
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

