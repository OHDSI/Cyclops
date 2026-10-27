/*
 * ccd.h
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#ifndef CCD_H_
#define CCD_H_

// #include <time.h>

// #ifdef _WIN32
// #include <stddef.h>
// #include <io.h>
// #include <stdlib.h>
// #include <winsock.h>
// #include <stdio.h>
// #else
// #include <sys/time.h>
// #endif

#include "Types.h"
#include "io/ProgressLogger.h"

namespace bsccs {

    class CyclicCoordinateDescent; // forward declaration
    class AbstractModelData;
    class AbstractModelSpecifics;

struct CrossValidationArguments {

    // All options related to cross-validation go here
	bool doCrossValidation;
	bool useAutoSearchCV;
	double lowerLimit;
	double upperLimit;
	int fold;
	int foldToCompute;
	int gridSteps;
	std::string cvFileName;
	bool doFitAtOptimal;
    double startingVariance;
    SelectorType selectorType;
    bool syncCV;
    int searchDirection;        // auto-search: 0 = up below variance 1 and down above (BBR), 1 = up, -1 = down
    int foldConvergenceType;    // fold fits only; -1 = as the final fit
    double foldTolerance;       // fold fits only; <= 0 = as the final fit
    int foldMaxIterations;      // fold fits only; <= 0 = as the final fit
    int foldMinIterations;      // fold fits only, when one of the three above is set
    bool warmStart;             // each fold starts from its own solution at the nearest hyperparameter tried
    bool refineSearch;          // auto-search: refine the result by local quadratic fits around the best point

    CrossValidationArguments() :
        doCrossValidation(false),
        useAutoSearchCV(false),
        lowerLimit(0.01),
        upperLimit(20.0),
        fold(10),
        foldToCompute(10),
        gridSteps(10),
        cvFileName("cv.txt"),
        doFitAtOptimal(true),
        startingVariance(-1),   // Use default from Genkins et al.
        selectorType(SelectorType::BY_PID),
		syncCV(false),
		searchDirection(0),
		foldConvergenceType(-1),
		foldTolerance(-1.0),
		foldMaxIterations(-1),
		foldMinIterations(5),
		warmStart(false),
		refineSearch(false)
        { }
};

struct ComputeDeviceArguments {
    std::string name;

    ComputeDeviceArguments() :
        name("native")
    { }
};

struct ModeFindingArguments {

	// All options related to mode-finding should (TODO) go here
	double tolerance;
	int maxIterations;
    std::string convergenceTypeString;
	int convergenceType;
	bool useKktSwindle;
	int swindleMultipler;
	double initialBound;
	int maxBoundCount;
	bool doItAll;
	AlgorithmType algorithmType;
	int activeSetEvery = 0; // 0: every cycle visits all coordinates; see CyclicCoordinateDescent::update()
	double activeSetThreshold = 0.01; // active-set cycling: relative step size below which a coordinate is skipped
	bool adaptiveIntercept = false; // extra intercept steps; see CyclicCoordinateDescent::trackInterceptGradient()
	int minIterations; // no convergence (SUCCESS) before this many cycles

	ModeFindingArguments() :
		tolerance(1E-6),
		maxIterations(1000),
		convergenceTypeString("gradient"),
		convergenceType(GRADIENT),
		useKktSwindle(false),
		swindleMultipler(10),
		initialBound(2.0),
		maxBoundCount(5),
		algorithmType(AlgorithmType::CCD),
		minIterations(0)
	    { }
};

struct CCDArguments {

	// Needed for fitting
	std::string inFileName;
	std::string outFileName;
	std::string fileFormat;
	std::string outDirectoryName;
	std::vector<std::string> outputFormat;
	bool useGPU;
	bool useBetterGPU;
	int deviceNumber;
// 	double tolerance;
	double hyperprior;
	bool computeMLE;
	bool fitMLEAtMode;
	bool reportASE;
	bool useNormalPrior;
	bool hyperPriorSet;
// 	int maxIterations;
// 	std::string convergenceTypeString;
// 	int convergenceType;
	long seed;

	//Needed for Hierarchy
	bool useHierarchy;
	std::string hierarchyFileName; //tshaddox
	double classHierarchyVariance; //tshaddox

	// Needed for boot-strapping
	bool doBootstrap;
	bool reportRawEstimates;
	bool reportDifference;
	int replicates;
	std::string bsFileName;
	bool doPartial;

	// Needed for model specification
	int modelType;
	std::string modelName;
	std::string fitter; // "legacy" (Cyclops 3.7 kernels), "default" or "super", from createControl(fitter = )

	NoiseLevels noiseLevel;

	ProfileVector profileCI;
	ProfileVector flatPrior;

	int threads;
	bool resetCoefficients;

	ModeFindingArguments modeFinding;
	CrossValidationArguments crossValidation;
	ComputeDeviceArguments computeDevice;
};


class CcdInterface {

public:

    CcdInterface();
    virtual ~CcdInterface();

    double initializeModel(
            AbstractModelData** modelData,
            CyclicCoordinateDescent** ccd,
            AbstractModelSpecifics** model);

    double fitModel(
            CyclicCoordinateDescent *ccd);

    double runFitMLEAtMode(
            CyclicCoordinateDescent* ccd);

    double predictModel(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData);

    double profileModel(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData,
            const ProfileVector& profileCI,
            ProfileInformationMap &profileMap,
            int threads,
            double threshold = 1.920729,
            bool overrideNoRegularization = false,
            bool includePenalty = false);

    double evaluateProfileModel(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData,
            const IdType covariate,
            const std::vector<double>& points,
            std::vector<double>& values,
            std::vector<double>* derivatives,
            int threads,
            bool includePenalty,
            const std::vector<double>& startPoints,
            const std::vector<double>& startBetas,
            std::vector<double>* betas,
            bool chain);

    void evaluateProfileModelLegacy(
            CyclicCoordinateDescent *ccd,
            int index,
            const std::vector<double>& x0s,
            const std::vector<double>& points,
            std::vector<double>& values,
            std::vector<double>* derivatives,
            int nThreads,
            bool includePenalty);

    double runCrossValidation(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData);

    double runBoostrap(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData,
            std::vector<double>& savedBeta,
	    std::string& treatmentId);

    void setDefaultArguments();

    void setZeroBetaAsFixed(
            CyclicCoordinateDescent *ccd);

    double logModel(CyclicCoordinateDescent *ccd, AbstractModelData *modelData,
            ProfileInformationMap &profileMap,
            bool withProfileBounds);

    double diagnoseModel(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData,
            double loadTime,
            double updateTime);

//     void parseCommandLine(
//             std::vector<std::string>& argcpp);

    CCDArguments& getArguments() {
        return arguments;  // TODO To depricate
    }

    static double calculateSeconds(
		const struct timeval &time1,
		const struct timeval &time2);

protected:
    std::string getPathAndFileName(const CCDArguments& arguments, std::string stem);
    bool includesOption(const std::string& line, const std::string& option);

	static SelectorType getDefaultSelectorTypeOrOverride(SelectorType selectorType, ModelType modelType);

    CCDArguments arguments;

    virtual void initializeModelImpl(
            AbstractModelData** modelData,
            CyclicCoordinateDescent** ccd,
            AbstractModelSpecifics** model) = 0;

    virtual void predictModelImpl(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData) = 0;

    virtual void logModelImpl(CyclicCoordinateDescent *ccd, AbstractModelData *modelData,
            ProfileInformationMap &profileMap,
            bool withProfileBounds) = 0;

    virtual void diagnoseModelImpl(
            CyclicCoordinateDescent *ccd,
            AbstractModelData *modelData,
    		double loadTime,
    		double updateTime) = 0;

    loggers::ProgressLoggerPtr logger;
    loggers::ErrorHandlerPtr error;

}; // class CcdInterface

// class RCcdInterface: public CcdInterface {
//
// }; // class RCcdInterface

} // namespace

#endif /* CCD_H_ */
