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
    class ModelData;
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
        selectorType(SelectorType::BY_PID)
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

	ModeFindingArguments() :
		tolerance(1E-6),
		maxIterations(1000),
		convergenceTypeString("gradient"),
		convergenceType(GRADIENT),
		useKktSwindle(false),
		swindleMultipler(10),
		initialBound(2.0),
		maxBoundCount(5)
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
	int replicates;
	std::string bsFileName;
	bool doPartial;

	// Needed for model specification
	int modelType;
	std::string modelName;

	NoiseLevels noiseLevel;

	ProfileVector profileCI;
	ProfileVector flatPrior;

	int threads;
	bool resetCoefficients;

	ModeFindingArguments modeFinding;
	CrossValidationArguments crossValidation;
};


class CcdInterface {

public:

    CcdInterface();
    virtual ~CcdInterface();

    double initializeModel(
            ModelData** modelData,
            CyclicCoordinateDescent** ccd,
            AbstractModelSpecifics** model);

    double fitModel(
            CyclicCoordinateDescent *ccd);

    double runFitMLEAtMode(
            CyclicCoordinateDescent* ccd);

    double predictModel(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData);

    double profileModel(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData,
            const ProfileVector& profileCI,
            ProfileInformationMap &profileMap,
            int threads,
            double threshold = 1.920729,
            bool overrideNoRegularization = false,
            bool includePenalty = false);

    double runCrossValidation(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData);

    double runBoostrap(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData,
            std::vector<double>& savedBeta);

    void setDefaultArguments();

    void setZeroBetaAsFixed(
            CyclicCoordinateDescent *ccd);

    double logModel(CyclicCoordinateDescent *ccd, ModelData *modelData,
            ProfileInformationMap &profileMap,
            bool withProfileBounds);

    double diagnoseModel(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData,
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
            ModelData** modelData,
            CyclicCoordinateDescent** ccd,
            AbstractModelSpecifics** model) = 0;

    virtual void predictModelImpl(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData) = 0;

    virtual void logModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData,
            ProfileInformationMap &profileMap,
            bool withProfileBounds) = 0;

    virtual void diagnoseModelImpl(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData,
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
