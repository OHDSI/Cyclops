/*
 * ccd.h
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#ifndef CCD_H_
#define CCD_H_

#include <time.h>

#ifdef _WIN32
#include <stddef.h>
#include <io.h>
#include <stdlib.h>
#include <winsock.h>
#include <stdio.h>
#else
#include <sys/time.h>
#endif

#include "Types.h"

namespace bsccs {

    class CyclicCoordinateDescent; // forward declaration
    class ModelData;
    class AbstractModelSpecifics;

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
	double tolerance;
	double hyperprior;
	bool computeMLE;
	bool fitMLEAtMode;
	bool reportASE;
	bool useNormalPrior;
	bool hyperPriorSet;
	int maxIterations;
	std::string convergenceTypeString;
	int convergenceType;
	long seed;

	// Needed for cross-validation
	bool doCrossValidation;
	bool useAutoSearchCV;
	double lowerLimit;
	double upperLimit;
	int fold;
	int foldToCompute;
	int gridSteps;
	std::string cvFileName;
	bool doFitAtOptimal;

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
            ProfileInformationMap &profileMap);

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
            
    CCDArguments getArguments() {
        return arguments;  // TODO To depricate
    }		
    		
protected:
    std::string getPathAndFileName(const CCDArguments& arguments, std::string stem);
    bool includesOption(const std::string& line, const std::string& option);
    
    CCDArguments arguments;    
    
    double calculateSeconds(
		const struct timeval &time1,
		const struct timeval &time2);
		
    virtual double initializeModelImpl(
            ModelData** modelData,
            CyclicCoordinateDescent** ccd,
            AbstractModelSpecifics** model) = 0;
            
    virtual double predictModelImpl(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData) = 0;  
            
    virtual double logModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData,        
            ProfileInformationMap &profileMap,
            bool withProfileBounds) = 0;  
            
    virtual double diagnoseModelImpl(CyclicCoordinateDescent *ccd, ModelData *modelData,	
		double loadTime,
		double updateTime) = 0;                                 
		

}; // class CcdInterface

class RCcdInterface: public CcdInterface {

}; // class RCcdInterface

} // namespace

#endif /* CCD_H_ */
