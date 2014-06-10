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
#include "engine/ModelSpecifics.h"

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

//Sushil:Implementing gettimeofday functionality for windows.
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
	arguments.maxIterations = 1000;
	arguments.inFileName = "default_in";
	arguments.outFileName = "default_out";
	arguments.outDirectoryName = "";
	arguments.hyperPriorSet = false;
	arguments.hyperprior = 1.0;
	arguments.tolerance = 1E-6; //5E-4;
	arguments.seed = 123;
	arguments.doCrossValidation = false;
	arguments.useAutoSearchCV = false;
	arguments.lowerLimit = 0.01;
	arguments.upperLimit = 20.0;
	arguments.fold = 10;
	arguments.gridSteps = 10;
	arguments.cvFileName = "cv.txt";
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
	arguments.convergenceType = GRADIENT;
	arguments.convergenceTypeString = "gradient";
	arguments.doPartial = false;
	arguments.noiseLevel = NOISY;
}

double CcdInterface::initializeModel(
		ModelData** modelData,
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

double CcdInterface::predictModel(CyclicCoordinateDescent *ccd, ModelData *modelData) {

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	
    predictModelImpl(ccd, modelData);
    
    gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

struct OptimizationProfile {

	CyclicCoordinateDescent& ccd;

	OptimizationProfile(CyclicCoordinateDescent& _ccd, int _index, double _max,
			double _threshold = 1.92) :
			ccd(_ccd), index(_index), max(_max), threshold(_threshold), nEvals(0) {
	}

	int getEvaluations() {
		return nEvals;
	}

	double objective(double x) {
		++nEvals;
		ccd.setBeta(index, x);
		return ccd.getLogLikelihood() + threshold - max;
	}

	double getMaximum() {
		return threshold;
	}

	int index;
	double max;
	double threshold;
	int nEvals;
};

double CcdInterface::profileModel(CyclicCoordinateDescent *ccd, ModelData *modelData,
		ProfileInformationMap& profileMap) {

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	// Attempt profile CIs
	for (std::vector<DrugIdType>::iterator it = arguments.profileCI.begin();
			it != arguments.profileCI.end(); ++it) {
		int index = modelData->getColumnIndexByName(*it);
		if (index == -1) {
		    std::ostringstream stream;
			stream << "Variable " << *it << " not found.";
			error->throwError(stream);
		} else {
			// TODO : Minor bug, order of column evaluation yields different estimates
			double mode = ccd->getLogLikelihood();

			// TODO Check prior on covariate

			// Bound edge
			OptimizationProfile upEval(*ccd, index, mode);
			RZeroIn<OptimizationProfile> zeroIn(upEval);

			double x0 = ccd->getBeta(index);
			double obj0 = upEval.getMaximum();

			RZeroIn<OptimizationProfile>::Coordinate upperBracket =
					zeroIn.bracketSignChange(x0, obj0, 1.0);
			double upperPt = zeroIn.getRoot(x0, upperBracket.first, obj0, upperBracket.second);

			RZeroIn<OptimizationProfile>::Coordinate lowerBracket =
					zeroIn.bracketSignChange(x0, obj0, -1.0);
			double lowerPt = zeroIn.getRoot(x0, lowerBracket.first, obj0, lowerBracket.second);

            std::ostringstream stream;
			stream << "Profile: " << modelData->getColumn(index).getLabel() << " (" << lowerPt << ", "
					<< upperPt << ")  in " << upEval.getEvaluations();
			logger->writeLine(stream);

			ProfileInformation profile(lowerPt, upperPt);
			profileMap.insert(std::pair<int,ProfileInformation>(index, profile));

			// Reset beta[index] to value at mode
			ccd->setBeta(index, x0);
		}
	}

	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

double CcdInterface::diagnoseModel(CyclicCoordinateDescent *ccd, ModelData *modelData,	
		double loadTime,
		double updateTime) {
		
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
			
    diagnoseModelImpl(ccd, modelData, loadTime, updateTime);	
    
	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);    	
}

double CcdInterface::logModel(CyclicCoordinateDescent *ccd, ModelData *modelData,
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

	ccd->update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);

	gettimeofday(&time2, NULL);

	return calculateSeconds(time1, time2);
}

double CcdInterface::runBoostrap(
		CyclicCoordinateDescent *ccd,
		ModelData *modelData,
		std::vector<double>& savedBeta) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	BootstrapSelector selector(arguments.replicates, modelData->getPidVectorSTL(),
			SUBJECT, arguments.seed);
	BootstrapDriver driver(arguments.replicates, modelData);

	driver.drive(*ccd, selector, arguments);
	gettimeofday(&time2, NULL);

	driver.logResults(arguments, savedBeta, ccd->getConditionId());
	return calculateSeconds(time1, time2);
}


double CcdInterface::runFitMLEAtMode(CyclicCoordinateDescent* ccd) {
	std::cout << std::endl << "Estimating MLE at posterior mode" << std::endl;

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	setZeroBetaAsFixed(ccd);
	ccd->setPriorType(priors::NONE);
	fitModel(ccd);

	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

double CcdInterface::runCrossValidation(CyclicCoordinateDescent *ccd, ModelData *modelData) {
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	CrossValidationSelector selector(arguments.fold, modelData->getPidVectorSTL(),
			SUBJECT, arguments.seed);

	AbstractCrossValidationDriver* driver;
	if (arguments.useAutoSearchCV) {
		if (arguments.useHierarchy) {
			driver = new HierarchyAutoSearchCrossValidationDriver(*modelData, arguments.gridSteps, arguments.lowerLimit, arguments.upperLimit);
		} else {
			driver = new AutoSearchCrossValidationDriver(*modelData, arguments.gridSteps, arguments.lowerLimit, arguments.upperLimit);
		}
	} else {
		if (arguments.useHierarchy) {
			driver = new HierarchyGridSearchCrossValidationDriver(arguments.gridSteps, arguments.lowerLimit, arguments.upperLimit);
		} else {
			driver = new GridSearchCrossValidationDriver(arguments.gridSteps, arguments.lowerLimit, arguments.upperLimit);
		}
	}

	driver->drive(*ccd, selector, arguments);

	gettimeofday(&time2, NULL);

	driver->logResults(arguments);

	if (arguments.doFitAtOptimal) {
		std::cout << "Fitting model at optimal hyperparameter" << std::endl;
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

