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

#include "CmdLineCcdInterface.h"
#include "CyclicCoordinateDescent.h"
#include "drivers/ProportionSelector.h"
#include "io/CmdLineProgressLogger.h"

using std::cout;

int main(int argc, char* argv[]) {

	using namespace bsccs;

	CyclicCoordinateDescent* ccd = NULL;
	AbstractModelSpecifics* model = NULL;
	AbstractModelData* modelData = NULL;
// 	CCDArguments arguments;
	
	CmdLineCcdInterface interface(argc, argv);
	
	CCDArguments arguments = interface.getArguments();

// 	interface.parseCommandLine(argc, argv, arguments);

	double timeInitialize = interface.initializeModel(&modelData, &ccd, &model);

	double timeUpdate;
	if (arguments.crossValidation.doCrossValidation) {
		timeUpdate = interface.runCrossValidation(ccd, modelData);
	} else {
		if (arguments.doPartial) {
		    // TODO Delegate to CcdInterface
			ProportionSelector selector(arguments.replicates, modelData->getPidVectorSTL(),
					SelectorType::BY_PID, arguments.seed,
					bsccs::make_shared<loggers::CoutLogger>(),
					bsccs::make_shared<loggers::CerrErrorHandler>());
			std::vector<double> weights;
			selector.getWeights(0, weights);
			ccd->setWeights(&weights[0]);
		}
		timeUpdate = interface.fitModel(ccd);
		if (arguments.fitMLEAtMode) {
			timeUpdate += interface.runFitMLEAtMode(ccd);
		}
	}
	
	double timeProfile;
	bool doProfile = false;
	bsccs::ProfileInformationMap profileMap;
	if (arguments.profileCI.size() > 0) {
		doProfile = true;
		timeProfile = interface.profileModel(ccd, modelData, arguments.profileCI, profileMap,
			1 /* threads */);
	}	

	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "estimates")
			!= arguments.outputFormat.end()) {
#ifndef MY_RCPP_FLAG
		// TODO Make into OutputWriter
		bool withASE = arguments.fitMLEAtMode || arguments.computeMLE || arguments.reportASE;    
		interface.logModel(ccd, modelData, profileMap, withASE);
#endif
	}

	double timePredict;
	bool doPrediction = false;
	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "prediction")
			!= arguments.outputFormat.end()) {
		doPrediction = true;
		timePredict = interface.predictModel(ccd, modelData);
	}

	double timeDiagnose;
	bool doDiagnosis = false;
	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "diagnostics")
			!= arguments.outputFormat.end()) {
		doDiagnosis = true;
		timeDiagnose = interface.diagnoseModel(ccd, modelData, timeInitialize, timeUpdate);
	}

	if (arguments.doBootstrap) {
		// Save parameter point-estimates
		std::vector<double> savedBeta;
		for (int j = 0; j < ccd->getBetaSize(); ++j) {
			savedBeta.push_back(ccd->getBeta(j));
		} // TODO Handle above work in interface.runBootstrap
		timeUpdate += interface.runBoostrap(ccd, modelData, savedBeta);
	}
		
	using std::scientific;
		
	cout << endl;
	cout << "Load    duration: " << scientific << timeInitialize << endl;
	cout << "Update  duration: " << scientific << timeUpdate << endl;
	if (doPrediction) {
		cout << "Predict duration: " << scientific << timePredict << endl;
	}
	
	if (doDiagnosis) {
		cout << "Diag    duration: " << scientific << timeDiagnose << endl;
	}

//#define PRINT_LOG_LIKELIHOOD
#ifdef PRINT_LOG_LIKELIHOOD
	cout << endl << setprecision(15) << ccd->getLogLikelihood() << endl;
#endif

	if (ccd)
		delete ccd;
	if (model)
		delete model;
	if (modelData)
		delete modelData;

    return 0;
}
