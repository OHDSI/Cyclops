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

#include "CcdInterface.h"
#include "drivers/ProportionSelector.h"

using std::cout;

int main(int argc, char* argv[]) {

	using namespace bsccs;

	CyclicCoordinateDescent* ccd = NULL;
	AbstractModelSpecifics* model = NULL;
	ModelData* modelData = NULL;
	CCDArguments arguments;

	parseCommandLine(argc, argv, arguments);

	double timeInitialize = initializeModel(&modelData, &ccd, &model, arguments);

	double timeUpdate;
	if (arguments.doCrossValidation) {
		timeUpdate = runCrossValidation(ccd, modelData, arguments);
	} else {
		if (arguments.doPartial) {
			ProportionSelector selector(arguments.replicates, modelData->getPidVectorSTL(),
					SUBJECT, arguments.seed);
			std::vector<bsccs::real> weights;
			selector.getWeights(0, weights);
			ccd->setWeights(&weights[0]);
		}
		timeUpdate = fitModel(ccd, arguments);
		if (arguments.fitMLEAtMode) {
			timeUpdate += runFitMLEAtMode(ccd, arguments);
		}
	}
	
	double timeProfile;
	bool doProfile = false;
	bsccs::ProfileInformationMap profileMap;
	if (arguments.profileCI.size() > 0) {
		doProfile = true;
		timeProfile = profileModel(ccd, modelData, arguments, profileMap);
	}	

	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "estimates")
			!= arguments.outputFormat.end()) {
#ifndef MY_RCPP_FLAG
		// TODO Make into OutputWriter
		bool withASE = arguments.fitMLEAtMode || arguments.computeMLE || arguments.reportASE;    
		logModel(ccd, modelData, arguments, profileMap, withASE);
#endif
	}

	double timePredict;
	bool doPrediction = false;
	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "prediction")
			!= arguments.outputFormat.end()) {
		doPrediction = true;
		timePredict = predictModel(ccd, modelData, arguments);
	}

	double timeDiagnose;
	bool doDiagnosis = false;
	if (std::find(arguments.outputFormat.begin(),arguments.outputFormat.end(), "diagnostics")
			!= arguments.outputFormat.end()) {
		doDiagnosis = true;
		timeDiagnose = diagnoseModel(ccd, modelData, arguments, timeInitialize, timeUpdate);
	}

	if (arguments.doBootstrap) {
		// Save parameter point-estimates
		std::vector<bsccs::real> savedBeta;
		for (int j = 0; j < ccd->getBetaSize(); ++j) {
			savedBeta.push_back(ccd->getBeta(j));
		}
		timeUpdate += runBoostrap(ccd, modelData, arguments, savedBeta);
	}
		
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
