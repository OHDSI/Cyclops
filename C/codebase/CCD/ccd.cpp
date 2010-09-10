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
#include <sys/time.h>

#include <math.h>

#include "CyclicCoordinateDescent.h"
#include "InputReader.h"
#include "CrossValidation.h"

#include "tclap/CmdLine.h"

#ifdef CUDA
	#include "GPUCyclicCoordinateDescent.h"
#endif


#define NEW

using namespace TCLAP;
using namespace std;

struct CCDArguments {

	// Needed for fitting
	std::string inFileName;
	std::string outFileName;
	bool useGPU;
	int deviceNumber;
	double tolerance;
	double hyperprior;
	bool useNormalPrior;
	bool hyperPriorSet;
	int maxIterations;
	int convergenceType;

	// Needed for cross-validation

	// Needed for boot-strapping
};

void parseCommandLine(int argc, char* argv[], CCDArguments &arguments) {

	try {
		CmdLine cmd("Cyclic coordinate descent algorithm for self-controlled case studies", ' ', "0.1");
		ValueArg<int> gpuArg("g","GPU","Use GPU device", false, -1, "device #");
		ValueArg<int> maxIterationsArg("i", "iterations", "Maximum iterations", false, 100, "int");
		UnlabeledValueArg<string> inFileArg("inFileName","Input file name", true, "default", "inFileName");
		UnlabeledValueArg<string> outFileArg("outFileName","Output file name", true, "default", "outFileName");

		// Prior arguments
		ValueArg<double> hyperPriorArg("v", "variance", "Hyperprior variance", false, 1.0, "real");
		SwitchArg normalPriorArg("n", "normalPrior", "Use normal prior, default is laplace", false);

		// Convergence criterion arguments
		ValueArg<double> toleranceArg("t", "tolerance", "Convergence criterion tolerance", false, 1E-4, "real");
		SwitchArg zhangOlesConvergenceArg("z", "zhangOles", "Use Zhange-Oles convergence criterion, default is false", false);

		cmd.add(gpuArg);
		cmd.add(toleranceArg);
		cmd.add(maxIterationsArg);
		cmd.add(hyperPriorArg);
		cmd.add(normalPriorArg);
		cmd.add(zhangOlesConvergenceArg);

		cmd.add(inFileArg);
		cmd.add(outFileArg);
		cmd.parse(argc, argv);

		if (gpuArg.getValue() > -1) {
			arguments.useGPU = true;
			arguments.deviceNumber = gpuArg.getValue();
		} else {
			arguments.useGPU = false;
		}

		arguments.inFileName = inFileArg.getValue();
		arguments.outFileName = outFileArg.getValue();
		arguments.tolerance = toleranceArg.getValue();
		arguments.maxIterations = maxIterationsArg.getValue();
		arguments.hyperprior = hyperPriorArg.getValue();
		arguments.useNormalPrior = normalPriorArg.getValue();

		if (hyperPriorArg.isSet()) {
			arguments.hyperPriorSet = true;
		} else {
			arguments.hyperPriorSet = false;
		}

		if (zhangOlesConvergenceArg.isSet()) {
			arguments.convergenceType = ZHANG_OLES;
		} else {
			arguments.convergenceType = LANGE;
		}
	} catch (ArgException &e) {
		cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
		exit(-1);
	}
}

int main(int argc, char* argv[]) {
	
	CyclicCoordinateDescent* ccd;

	CCDArguments arguments;
	parseCommandLine(argc, argv, arguments);

	cout << "Running CCD (" <<
#ifdef DOUBLE_PRECISION
	"double"
#else
	"single"
#endif
	"-precision) ..." << endl;

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);

	InputReader* reader = new InputReader(arguments.inFileName.c_str());

#ifdef CUDA
	if (arguments.useGPU) {
		ccd = new GPUCyclicCoordinateDescent(arguments.deviceNumber, reader);
	} else {
#endif

	ccd = new CyclicCoordinateDescent(reader);

#ifdef CUDA
	}
#endif

	// Set prior from the command-line
	if (arguments.useNormalPrior) {
		ccd->setPriorType(NORMAL);
	}
	if (arguments.hyperPriorSet) {
		ccd->setHyperprior(arguments.hyperprior);
	}

	cout << "Using prior: " << ccd->getPriorInfo() << endl;
	cout << "Everything loaded and ready to run ..." << endl;

	gettimeofday(&time2, NULL);	
	double sec1 = time2.tv_sec - time1.tv_sec +
		(double)(time2.tv_usec - time1.tv_usec) / 1000000.0;
	
	gettimeofday(&time1, NULL);

	cout << "Playing with weights..." << endl;

	int fold = 30;

	CrossValidation crossValidation(fold, reader->getPidVectorSTL(), SUBJECT);
	std::vector<real> weights;
	crossValidation.getWeights(0, weights);
	ccd->setWeights(&weights[0]);

	ccd->update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);

	gettimeofday(&time2, NULL);	
	double sec2 = time2.tv_sec - time1.tv_sec +
		(double)(time2.tv_usec - time1.tv_usec) / 1000000.0;
		
	cout << "Load   duration: " << scientific << sec1 << endl;
	cout << "Update duration: " << scientific << sec2 << endl;
	
	ccd->logResults(arguments.outFileName.c_str());

	delete ccd;

    return 0;
}
