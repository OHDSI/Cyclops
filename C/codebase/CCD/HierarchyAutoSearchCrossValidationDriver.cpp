/*
 * HierarchyAutoSearchCrossValidationDriver.cpp
 *
 *  Created on: April 10, 2014
 *      Author: Trevor Shaddox
 */

#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>

#include "HierarchyAutoSearchCrossValidationDriver.h"
#include "AutoSearchCrossValidationDriver.h"
#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "ccd.h"
#include "../utils/HParSearch.h"

namespace bsccs {

const static int MAX_STEPS = 50;

HierarchyAutoSearchCrossValidationDriver::HierarchyAutoSearchCrossValidationDriver(const ModelData& _modelData,
		int iGridSize,
		double iLowerLimit,
		double iUpperLimit,
		vector<real>* wtsExclude) : AutoSearchCrossValidationDriver(
				_modelData,
				iGridSize,
				iLowerLimit,
				iUpperLimit,
				wtsExclude)
			 {}

HierarchyAutoSearchCrossValidationDriver::~HierarchyAutoSearchCrossValidationDriver() {
	// Do nothing
}

void HierarchyAutoSearchCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		CrossValidationSelector& selector,
		const CCDArguments& arguments) {

	ccd.setWeights(NULL);
	ccd.setHyperprior(maxPoint);
	ccd.setClassHyperprior(maxPointClass);
	ccd.resetBeta(); // Cold-start
}


void HierarchyAutoSearchCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// TODO Check that selector is type of CrossValidationSelector
	std::vector<real> weights;


	double tryvalue = 1;//modelData.getNormalBasedDefaultVar();
	double tryvalueClass = tryvalue; // start with same variance at the class and element level; // for hierarchy class variance
	double tryvalueStored = tryvalue;
	double tryvalueClassStored = tryvalue; // start with same variance at the class and element level; // for hierarchy class variance

	//UniModalSearch searcher(100, 0.01, log(1.5));
	//UniModalSearch searcherClass(100, 0.01, log(1.5)); // Need a better way to do this.

	const double eps = 1.00; //search stopper
	std::cout << "Default var = " << tryvalue << std::endl;


	bool finished = false;
	bool finished2 = false;
	bool outerFinished = false;
	bool drugLevelFinished = false;
	bool classLevelFinished = false;

	int step = 0;
	int step2 = 0;
	ccd.setHyperprior(0.1);
	ccd.setClassHyperprior(1);


	while (!outerFinished){
		step = 0;
		step2 = 0;
		UniModalSearch searcher(10, 0.01, log(1.5));
		while (!finished) {

			cout << "\n \n \n \t \t LOOP 1" << endl;
			ccd.setHyperprior(tryvalue);

			std::vector<double> predLogLikelihood;

			// Newly re-located code
			double pointEstimate = doCrossValidation(ccd, selector, arguments, step, predLogLikelihood);

			double stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);

			std::cout << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;
			searcher.tried(tryvalue, pointEstimate, stdDevEstimate);
			pair<bool,double> next = searcher.step();
			std::cout << "Completed at " << tryvalue << std::endl;
			std::cout << "Next point at " << next.second << " and " << next.first << std::endl;

			tryvalue = next.second;
			if (!next.first) {
				finished = true;
			}
			std::cout << searcher;
			step++;
			if (step >= maxSteps) {
				std::cerr << "Max steps reached!" << std::endl;
				finished = true;
			}
		}

		ccd.setHyperprior(tryvalue);
		//exit(-1);
		UniModalSearch searcherClass(10, 0.01, log(1.5));
		while (!finished2) {
			cout << "\n \n \n \t \t LOOP 2" << endl;

			ccd.setClassHyperprior(tryvalueClass);

			std::vector<double> predLogLikelihood;

			// Newly re-located code
			double pointEstimate = doCrossValidation(ccd, selector, arguments, step2, predLogLikelihood);

			double stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);

			std::cout << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;
			searcherClass.tried(tryvalueClass, pointEstimate, stdDevEstimate);
			pair<bool,double> next = searcherClass.step();
			std::cout << "Completed at " << tryvalueClass << std::endl;
			std::cout << "Next point at " << next.second << " and " << next.first << std::endl;

			tryvalueClass = next.second;
			if (!next.first) {
				cout << "tryvalueClass = " << tryvalueClass << endl;
				finished2 = true;
			}
			std::cout << searcherClass;
			step2++;
			if (step2 >= maxSteps) {
				std::cerr << "Max steps reached!" << std::endl;
				finished2 = true;
			}
		}
		ccd.setClassHyperprior(tryvalueClass);
		if (abs(tryvalueStored - tryvalue)/tryvalueStored < eps && abs(tryvalueClassStored - tryvalueClass)/tryvalueClassStored < eps){
			outerFinished = true;
		} else {
			finished = false;
			finished2 = false;
		}
		tryvalueStored = tryvalue;
		tryvalueClassStored = tryvalueClass;
		cout << "\n \n \n \t \t tryvalueStored = " << tryvalue << endl;
		cout << "\n \n \n \t \t tryvalueClassStored = " << tryvalueClassStored << endl;
	}

	maxPoint = tryvalue;
	maxPointClass = tryvalueClass;

	// Report results
	std::cout << std::endl;
	std::cout << "Maximum predicted log likelihood estimated at:" << std::endl;
	std::cout << "\t" << maxPoint << " (variance)" << std::endl;
	std::cout << "class level = " << maxPointClass << endl;


	if (!arguments.useNormalPrior) {
		double lambda = convertVarianceToHyperparameter(maxPoint);
		std::cout << "\t" << lambda << " (lambda)" << std::endl;
	}
	std:cout << std::endl;
}


} // namespace
