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
//#include "ccd.h"
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


	double tryvalue = modelData.getNormalBasedDefaultVar();
	double tryvalueClass = tryvalue; // start with same variance at the class and element level; // for hierarchy class variance
	UniModalSearch searcher(10, 0.01, log(1.5));
	UniModalSearch searcherClass(10, 0.01, log(1.5)); // Need a better way to do this.

	const double eps = 0.05; //search stopper
	std::cout << "Default var = " << tryvalue << std::endl;


	bool finished = false;
	bool drugLevelFinished = false;
	bool classLevelFinished = false;

	int step = 0;
	while (!finished) {

		// More hierarchy logic
		ccd.setHyperprior(tryvalue);
		ccd.setClassHyperprior(tryvalueClass);

		std::vector<double> predLogLikelihood;

		// Newly re-located code
		double pointEstimate = doCrossValidation(ccd, selector, arguments, step, predLogLikelihood);

		double stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);

		std::cout << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;


        // alternate adapting the class and element level, unless one is finished
        if ((step % 2 == 0 && !drugLevelFinished) || classLevelFinished){
        	searcher.tried(tryvalue, pointEstimate, stdDevEstimate);
        	pair<bool,double> next = searcher.step();
        	tryvalue = next.second;
            std::cout << "Next point at " << next.second << " and " << next.first << std::endl;
            if (!next.first) {
               	drugLevelFinished = true;
            }
       	} else {
       		searcherClass.tried(tryvalueClass, pointEstimate, stdDevEstimate);
       		pair<bool,double> next = searcherClass.step();
       		tryvalueClass = next.second;
       	    std::cout << "Next Class point at " << next.second << " and " << next.first << std::endl;
            if (!next.first) {
               	classLevelFinished = true;
            }
        }
        // if everything is finished, end.
        if (drugLevelFinished && classLevelFinished){
        	finished = true;
        }

        std::cout << searcher;
        step++;
        if (step >= maxSteps) {
        	std::cerr << "Max steps reached!" << std::endl;
        	finished = true;
        }
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
