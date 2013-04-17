/*
 * LeaveOneOutDriver.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: msuchard
 */

#include "LeaveOneOutDriver.h"
#include "io/OutputWriter.h"

namespace bsccs {

	LeaveOneOutDriver::LeaveOneOutDriver(long _length) : length(_length) {
		// Do nothing
	}

	LeaveOneOutDriver::~LeaveOneOutDriver() {
		// Do nothing
	}

	void logResults(const CCDArguments& arguments) {
		// Do nothing
	}

	void LeaveOneOutDriver::drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments) {

		std::vector<real> weights;

		int loops = std::min(arguments.replicates, (int) length);

		for (int i = 0; i < loops; ++i) {
			selector.permute(); // pick subject to leave out
			selector.getWeights(0, weights);
			ccd.setWeights(&weights[0]);
			ccd.update(arguments.maxIterations, arguments.convergenceType,
					arguments.tolerance);
			for (int j = 0; j < outputList.size(); ++j) {
				outputList[j]->writeFile();
			}
		}

//				double logLikelihood = ccd.getPredictiveLogLikelihood(&weights[0]);
//
//				std::cout << "Grid-point #" << (step + 1) << " at " << point;
//				std::cout << "\tFold #" << (fold + 1)
//				          << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
//				          << logLikelihood << std::endl;
//
//				// Store value
//				predLogLikelihood.push_back(logLikelihood);
//			}
//
//			double value = computePointEstimate(predLogLikelihood) /
//					(double(arguments.foldToCompute) / double(arguments.fold));
//			gridPoint.push_back(point);
//			gridValue.push_back(value);
//		}
//
//		// Report results
//		double maxPoint;
//		double maxValue;
//		findMax(&maxPoint, &maxValue);
//
//		std::cout << std::endl;
//		std::cout << "Maximum predicted log likelihood (" << maxValue << ") found at:" << std::endl;
//		std::cout << "\t" << maxPoint << " (variance)" << std::endl;
//		if (!arguments.useNormalPrior) {
//			double lambda = convertVarianceToHyperparameter(maxPoint);
//			std::cout << "\t" << lambda << " (lambda)" << std::endl;
//		}
//		std:cout << std::endl;
	}
	void LeaveOneOutDriver::logResults(const CCDArguments& arguments) {

	}

} /* namespace bsccs */
