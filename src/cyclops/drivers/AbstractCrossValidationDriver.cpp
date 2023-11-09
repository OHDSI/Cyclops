/*
 * AbstractDriver.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <numeric>
#include <cmath>
#include <iterator>

// #include "boost/iterator/counting_iterator.hpp"

#include "Types.h"
#include "Thread.h"
#include "AbstractCrossValidationDriver.h"

namespace bsccs {

AbstractCrossValidationDriver::AbstractCrossValidationDriver(
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
			std::vector<double>* wtsExclude
	) : AbstractDriver(_logger, _error), weightsExclude(wtsExclude) {
	// Do nothing
}

AbstractCrossValidationDriver::~AbstractCrossValidationDriver() {
	// Do nothing
}

void AbstractCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// turn off syncCV for calculation of single GPU model
	if (ccd.GPU() && arguments.crossValidation.syncCV) {
		ccd.turnOffSyncCV();
	}

	ccd.setWeights(NULL);

    for (int i = 0; i < maxPoint.point.size(); ++i) {
	    ccd.setHyperprior(i, maxPoint.point[i]);
    }
	ccd.resetBeta(); // Cold-start
}

void AbstractCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& allArguments) {

	// TODO Check that selector is type of CrossValidationSelector
//     const auto& arguments = allArguments.crossValidation;

    // Start of new multi-thread set-up
	int nThreads = (allArguments.threads == -1) ?
        bsccs::thread::hardware_concurrency() :
	    allArguments.threads;

    if (nThreads < 1) {
        nThreads = 1;
    }

	std::ostringstream stream2;
	stream2 << "Using " << nThreads << " thread(s)";
	logger->writeLine(stream2);

	if (ccd.GPU() && allArguments.crossValidation.syncCV) {
//	if (false) {
		ccd.turnOnSyncCV(allArguments.crossValidation.foldToCompute);
		nThreads = 1;
	}
/*
	bool streamCV = true;
	if (streamCV) {
		ccd.turnOnStreamCV(allArguments.crossValidation.foldToCompute);
	}
*/
	std::vector<CyclicCoordinateDescent*> ccdPool;
	std::vector<AbstractSelector*> selectorPool;

	ccdPool.push_back(&ccd);
	selectorPool.push_back(&selector);

	for (int i = 1; i < nThreads; ++i) {
		ccdPool.push_back(ccd.clone(allArguments.computeDevice));
		selectorPool.push_back(selector.clone());
	}

	// Check of poor allocation
	bool allocationError = false;
	for (auto element : ccdPool) {
	    if (element == nullptr) {
	        allocationError = true;
	    }
	}

    for (auto element : selectorPool) {
        if (element == nullptr) {
            allocationError = true;
        }
    }

    if (allocationError) {
        std::ostringstream errorStream;
        errorStream << "Memory allocation error in multi-threaded cross validation driver";
        error->throwError(errorStream);
    }
	// End of multi-thread set-up

	// Delegate to auto or grid loop
    maxPoint = doCrossValidationLoop(ccd, selector, allArguments, nThreads, ccdPool, selectorPool);

	// Clean up
	for (int i = 1; i < nThreads; ++i) {
		delete ccdPool[i];
		delete selectorPool[i];
	}

	// Report results
	std::ostringstream stream1;
	std::ostringstream report;
	stream1 << std::endl;
	stream1 << "Maximum predicted log likelihood (" << maxPoint.value << ") estimated at:" << std::endl;
	report << maxPoint.value;
	stream1 << "\t";
	for (int i = 0; i < maxPoint.point.size(); ++i) {
	    stream1 << maxPoint.point[i] << " ";
	    report << " " << maxPoint.point[i];
	}
	stream1 << "(variance)" << std::endl;
	if (!allArguments.useNormalPrior) {
	    stream1 << "\t";
	    for (int i = 0; i < maxPoint.point.size(); ++i) {
		    double lambda = convertVarianceToHyperparameter(maxPoint.point[i]);
	        stream1 << lambda << " ";
	    }
		stream1 << "(lambda)" << std::endl;
	}
	logger->writeLine(stream1);

	ccd.setCrossValidationInfo(report.str());
}

double AbstractCrossValidationDriver::doCrossValidationStep(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& allArguments,
		int step,
		int nThreads,
		std::vector<CyclicCoordinateDescent*>& ccdPool,
		std::vector<AbstractSelector*>& selectorPool,
		std::vector<double>& predLogLikelihood){

    const auto& arguments = allArguments.crossValidation;
    bool coldStart = allArguments.resetCoefficients;

	predLogLikelihood.resize(arguments.foldToCompute);

	// GPU using syncCV
	//if (false) {
	if (ccd.GPU() && allArguments.crossValidation.syncCV) {
		int repetitions = arguments.foldToCompute / arguments.fold;

		// set weights
		std::vector<std::vector<double>> weightsPool;
		weightsPool.resize(arguments.foldToCompute);
		for (int i = 0; i < repetitions; ++i) {
			selector.permute();
			for (int j = 0; j < arguments.fold; ++j) {
				int index = i * arguments.fold + j;
				selector.getWeights(j, weightsPool[i * arguments.fold + j]);
				if (weightsExclude){
					for(size_t k = 0; k < weightsExclude->size(); k++){
						if (weightsExclude->at(k) == 1.0){
							weightsPool[i * arguments.fold + j][k] = 0.0;
						}
					}
				}
				ccd.setWeights(&weightsPool[i * arguments.fold + j][0], index);
			}
		}

		std::ostringstream stream;
		stream << "Running at " << ccd.getPriorInfo() << " ";
		stream << "Grid-point #" << (step + 1) << " at ";
		std::vector<double> hyperprior = ccd.getHyperprior();
		std::copy(hyperprior.begin(), hyperprior.end(),
				std::ostream_iterator<double>(stream, " "));

		bool write = true;

		if (write) logger->writeLine(stream);

		if (coldStart) {
			ccd.resetBeta();
		}

		ccd.update(allArguments.modeFinding);

		// calculate predictive log likelihoods
		for (int cvIndex = 0; cvIndex < arguments.foldToCompute; cvIndex++) {
			selector.getComplement(weightsPool[cvIndex]);  // TODO THREAD_SAFE
			if (weightsExclude){
				for(int j = 0; j < (int)weightsExclude->size(); j++){
					if(weightsExclude->at(j) == 1.0){
						weightsPool[cvIndex][j] = 0.0;
					}
				}
			}
		}

		std::vector<double> logLikelihood = ccd.getPredictiveLogLikelihood(weightsPool);
		for (int i=0; i<arguments.foldToCompute; i++) {
			predLogLikelihood[i] = logLikelihood[i];
		}

	} else {

		auto& weightsExclude = this->weightsExclude;
		auto& logger = this->logger;

		// auto scheduler = TaskScheduler<decltype(boost::make_counting_iterator(0))>(
		// 		boost::make_counting_iterator(0),
		// 		boost::make_counting_iterator(arguments.foldToCompute),
		// 		nThreads);

		auto scheduler = TaskScheduler<IncrementableIterator<size_t>>(
		    IncrementableIterator<size_t>(0),
		    IncrementableIterator<size_t>(arguments.foldToCompute),
		    nThreads);

		auto oneTask =
				[step, coldStart, nThreads, &ccdPool, &selectorPool,
				 &arguments, &allArguments, &predLogLikelihood,
				 &weightsExclude, &logger //, &lock
				 //    ,&ccd, &selector
				 , &scheduler
				 ](int task) {

			const auto uniqueId = scheduler.getThreadIndex(task);
			auto ccdTask = ccdPool[uniqueId];
			auto selectorTask = selectorPool[uniqueId];

			// Bring selector up-to-date
			if (task == 0 || nThreads > 1) {
				selectorTask->reseed();
			}
			int i = (nThreads == 1) ? task : 0;
			for ( ; i <= task; ++i) {
				int fold = i % arguments.fold;
				if (fold == 0) {
					selectorTask->permute();
				}
			}

			int fold = task % arguments.fold;
			ccdTask->getFold(fold);

			// Get this fold and update
			std::vector<double> weights; // Task-specific
			selectorTask->getWeights(fold, weights);
			if (weightsExclude){
				for(size_t j = 0; j < weightsExclude->size(); j++){
					if (weightsExclude->at(j) == 1.0){
						weights[j] = 0.0;
					}
				}
			}
			ccdTask->setWeights(&weights[0]);

			std::ostringstream stream;
			stream << "Running at " << ccdTask->getPriorInfo() << " ";
			stream << "Grid-point #" << (step + 1) << " at ";
			std::vector<double> hyperprior = ccdTask->getHyperprior();
			std::copy(hyperprior.begin(), hyperprior.end(),
					std::ostream_iterator<double>(stream, " "));
			stream << "\tFold #" << (fold + 1)
								  << " Rep #" << (task / arguments.fold + 1) << " pred log like = ";

			if (coldStart) {
				ccdTask->resetBeta();
			}

			ccdTask->update(allArguments.modeFinding);

			if (ccdTask->getUpdateReturnFlag() == SUCCESS) {

				// Compute predictive loglikelihood for this fold
				selectorTask->getComplement(weights);  // TODO THREAD_SAFE
				if (weightsExclude){
					for(int j = 0; j < (int)weightsExclude->size(); j++){
						if(weightsExclude->at(j) == 1.0){
							weights[j] = 0.0;
						}
					}
				}

				double logLikelihood = ccdTask->getNewPredictiveLogLikelihood(&weights[0]);

				// Store value
				stream << logLikelihood;
				predLogLikelihood[task] = logLikelihood;
			} else {
				ccdTask->resetBeta(); // cold start for stability
				stream << "Not computed";
				predLogLikelihood[task] = std::numeric_limits<double>::quiet_NaN();
			}

			bool write = true;

			if (write) logger->writeLine(stream);
		};

/*
		// Run all task in serial
		for (int i = 0; i < arguments.foldToCompute; i++) {
			oneTask(i);
		}
*/
		// Run all tasks in parallel
		if (nThreads > 1) {
			ccd.getProgressLogger().setConcurrent(true);
		}
		scheduler.execute(oneTask);
		if (nThreads > 1) {
			ccd.getProgressLogger().setConcurrent(false);
			ccd.getProgressLogger().flush();
		}

	}

	double pointEstimate = computePointEstimate(predLogLikelihood);

	return(pointEstimate);
}

double AbstractCrossValidationDriver::computePointEstimate(const std::vector<double>& value) {
	// Mean of log values, ignoring nans
	double total = 0.0;
	int count = 0;
	for (auto x : value) {
		if (x == x) {
			total += x;
			count += 1;
		}
	}
	return total / static_cast<double>(count);
}

double AbstractCrossValidationDriver::computeStDev(const std::vector<double>& value, double mean) {
	// Ignoring nans
	double inner_product = 0.0;
	int count = 0;
	for (auto x : value) {
		if (x == x) {
			inner_product += x * x;
			count += 1;
		}
	}
	return std::sqrt(inner_product / static_cast<double>(count) - mean * mean);
// 	return std::sqrt(std::inner_product(value.begin(), value.end(), value.begin(), 0.0)
// 	/ static_cast<double>(value.size()) - mean * mean);
}


} // namespace
