/*
 * AbstractDriver.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <numeric>
#include <cmath>
#include <iterator>
#include <mutex>

// #include "boost/iterator/counting_iterator.hpp"

#include "Types.h"
#include "Thread.h"
#include "AbstractCrossValidationDriver.h"

namespace bsccs {

// The fold partition of one cross-validation repetition: built by its first task, shared read-only by
// its tasks, released when the last one finishes
struct RepetitionPartition {
	std::mutex lock;
	bool built = false;
	FoldPartitionPtr partition;
	int remaining = 0; // tasks of this repetition not yet finished
};

// Segment of each row (see AbstractModelSpecifics::makeFoldPartition()) from the selector's weights for
// every fold of its current permutation; nullptr unless every weight is 0 or 1 and each row is held out
// in one fold, every fold or none
static FoldPartitionPtr buildFoldPartition(CyclicCoordinateDescent& ccd, AbstractSelector& selector,
                                           int folds, std::vector<double>* weightsExclude) {
	std::vector<double> weights;
	std::vector<int> zeros, heldOutIn;
	for (int fold = 0; fold < folds; ++fold) {
		selector.getWeights(fold, weights);
		if (weightsExclude) {
			for (size_t k = 0; k < weightsExclude->size(); ++k) {
				if (weightsExclude->at(k) == 1.0) {
					weights[k] = 0.0;
				}
			}
		}
		if (fold == 0) {
			zeros.assign(weights.size(), 0);
			heldOutIn.assign(weights.size(), -1);
		}
		for (size_t k = 0; k < weights.size(); ++k) {
			if (weights[k] == 0.0) {
				++zeros[k];
				heldOutIn[k] = fold;
			} else if (weights[k] != 1.0) {
				return nullptr;
			}
		}
	}
	std::vector<int> segment(zeros.size());
	for (size_t k = 0; k < zeros.size(); ++k) {
		if (zeros[k] == 0) {
			segment[k] = folds;      // in every training set
		} else if (zeros[k] == 1) {
			segment[k] = heldOutIn[k];
		} else if (zeros[k] == folds) {
			segment[k] = folds + 1;  // in none
		} else {
			return nullptr;
		}
	}
	return ccd.makeFoldPartition(segment, folds);
}

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

	restoreWeights(ccd, arguments);

    for (int i = 0; i < maxPoint.point.size(); ++i) {
	    ccd.setHyperprior(i, maxPoint.point[i]);
    }
	ccd.resetBeta(); // Cold-start
}

void AbstractCrossValidationDriver::restoreWeights(
		CyclicCoordinateDescent& ccd,
		const CCDArguments& arguments) {

	// The weight-based search (fold = -1) uses its weights to split in- from out-of-sample rows,
	// so its final fit uses all rows
	if (savedWeights.empty() || arguments.crossValidation.fold == -1) {
		ccd.setWeights(NULL);
	} else {
		ccd.setWeights(savedWeights.data());
	}
}

void AbstractCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& allArguments) {

	savedWeights = ccd.getWeights(); // restored for the final fit, see restoreWeights()

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
	ccd.setHeldOutRowsUnused(false); // later fits keep every row up to date

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

		// Fold fits with 0/1 weights may leave held-out rows stale (models that support it): their xBeta is
		// recomputed before the predictive log likelihood. Logistic regression also shares, per repetition,
		// the entries of every column grouped by fold, so that fold fits read only their training rows.
		const bool usePartitions = arguments.fold > 0; // not the weight-based search

		// One partition per repetition, built when its first task needs it
		const int repetitions = usePartitions ? (arguments.foldToCompute + arguments.fold - 1) / arguments.fold : 0;
		std::vector<RepetitionPartition> repetitionPartitions(repetitions);
		for (int r = 0; r < repetitions; ++r) {
			repetitionPartitions[r].remaining = std::min(arguments.fold, arguments.foldToCompute - r * arguments.fold);
		}

		// auto scheduler = TaskScheduler<decltype(boost::make_counting_iterator(0))>(
		// 		boost::make_counting_iterator(0),
		// 		boost::make_counting_iterator(arguments.foldToCompute),
		// 		nThreads);

		auto scheduler = TaskScheduler<IncrementableIterator<size_t>>(
		    IncrementableIterator<size_t>(0),
		    IncrementableIterator<size_t>(arguments.foldToCompute),
		    nThreads);

		auto runTask =
				[step, coldStart, nThreads, &ccdPool, &selectorPool,
				 &arguments, &allArguments, &predLogLikelihood,
				 &weightsExclude, &logger, usePartitions, &repetitionPartitions //, &lock
				 //    ,&ccd, &selector
				 ](int task, size_t uniqueId) {

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
			ccdTask->setHeldOutRowsUnused(true); // xBeta is recomputed before the predictive log likelihood

			FoldPartitionPtr partition;
			if (usePartitions) {
				RepetitionPartition& entry = repetitionPartitions[task / arguments.fold];
				std::lock_guard<std::mutex> guard(entry.lock);
				if (!entry.built) {
					entry.partition = buildFoldPartition(*ccdTask, *selectorTask, arguments.fold, weightsExclude);
					entry.built = true;
				}
				partition = entry.partition;
			}
			if (partition) {
				ccdTask->setFoldPartition(partition, fold);
			}

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

			if (partition) {
				ccdTask->clearFoldPartition();
				partition.reset();
			}
			if (usePartitions) {
				RepetitionPartition& entry = repetitionPartitions[task / arguments.fold];
				std::lock_guard<std::mutex> guard(entry.lock);
				if (--entry.remaining == 0) {
					entry.partition.reset(); // the repetition's last fold fit is done
				}
			}

			bool write = true;

			if (write) logger->writeLine(stream);
		};

		auto oneTask = [&runTask, &scheduler](int task) {
			runTask(task, scheduler.getThreadIndex(task));
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
