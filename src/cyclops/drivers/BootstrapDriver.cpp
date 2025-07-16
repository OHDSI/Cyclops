/*
 * BootstrapDriver.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <sstream>

#include "BootstrapDriver.h"
#include "AbstractSelector.h"
#include "Thread.h"

namespace bsccs {

using std::ostream_iterator;

BootstrapDriver::BootstrapDriver(
		int inReplicates,
		AbstractModelData* inModelData,
		loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error
		) : AbstractDriver(_logger, _error), replicates(inReplicates), modelData(inModelData),
		J(inModelData->getNumberOfCovariates()) {

	// Set-up storage for bootstrap estimates
	estimates.resize(J);
//	int count = 0;
	for (rarrayIterator it = estimates.begin(); it != estimates.end(); ++it) {
		*it = new rvector();
	}
}

BootstrapDriver::~BootstrapDriver() {
	for (rarrayIterator it = estimates.begin(); it != estimates.end(); ++it) {
		if (*it) {
			delete *it;
		}
	}
}

std::vector<double> BootstrapDriver::flattenEstimates() {

    std::vector<double> flat;
    for (int j = 0; j < J; ++j) {
        for (int i = 0; i < replicates; ++i) {
            flat.push_back(estimates[j]->at(i));
        }
    }

    return flat;
}

void BootstrapDriver::drive(
        CyclicCoordinateDescent& ccd,
        AbstractSelector& selector,
        const CCDArguments& allArguments) {

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
    doBootstrap(ccd, selector, allArguments, nThreads, ccdPool, selectorPool);

    // Clean up
    for (int i = 1; i < nThreads; ++i) {
        delete ccdPool[i];
        delete selectorPool[i];
    }

    ccd.setBootStrapInfo(flattenEstimates());
}

void BootstrapDriver::doBootstrap(
        CyclicCoordinateDescent& ccd,
        AbstractSelector& selector,
        const CCDArguments& allArguments,
        int nThreads,
        std::vector<CyclicCoordinateDescent*>& ccdPool,
        std::vector<AbstractSelector*>& selectorPool) {

    // selector.permute() in series to keep same PRNG stream
    std::vector<std::vector<double>> weightsPool(replicates);
    for (int i = 0; i < replicates; ++i) {
        selector.permute();
        selector.getWeights(0, weightsPool[i]);}

    // one task
    auto scheduler = TaskScheduler<IncrementableIterator<size_t>>(
        IncrementableIterator<size_t>(0),
        IncrementableIterator<size_t>(replicates),
        nThreads);

    auto oneTask = [&](int task) {

        const auto uniqueId = scheduler.getThreadIndex(task);
        auto ccdTask = ccdPool[uniqueId];
        auto selectorTask = selectorPool[uniqueId];

        // for loop to parallelize
        ccdTask->setWeights(&weightsPool[task][0]);
        std::ostringstream stream;
        stream << "\nRunning replicate #" << (task + 1);
        logger->writeLine(stream);

        ccdTask->update(allArguments.modeFinding);

        // store results without race-conditions
        std::lock_guard<std::mutex> lock(estimatesMutex);
        for (int j = 0; j < J; ++j) {
            estimates[j]->push_back(ccdTask->getBeta(j));
        }
    };


    // execute all tasks in parallel
    if (nThreads > 1) {
        ccd.getProgressLogger().setConcurrent(true);
        }
    scheduler.execute(oneTask);
    if (nThreads > 1) {
        ccd.getProgressLogger().setConcurrent(false);
        ccd.getProgressLogger().flush();
        }
    }



void BootstrapDriver::logResults(const CCDArguments& arguments) {
    std::ostringstream stream;
    stream << "Not yet implemented.";
    error->throwError(stream);
}

// std::string BootstrapDriver::logResultsToString(const CCDArguments& arguments) {
//
//     string sep(","); // TODO Make option
//
//     ostream outLog;
//     for (int j = 0; j < J; ++j) {
//         outLog << modelData->getColumnLabel(j) <<
//             sep;
//
//         ostream_iterator<double> output(outLog, sep.c_str());
//         copy(estimates[j]->begin(), estimates[j]->end(), output);
//         outLog << endl;
//     }
//
//     return outLog.c_str();
// }

void BootstrapDriver::logResults(const CCDArguments& arguments, std::vector<double>& savedBeta, std::string conditionId) {

    ofstream outLog(arguments.outFileName.c_str());
    if (!outLog) {
        std::ostringstream stream;
        stream << "Unable to open log file: " << arguments.bsFileName;
        error->throwError(stream);
    }

    logResultsImpl(outLog, arguments, savedBeta, conditionId);

    outLog.close();
}

void BootstrapDriver::logResultsImpl(
        ostream& outLog,
        const CCDArguments& arguments, std::vector<double>& savedBeta, std::string conditionId) {

// void BootstrapDriver::logResults(const CCDArguments& arguments, std::vector<double>& savedBeta, std::string conditionId) {
//
// 	ofstream outLog(arguments.outFileName.c_str());
// 	if (!outLog) {
//         std::ostringstream stream;
// 		stream << "Unable to open log file: " << arguments.bsFileName;
// 		error->throwError(stream);
// 	}

	string sep(","); // TODO Make option

	if (!arguments.reportRawEstimates) {
		outLog << "Drug_concept_id" << sep << "Condition_concept_id" << sep <<
				"score" << sep << "standard_error" << sep << "bs_mean" << sep << "bs_lower" << sep <<
				"bs_upper" << sep << "bs_prob0" << endl;
	}

	for (int j = 0; j < J; ++j) {
		outLog << modelData->getColumnLabel(j) <<
			sep << conditionId << sep;
		if (arguments.reportRawEstimates) {
			ostream_iterator<double> output(outLog, sep.c_str());
			copy(estimates[j]->begin(), estimates[j]->end(), output);
			outLog << endl;
		} else {
			double mean = 0.0;
			double var = 0.0;
			double prob0 = 0.0;
			for (rvector::iterator it = estimates[j]->begin(); it != estimates[j]->end(); ++it) {
				mean += *it;
				var += *it * *it;
				if (*it == 0.0) {
					prob0 += 1.0;
				}
			}

			double size = static_cast<double>(estimates[j]->size());
			mean /= size;
			var = (var / size) - (mean * mean);
			prob0 /= size;

			sort(estimates[j]->begin(), estimates[j]->end());
			int offsetLower = static_cast<int>(size * 0.025);
			int offsetUpper = static_cast<int>(size * 0.975);

			double lower = *(estimates[j]->begin() + offsetLower);
			double upper = *(estimates[j]->begin() + offsetUpper);

			outLog << savedBeta[j] << sep;
			outLog << std::sqrt(var) << sep << mean << sep << lower << sep << upper << sep << prob0 << endl;
		}
	}
	// outLog.close();
}

void BootstrapDriver::logHR(const CCDArguments& arguments, std::vector<double>& savedBeta, std::string treatmentId) {

	int tId = 0;
	while (modelData->getColumnLabel(tId) != treatmentId) tId++;

	if (arguments.outFileName.length() < 1) {
	    return;
	}

	ofstream outLog(arguments.outFileName.c_str());
	if (!outLog) {
        std::ostringstream stream;
		stream << "Unable to open log file: " << arguments.bsFileName;
		error->throwError(stream);
	}

	string sep(","); // TODO Make option

	// Stats
	outLog << "Drug_concept_id" << sep << "Condition_concept_id" << sep <<
		"score" << sep << "standard_error" << sep << "bs_mean" << sep << "bs_lower" << sep <<
		"bs_upper" << sep << "bs_prob0" << endl;

	for (int j = tId; j < J; ++j) {
		outLog << modelData->getColumnLabel(j) <<
			sep << treatmentId << sep;

		double mean = 0.0;
		double var = 0.0;
		double prob0 = 0.0;
		for (rvector::iterator it = estimates[j]->begin(); it != estimates[j]->end(); ++it) {
			mean += *it;
			var += *it * *it;
			if (*it == 0.0) {
				prob0 += 1.0;
			}
		}

		double size = static_cast<double>(estimates[j]->size());
		mean /= size;
		var = (var / size) - (mean * mean);
		prob0 /= size;

		sort(estimates[j]->begin(), estimates[j]->end());
		int offsetLower = static_cast<int>(size * 0.025);
		int offsetUpper = static_cast<int>(size * 0.975);

		double lower = *(estimates[j]->begin() + offsetLower);
		double upper = *(estimates[j]->begin() + offsetUpper);

		outLog << savedBeta[j] << sep;
		outLog << std::sqrt(var) << sep << mean << sep << lower << sep << upper << sep << prob0 << endl;
	}

        if (arguments.reportDifference) {
                double mean = 0.0;
                double var = 0.0;
                double prob0 = 0.0;

		double size = static_cast<double>(estimates[tId]->size());
                std::vector<double> diff(size, static_cast<double>(0));
                for (int i = 0; i < size; i++) {
                        diff[i] = *(estimates[tId]->begin() + i) - *(estimates[tId+1]->begin() + i);
                        mean += diff[i];
                        var += diff[i] * diff[i];
                        if (diff[i] == 0.0) {
                                prob0 += 1.0;
                        }
                }

                mean /= size;
                var = (var / size) - (mean * mean);
                prob0 /= size;
                sort(diff.begin(), diff.end());

                int offsetLower = static_cast<int>(size * 0.025);
                int offsetUpper = static_cast<int>(size * 0.975);

                double lower = diff[offsetLower];
                double upper = diff[offsetUpper];

                outLog << savedBeta[tId] - savedBeta[tId+1] << sep;
                outLog << std::sqrt(var) << sep << mean << sep << lower << sep << upper << sep << prob0 << endl;
	}
	outLog.close();
}

} // namespace
