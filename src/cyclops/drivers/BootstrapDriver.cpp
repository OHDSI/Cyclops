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
#include <limits>
#include <memory>
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
    const int allThreads = nThreads; // before the clamp: see the second-order threads below

    // More threads than replicates would only hold unused copies of the model
    if (nThreads > replicates) {
        nThreads = std::max(1, replicates);
    }

	// Replicates overwrite the weights and coefficients of the fit; restored by fitRestore below
	std::vector<double> savedWeights = ccd.getWeights();
	std::vector<double> savedBeta(J);
	for (int j = 0; j < J; ++j) {
		savedBeta[j] = ccd.getBeta(j);
	}

    std::ostringstream stream2;
    stream2 << "Using " << nThreads << " thread(s)";
    logger->writeLine(stream2);

    std::vector<CyclicCoordinateDescent*> ccdPool;
    std::vector<AbstractSelector*> selectorPool;

    ccdPool.push_back(&ccd);
    selectorPool.push_back(&selector);

    // The copies are owned here, so they go also when a replicate throws; the fit gets back its second-order threads,
    // weights and coefficients also then
    std::vector<std::unique_ptr<CyclicCoordinateDescent>> ccdCopies;
    std::vector<std::unique_ptr<AbstractSelector>> selectorCopies;
    struct FitRestore {
        CyclicCoordinateDescent& ccd;
        int threads;
        std::vector<double>& weights;
        const std::vector<double>& beta;
        ~FitRestore() {
            try {
                ccd.setThreads(threads);
                ccd.setWeights(weights.empty() ? NULL : weights.data());
                ccd.setBeta(beta); // assigns rather than updates, so non-finite replicate estimates do not carry over
            } catch (...) { } // a destructor must not throw
        }
    } fitRestore{ccd, ccd.getThreads(), savedWeights, savedBeta};

    for (int i = 1; i < nThreads; ++i) {
        ccdCopies.emplace_back(ccd.clone(allArguments.computeDevice));
        selectorCopies.emplace_back(selector.clone());
        ccdPool.push_back(ccdCopies.back().get());
        selectorPool.push_back(selectorCopies.back().get());
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
        errorStream << "Memory allocation error in multi-threaded bootstrap driver";
        error->throwError(errorStream);
        }

    // Threads beyond the replicates in parallel go to each replicate's second-order kernels (lr, plr, cox), whose
    // results do not depend on their threads. Coordinate descent on a copy keeps one thread; on the fit itself
    // (ccdPool[0]), Cox drops concurrent coordinate gradients while plr may use the spare pool threads, and the
    // estimates are identical either way.
    for (auto element : ccdPool) {
        element->setSecondOrderThreads(allThreads / nThreads);
    }
    // End of multi-thread set-up

    doBootstrap(ccd, allArguments, nThreads, ccdPool, selectorPool, savedBeta);

    ccd.setBootStrapInfo(flattenEstimates());
}

void BootstrapDriver::doBootstrap(
        CyclicCoordinateDescent& ccd,
        const CCDArguments& allArguments,
        int nThreads,
        std::vector<CyclicCoordinateDescent*>& ccdPool,
        std::vector<AbstractSelector*>& selectorPool,
        const std::vector<double>& startingBeta) {

    bool coldStart = allArguments.resetCoefficients;

    std::vector<std::vector<double>> weightsPool(nThreads);

    // Replicates drawn so far by each selector: a selector skips the draws of the replicates other threads ran, so
    // replicate r has the r-th draws of the serial stream whichever thread runs it
    std::vector<int> drawn(nThreads, 0);

    // allocate results so cache-lines do not thrash each other
    std::vector<double> storage(J * replicates);

    // Every replicate starts from the coefficients of the fit (or cold), so its estimates do not depend on the
    // replicates its thread ran before, nor on the number of threads
    auto oneTask = [&](size_t task, size_t uniqueId) {

        auto ccdTask = ccdPool[uniqueId];
        auto selector = selectorPool[uniqueId];

        selector->advance(static_cast<int>(task) - drawn[uniqueId]); // tasks start in increasing order
        selector->permute();
        drawn[uniqueId] = static_cast<int>(task) + 1;

        selector->getWeights(0, weightsPool[uniqueId]);
        ccdTask->setWeights(weightsPool[uniqueId].data());

        std::ostringstream stream;
        stream << "\nRunning replicate #" << (task + 1);
        logger->writeLine(stream);

        if (coldStart) {
            ccdTask->resetBeta();
        } else {
            ccdTask->setBeta(startingBeta);
        }

        ccdTask->update(allArguments.modeFinding);

        if (ccdTask->getUpdateReturnFlag() == SUCCESS) {

            for (int j = 0; j < J; ++j) {
                storage[task * J + j] = ccdTask->getBeta(j);
            }

        } else {

            for (int j = 0; j < J; ++j) {
                storage[task * J + j] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    };

    // execute all tasks in parallel; the first error stops the others and is rethrown here
    if (nThreads > 1) {
        ccd.getProgressLogger().setConcurrent(true);
    }

    try {
        executeDynamic(replicates, nThreads, oneTask);
    } catch (...) {
        if (nThreads > 1) {
            ccd.getProgressLogger().setConcurrent(false);
            ccd.getProgressLogger().flush();
        }
        throw;
    }

    if (nThreads > 1) {
        ccd.getProgressLogger().setConcurrent(false);
        ccd.getProgressLogger().flush();
    }

    // transpose results
    for (int i = 0; i < replicates; ++i) {
        for (int j = 0; j < J; ++j) {
            estimates[j]->push_back(storage[i * J + j]);
        }
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
