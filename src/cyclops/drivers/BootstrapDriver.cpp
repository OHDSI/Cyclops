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

namespace bsccs {

using std::ostream_iterator;

BootstrapDriver::BootstrapDriver(
		int inReplicates,
		ModelData* inModelData,
		loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error		
		) : AbstractDriver(_logger, _error), replicates(inReplicates), modelData(inModelData),
		J(inModelData->getNumberOfColumns()) {

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

void BootstrapDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// TODO Make sure that selector is type-of BootstrapSelector
	std::vector<real> weights;

	for (int step = 0; step < replicates; step++) {
		selector.permute();
		selector.getWeights(0, weights);
		ccd.setWeights(&weights[0]);

        std::ostringstream stream;
		stream << std::endl << "Running replicate #" << (step + 1);
		logger->writeLine(stream);
		// Run CCD using a warm start
		ccd.update(arguments.modeFinding);

		// Store point estimates
		for (int j = 0; j < J; ++j) {
			estimates[j]->push_back(ccd.getBeta(j));
		}		
	}
}

void BootstrapDriver::logResults(const CCDArguments& arguments) {
    std::ostringstream stream;
    stream << "Not yet implemented.";
    error->throwError(stream);
}

void BootstrapDriver::logResults(const CCDArguments& arguments, std::vector<double>& savedBeta, std::string conditionId) {

	ofstream outLog(arguments.outFileName.c_str());
	if (!outLog) {
        std::ostringstream stream;        
		stream << "Unable to open log file: " << arguments.bsFileName;
		error->throwError(stream);
	}

	string sep(","); // TODO Make option

	if (!arguments.reportRawEstimates) {
		outLog << "Drug_concept_id" << sep << "Condition_concept_id" << sep <<
				"score" << sep << "standard_error" << sep << "bs_mean" << sep << "bs_lower" << sep <<
				"bs_upper" << sep << "bs_prob0" << endl;
	}

	for (int j = 0; j < J; ++j) {
		outLog << modelData->getColumn(j).getLabel() <<
			sep << conditionId << sep;
		if (arguments.reportRawEstimates) {
			ostream_iterator<real> output(outLog, sep.c_str());
			copy(estimates[j]->begin(), estimates[j]->end(), output);
			outLog << endl;
		} else {
			real mean = 0.0;
			real var = 0.0;
			real prob0 = 0.0;
			for (rvector::iterator it = estimates[j]->begin(); it != estimates[j]->end(); ++it) {
				mean += *it;
				var += *it * *it;
				if (*it == 0.0) {
					prob0 += 1.0;
				}
			}

			real size = static_cast<real>(estimates[j]->size());
			mean /= size;
			var = (var / size) - (mean * mean);
			prob0 /= size;

			sort(estimates[j]->begin(), estimates[j]->end());
			int offsetLower = static_cast<int>(size * 0.025);
			int offsetUpper = static_cast<int>(size * 0.975);

			real lower = *(estimates[j]->begin() + offsetLower);
			real upper = *(estimates[j]->begin() + offsetUpper);

			outLog << savedBeta[j] << sep;
			outLog << std::sqrt(var) << sep << mean << sep << lower << sep << upper << sep << prob0 << endl;
		}
	}
	outLog.close();
}

} // namespace
