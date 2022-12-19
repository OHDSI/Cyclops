/*
 * WeightBasedSelector.cpp
 *
 *  Created on: Aug 26, 2022
 *      Author: msuchard
 */

#include <iostream>
#include <algorithm>
#include <iterator>
#include <set>

#include "WeightBasedSelector.h"

namespace bsccs {

using std::vector;

WeightBasedSelector::WeightBasedSelector(
		int inFold,
		std::vector<int> inIds,
		SelectorType inType,
		long inSeed,
	    loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error,
		std::vector<double>* wtsExclude,
		std::vector<double>* wtsOriginal) : AbstractSelector(inIds, inType, inSeed, _logger, _error) {

    std::ostringstream stream;
    stream << "Performing in- / out-of-sample search based on provided weights";
	logger->writeLine(stream);

	weightsExclude = wtsExclude;
	weightsOriginal = wtsOriginal;
}

void WeightBasedSelector::reseed() {
	// Do nothing
}

WeightBasedSelector::~WeightBasedSelector() {
	// Do nothing
}

void WeightBasedSelector::getWeights(int batch, std::vector<double>& weights) {
   if (weights.size() < weightsOriginal->size()) {
       weights.resize(weightsOriginal->size());
   }
	std::copy(weightsOriginal->begin(), weightsOriginal->end(), weights.begin());
}

AbstractSelector* WeightBasedSelector::clone() const {
	return new (std::nothrow) WeightBasedSelector(*this); // default copy constructor
}

void WeightBasedSelector::getComplement(std::vector<double>& weights) {
    for (auto it = weights.begin(); it != weights.end(); it++) {
	    *it = 1 - *it;
    }
}

void WeightBasedSelector::permute() {
	// Do nothing
}

} // namespace
