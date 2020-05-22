/*
 * ProportionSelector.cpp
 *
 *  Created on: Jul 18, 2012
 *      Author: msuchard
 */

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "ProportionSelector.h"

namespace bsccs {

ProportionSelector::ProportionSelector(
		int inTotal,
		std::vector<int> inIds,
		SelectorType inType,
		long inSeed,
	    loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error) : AbstractSelector(inIds, inType, inSeed, _logger, _error), total(inTotal) {

    std::ostringstream stream;
	stream << "Performing partial estimation with " << total
		<< " data lines.";
	logger->writeLine(stream);

//	permute();
}

ProportionSelector::~ProportionSelector() {
	// Nothing to do
}


AbstractSelector* ProportionSelector::clone() const {
	return new ProportionSelector(*this); // default copy constructor
}

void ProportionSelector::permute() {
//	selectedSet.clear();
//
//	if (type == SUBJECT) {
//		for (int i = 0; i < total; i++) {
//			selectedSet.insert(i);
//		}
//	} else {
        std::ostringstream stream;
        stream <<  "ProportionSelector::permute is not yet implemented.";
        error->throwError(stream);
//	}
}

void ProportionSelector::getWeights(int batch, std::vector<double>& weights) {
	if (weights.size() != K) {
		weights.resize(K);
	}

	std::fill(weights.begin(), weights.end(), 0.0);
	std::fill(weights.begin(), weights.begin() + total, 1.0);
}

void ProportionSelector::getComplement(std::vector<double>& weights) {
    std::ostringstream stream;
    stream <<  "ProportionSelector::getComplement is not yet implemented.";
    error->throwError(stream);
}

} // namespace

