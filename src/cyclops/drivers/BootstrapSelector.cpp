/*
 * BootstrapSelector.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "BootstrapSelector.h"

namespace bsccs {

BootstrapSelector::BootstrapSelector(
		int replicates,
		std::vector<int> inIds,
		SelectorType inType,
		long inSeed,
	    loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error,
		std::vector<real>* wtsExclude) : AbstractSelector(inIds, inType, inSeed, _logger, _error) {

    std::ostringstream stream;
	stream << "Performing bootstrap estimation with " << replicates
		<< " replicates [seed = " << seed << "]";
	logger->writeLine(stream);

	if(wtsExclude){
		for(size_t i = 0; i < wtsExclude->size(); i++){
			if(wtsExclude->at(i) == 0){
				indicesIncluded.push_back(i);
			}
		}
	}
	else{
		for(size_t i = 0; i < N; i++){
			indicesIncluded.push_back(i);
		}
	}

	permute();
}

BootstrapSelector::~BootstrapSelector() {
	// Nothing to do
}

AbstractSelector* BootstrapSelector::clone() const {
	return new BootstrapSelector(*this);
}

void BootstrapSelector::permute() {
	selectedSet.clear();

	// Get non-excluded indices
	int N_new = indicesIncluded.size();
//	if (type == SelectorType::BY_PID) {
	    std::uniform_int_distribution<int> uniform(0, N_new - 1);
		for (int i = 0; i < N_new; i++) {
            int ind =  uniform(prng);
			int draw = indicesIncluded[ind];
			selectedSet.insert(draw);
		}
/*
	} else {
        std::ostringstream stream;
        stream << "BootstrapSelector::permute is not yet implemented.";
        error->throwError(stream);
	}
*/
}

void BootstrapSelector::getWeights(int batch, std::vector<double>& weights) {
	if (weights.size() != K) {
		weights.resize(K);
	}

	std::fill(weights.begin(), weights.end(), 0.0);
	if (batch == -1) {
		return;
	}

	//if (type == SelectorType::BY_PID) {
		for (size_t k = 0; k < K; k++) {
			int count = selectedSet.count(ids.at(k));
			weights[k] = static_cast<real>(count);
		}
/*
	} else {
        std::ostringstream stream;
        stream << "BootstrapSelector::getWeights is not yet implemented.";
        error->throwError(stream);
	}
*/
}

void BootstrapSelector::getComplement(std::vector<double>& weights) {
    std::ostringstream stream;
    stream << "BootstrapSelector::getComplement is not yet implemented.";
    error->throwError(stream);
}

} // namespace

