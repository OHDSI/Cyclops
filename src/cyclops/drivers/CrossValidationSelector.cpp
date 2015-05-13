/*
 * CrossValidation.cpp
 *
 *  Created on: Sep 9, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <algorithm>
#include <iterator>
#include <set>
#include <numeric>

#include "CrossValidationSelector.h"

namespace bsccs {

using std::vector;
using std::set;
using std::insert_iterator;

CrossValidationSelector::CrossValidationSelector(
		int inFold,
		std::vector<int> inIds,
		SelectorType inType,
		long inSeed,
	    loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error,		
		std::vector<double> const* baseweights) : AbstractSelector(inIds, inType, inSeed, _logger, _error), fold(inFold) {
	base_weights = baseweights;

	// Calculate interval starts
	intervalStart.reserve(fold + 1);
	int total_weight = std::accumulate(base_weights->begin(), base_weights->end(), 0);
	int fraction = total_weight / fold;
	int extra = total_weight - fraction * fold;

	int index = 0;
	for (int i = 0; i < fold; ++i) {
		intervalStart.push_back(index);
		for (int count = 0; count < fraction; ) {
			count += (*base_weights)[index];
			index += 1;
		}
		if (i < extra) {
			do {
				index++;
			} while ((*base_weights)[index] == 0);
		}
	}
	intervalStart.push_back(N);

	std::ostringstream stream;
	stream << "Performing " << fold << "-fold cross-validation [seed = "
		      << seed << "] with data partitions of sizes";
		      
	for (int i = 0; i < fold; ++i) {
		stream << " " << (intervalStart[i+1] - intervalStart[i]);
	}		      
	logger->writeLine(stream);

	permutation.resize(N);
}

void CrossValidationSelector::reseed() { 
//	std::cerr << "RESEEDING"  << std::endl;
	prng.seed(seed);
	for (size_t i = 0; i < N; ++i) {
		permutation[i] = i;
	}
} 

CrossValidationSelector::~CrossValidationSelector() {
	// Do nothing
}

void CrossValidationSelector::getWeights(int batch, std::vector<real>& weights) {
	if (weights.size() != K) {
		weights.resize(K);
	}

	std::copy((*base_weights).begin(), (*base_weights).end(), weights.begin());

	if (batch == -1) {
		return;
	}


	if (type == SelectorType::BY_PID) {
		std::set<int> excludeSet;
		std::copy(
				permutation.begin() + intervalStart[batch],
				permutation.begin() + intervalStart[batch + 1],
				insert_iterator< std::set<int> >(excludeSet, excludeSet.begin())
				);


		for (size_t k = 0; k < K; k++) {
			if (excludeSet.find(ids.at(k)) != excludeSet.end()) { // found
				weights[k] = 0.0;
			}
		}		
	} else { // SelectorType::BY_ROW
// 		std::fill(weights.begin(), weights.end(), 0.0);
// 		std::fill(weights.begin(), weights.begin() + 100, 1.0);
		std::for_each(
			permutation.begin() + intervalStart[batch],
			permutation.begin() + intervalStart[batch + 1],
			[&weights](const int excludeIndex) {
				weights[excludeIndex] = 0.0;
		});
	}
}

AbstractSelector* CrossValidationSelector::clone() const {
	return new CrossValidationSelector(*this); // default copy constructor
}

void CrossValidationSelector::getComplement(std::vector<real>& weights) {
	for(std::vector<real>::iterator it = weights.begin(); it != weights.end(); it++) {
		*it = 1 - *it;
	}
}

void CrossValidationSelector::permute() {

	// Do random shuffle
	if (!deterministic) {
//      	std::cerr << "PERMUTE" << std::endl;
		std::shuffle(permutation.begin(), permutation.end(), prng);
	}
}

} // namespace
