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

	for (int i = 0; i < base_weights->size(); ++i) {
		if (base_weights->at(i) != 0) {
			weight_map.push_back(i);
		}
	}

	num_base_weights = weight_map.size();

	// Calculate interval starts
	intervalStart.reserve(fold + 1);
	int index = 0;
	int fraction = num_base_weights / fold;
	int extra = num_base_weights - fraction * fold;
	for (int i = 0; i < fold; i++) {
		intervalStart.push_back(index);
		index += fraction;
		if (i < extra) {
			index++;
		}
	}
	intervalStart.push_back(num_base_weights);

	std::ostringstream stream;
	stream << "Performing " << fold << "-fold cross-validation [seed = "
		      << seed << "] with data partitions of sizes";
		      
	for (int i = 0; i < fold; ++i) {
		stream << " " << (intervalStart[i+1] - intervalStart[i]);
	}		      
	logger->writeLine(stream);

	permutation.resize(num_base_weights);
}

void CrossValidationSelector::reseed() { 
//	std::cerr << "RESEEDING"  << std::endl;
	prng.seed(seed);
	for (size_t i = 0; i < num_base_weights; ++i) {
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


		for (size_t k = 0; k < num_base_weights; k++) {
			if (excludeSet.find(ids.at(k)) != excludeSet.end()) { // found
				weights[weight_map[k]] = 0.0;
			}
		}		
	} else { // SelectorType::BY_ROW
// 		std::fill(weights.begin(), weights.end(), 0.0);
// 		std::fill(weights.begin(), weights.begin() + 100, 1.0);
		auto local_weight_map = weight_map;
		std::for_each(
			permutation.begin() + intervalStart[batch],
			permutation.begin() + intervalStart[batch + 1],
			[&weights, &local_weight_map](const int excludeIndex) {
				weights[local_weight_map[excludeIndex]] = 0.0;
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
