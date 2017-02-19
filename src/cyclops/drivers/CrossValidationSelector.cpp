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
		std::vector<real>* wtsExclude) : AbstractSelector(inIds, inType, inSeed, _logger, _error), fold(inFold) {

	// Calculate interval starts
	intervalStart.reserve(fold + 1);
	int index = 0;
	int fraction = N / fold;
	int extra = N - fraction * fold;
	for (int i = 0; i < fold; i++) {
		intervalStart.push_back(index);
		index += fraction;
		if (i < extra) {
			index++;
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

	weightsExclude = wtsExclude;
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

	std::fill(weights.begin(), weights.end(), 1.0);

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
			} else {
				weights[k] = 1.0; // TODO Is this necessary?
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
	return new (std::nothrow) CrossValidationSelector(*this); // default copy constructor
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

	if(weightsExclude){
		vector<int> permutationCopy = permutation;
		int nExcluded = 0;
		for(int i = 0; i < (int)weightsExclude->size(); i++){
			if(weightsExclude->at(i) != 0.0){
				nExcluded++;
			}
		}
		int fraction = nExcluded / fold;
		int extra = nExcluded - fraction * fold;

		vector<int> nExcludedPerFold;
		for(int i = 0; i < fold; i++){
			if(i < extra){
				nExcludedPerFold.push_back(fraction + 1);
			}
			else{
				nExcludedPerFold.push_back(fraction);
			}
		}
		int foldIncluded = 0;
		int foldExcluded = 0;
		int nextExcluded = intervalStart[0];
		int nextIncluded = intervalStart[0] + nExcludedPerFold[0];
		for(size_t i = 0; i < permutationCopy.size(); i++){
			if(weightsExclude->at(permutationCopy[i]) == 0.0){
				permutation[nextIncluded] = permutationCopy[i];
				nextIncluded++;
				if(nextIncluded == intervalStart[foldIncluded + 1]){
					nextIncluded = intervalStart[foldIncluded + 1] + nExcludedPerFold[foldIncluded + 1];
					foldIncluded++;
				}
			}
			else{
				permutation[nextExcluded] = permutationCopy[i];
				nextExcluded++;
				if(nextExcluded == intervalStart[foldExcluded] + nExcludedPerFold[foldExcluded]){
					nextExcluded = intervalStart[foldExcluded + 1];
					foldExcluded++;
				}
			}
		}
	}
}

} // namespace
