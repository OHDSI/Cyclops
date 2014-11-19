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

CrossValidationSelector::CrossValidationSelector(
		int inFold,
		std::vector<int>* inIds,
		SelectorType inType,
		long inSeed,
		std::vector<real>* wtsExclude) : AbstractSelector(inIds, inType, inSeed), fold(inFold) {

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

	for (int i = 0; i < fold; i++) {
		std::cout << (intervalStart[i+1] - intervalStart[i]) << " ";
	}
	std::cout << std::endl;

	std::cout << "Performing " << fold << "-fold cross-validation [seed = "
		      << seed << "]" << std::endl;

	// Generate random permutation
	permutation.reserve(N);
	for (int i = 0; i < N; ++i) {
		permutation.push_back(i);
	}
	weightsExclude = wtsExclude;
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

	if (type == SUBJECT) {
		std:set<int> excludeSet;
		std::copy(
				permutation.begin() + intervalStart[batch],
				permutation.begin() + intervalStart[batch + 1],
				insert_iterator< std::set<int> >(excludeSet, excludeSet.begin())
				);

		for (int k = 0; k < K; k++) {
			if (excludeSet.find(ids->at(k)) != excludeSet.end()) { // found
				weights[k] = 0.0;
			} else {
				weights[k] = 1.0;
			}
		}
	} else {
		std::fill(weights.begin(), weights.end(), 0.0);
		std::fill(weights.begin(), weights.begin() + 100, 1.0);
	}
}

void CrossValidationSelector::getComplement(std::vector<real>& weights) {
	for(std::vector<real>::iterator it = weights.begin(); it != weights.end(); it++) {
		*it = 1 - *it;
	}
}

void CrossValidationSelector::permute() {

	// Do random shuffle
	if (!deterministic) {
		std::random_shuffle(permutation.begin(), permutation.end());
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
		for(int i = 0; i < permutationCopy.size(); i++){
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
