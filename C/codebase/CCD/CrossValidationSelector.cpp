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

CrossValidationSelector::CrossValidationSelector(
		int inFold,
		std::vector<int>* inIds,
		CrossValidationType inType,
		long inSeed) : fold(inFold), ids(inIds), type(inType), seed(inSeed),
					   K(inIds->size()) {

//	std::cerr << "id length = " << ids->size() << std::endl;
	if (type == SUBJECT) {
		N = *(std::max_element(ids->begin(), ids->end())) + 1;
	} else {
		N = ids->size();
	}
//	std::cerr << "# objects = " << N << std::endl;

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

//	std::copy(intervalStart.begin(), intervalStart.end(),
//			ostream_iterator<int> (std::cout, " "));
//	std::cout << std::endl;
//
	for (int i = 0; i < fold; i++) {
		std::cout << (intervalStart[i+1] - intervalStart[i]) << " ";
	}
	std::cout << std::endl;

	if (seed == -1) {
		deterministic = true;
	} else {
		deterministic = false;
		if (seed == 0) {
			seed = time(NULL);
		}
		srand(seed);
	}

	std::cout << "Performing " << fold << "-fold cross-validation [seed = "
		      << seed << "]" << std::endl;

	// Generate random permutation
	permutation.reserve(N);
	for (int i = 0; i < N; ++i) {
		permutation.push_back(i);
	}
}

CrossValidationSelector::~CrossValidationSelector() {
	if (ids) {
		delete ids;
	}
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
	if (!deterministic) {
		std::random_shuffle(permutation.begin(), permutation.end());
	}
}
