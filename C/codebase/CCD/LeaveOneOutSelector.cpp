/*
 * LeaveOneOutSelector.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: msuchard
 */

#include <algorithm>
#include <iostream>

#include "LeaveOneOutSelector.h"

namespace bsccs {

	LeaveOneOutSelector::LeaveOneOutSelector(
		const std::vector<int>& inIds,
		SelectorType inType,
		long inSeed) : AbstractSelector(inIds, inType, inSeed) {

		for (int i = N-1; i >= 0; --i) {
			permutation.push_back(i);
		}
		currentID = -1;
	}

	LeaveOneOutSelector::~LeaveOneOutSelector() { /* nothing */ }

	void LeaveOneOutSelector::permute() {

		if (permutation.size() == 0) {
			fprintf(stderr, "Error in leave one out selector!\n");
			exit(-1);
		}
		// Do random shuffle
		if (!deterministic) {
			std::random_shuffle(permutation.begin(), permutation.end());
		}
		currentID = permutation[permutation.size() - 1];
		permutation.pop_back();
		std::cout << std::endl << "Leaving out entry " << (currentID + 1)
				<< " / " << N << std::endl;
	}

	void LeaveOneOutSelector::getWeights(int batch, std::vector<real>& weights) {
		if (weights.size() != K) {
			weights.resize(K);
		}

		std::fill(weights.begin(), weights.end(), static_cast<real>(1.0));

		if (batch == -1) {
			return;
		}
		for (int i = 0; i < K; ++i) {
			if (ids[i] == currentID) {
				weights[i] = static_cast<real>(0);
			}
		}
	}

	void LeaveOneOutSelector::getComplement(std::vector<real>& weights) {
		std::vector<real>::const_iterator end = weights.end();
		for(std::vector<real>::iterator it = weights.begin(); it != end; ++it) {
			*it = static_cast<real>(1.0) - *it;
		}
	}
} /* namespace bsccs */
