/*
 * AbstractSelector.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <algorithm>

#include "AbstractSelector.h"

AbstractSelector::AbstractSelector(
		std::vector<int>* inIds,
		SelectorType inType,
		long inSeed) : ids(inIds), type(inType), seed(inSeed), K(inIds->size()) {

	// Set up number of exchangeable objects
	if (type == SUBJECT) {
		N = *(std::max_element(ids->begin(), ids->end())) + 1;
	} else {
		N = ids->size();
	}

	// Set up seed
	if (seed == -1) {
		deterministic = true;
	} else {
		deterministic = false;
		if (seed == 0) 
		{
#ifdef _WIN32
			seed = time_t(NULL);
#else
			seed = time(NULL);
#endif
		}
		srand(seed);
	}
}

AbstractSelector::~AbstractSelector() {
	if (ids) {
		delete ids;
	}
}
