/*
 * AbstractSelector.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <algorithm>

#include "AbstractSelector.h"

namespace bsccs {

AbstractSelector::AbstractSelector(
		std::vector<int> inIds,
		SelectorType inType,
		long inSeed,
	    loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error		
		) : ids(inIds), type(inType), seed(inSeed), K(ids.size()), logger(_logger), error(_error) {

	// Set up number of exchangeable objects
	if (type == SelectorType::BY_PID) {
		N = *(std::max_element(ids.begin(), ids.end())) + 1; 
		// Assumes that smallest ids == 0 and they are consecutive
	} else {
		N = ids.size();
	}

	// Set up seed
	if (seed == -1) {
		deterministic = true;
	} else {
		deterministic = false;
		if (seed == -99) {
#ifdef _WIN32

			seed = time_t(NULL);

#else

			seed = time(NULL);

#endif
		}
		prng.seed(seed);
	}
}

AbstractSelector::~AbstractSelector() {
// 	if (ids) {
// 		delete ids;
// 	}
}

} // namespace
