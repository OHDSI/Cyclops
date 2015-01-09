/*
 * AbstractSelector.h
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#ifndef ABSTRACTSELECTOR_H_
#define ABSTRACTSELECTOR_H_

#include <vector>
#include <random>
#include <iostream> // TODO REMOVE

#include "Types.h"
#include "io/ProgressLogger.h"

namespace bsccs {

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif

class AbstractSelector {
public:
	AbstractSelector(
			std::vector<int> inIds,
			SelectorType inType,
			long inSeed,
			loggers::ProgressLoggerPtr _logger,
		    loggers::ErrorHandlerPtr _error);

	virtual ~AbstractSelector();

	virtual void permute() = 0; // pure virtual
	
	// TODO
	virtual void reseed() { /* std::cerr << "RESEED" << std::endl;*/ } // Do nothing by default

	virtual void getWeights(int batch, std::vector<real>& weights) = 0; // pure virtual

	virtual void getComplement(std::vector<real>& weights) = 0; // pure virtual
	
	virtual AbstractSelector* clone() const = 0; // pure virtual

protected:
	const std::vector<int> ids;
	SelectorType type;
	long seed;
	size_t K;
	size_t N;
	bool deterministic;
	std::mt19937 prng;
	
	
    loggers::ProgressLoggerPtr logger;
	loggers::ErrorHandlerPtr error;	
};

} // namespace

#endif /* ABSTRACTSELECTOR_H_ */
