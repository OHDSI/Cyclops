/*
 * BootstrapSelector.h
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#ifndef BOOTSTRAPSELECTOR_H_
#define BOOTSTRAPSELECTOR_H_

#include <set>

#include "AbstractSelector.h"

namespace bsccs {

class BootstrapSelector : public AbstractSelector {
public:
	BootstrapSelector(
			int inReplicates,
			std::vector<int> inIds,
			SelectorType inType,
			long inSeed,
    	    loggers::ProgressLoggerPtr _logger,
	    	loggers::ErrorHandlerPtr _error,
			std::vector<real>* wtsExclude = NULL);

	virtual ~BootstrapSelector();

	virtual void permute();

	virtual void getWeights(int batch, std::vector<real>& weights);

	virtual void getComplement(std::vector<real>& weights);
	
	AbstractSelector* clone() const;

private:
	std::multiset<int> selectedSet;
	std::vector<int> indicesIncluded;
};

} // namespace

#endif /* BOOTSTRAPSELECTOR_H_ */
