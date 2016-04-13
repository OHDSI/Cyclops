/*
 * BootstrapSelector.h
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#ifndef PROPORTIONSELECTOR_H_
#define PROPORTIONSELECTOR_H_

#include <set>

#include "AbstractSelector.h"

namespace bsccs {

class ProportionSelector : public AbstractSelector {
public:
	ProportionSelector(
			int inReplicates,
			std::vector<int> inIds,
			SelectorType inType,
			long inSeed,
    	    loggers::ProgressLoggerPtr _logger,
	    	loggers::ErrorHandlerPtr _error			
			);

	virtual ~ProportionSelector();

	virtual void permute();

	virtual void getWeights(int batch, std::vector<double>& weights);

	virtual void getComplement(std::vector<double>& weights);
	
	AbstractSelector* clone() const;

private:
	std::multiset<int> selectedSet;

	int total;

};

} // namespace

#endif /* PROPORTIONSELECTOR_H_ */
