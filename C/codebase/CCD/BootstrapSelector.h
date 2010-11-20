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

class BootstrapSelector : public AbstractSelector {
public:
	BootstrapSelector(
			int inReplicates,
			std::vector<int>* inIds,
			SelectorType inType,
			long inSeed);

	virtual ~BootstrapSelector();

	virtual void permute();

	virtual void getWeights(int batch, std::vector<real>& weights);

	virtual void getComplement(std::vector<real>& weights);

private:
	std::multiset<int> selectedSet;
};

#endif /* BOOTSTRAPSELECTOR_H_ */
