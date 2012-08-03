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

namespace BayesianSCCS {

class BootstrapSelector : public AbstractSelector {
public:
	BootstrapSelector(
			int inReplicates,
			std::vector<int>* inIds,
			SelectorType inType,
			long inSeed);

	virtual ~BootstrapSelector();

	virtual void permute();

	virtual void getWeights(int batch, std::vector<realTRS>& weights);

	virtual void getComplement(std::vector<realTRS>& weights);

private:
	std::multiset<int> selectedSet;
};
}
#endif /* BOOTSTRAPSELECTOR_H_ */
