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

class ProportionSelector : public AbstractSelector {
public:
	ProportionSelector(
			int inReplicates,
			const std::vector<int>& inIds,
			SelectorType inType,
			long inSeed);

	virtual ~ProportionSelector();

	virtual void permute();

	virtual void getWeights(int batch, std::vector<real>& weights);

	virtual void getComplement(std::vector<real>& weights);

private:
	std::multiset<int> selectedSet;

	int total;

};

#endif /* PROPORTIONSELECTOR_H_ */
