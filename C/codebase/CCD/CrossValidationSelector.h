/*
 * CrossValidation.h
 *
 *  Created on: Sep 9, 2010
 *      Author: msuchard
 */

#ifndef CROSSVALIDATION_H_
#define CROSSVALIDATION_H_

#include <vector>
#include "SCCSInputReader.h"

#include "AbstractSelector.h"
namespace bsccs {
class CrossValidationSelector : public AbstractSelector {
public:
	CrossValidationSelector(
			int inFold,
			std::vector<int>* inIds,
			SelectorType inType,
			long inSeed = 0);

	virtual ~CrossValidationSelector();

	void permute();

	void getWeights(int batch, std::vector<bsccs::real>& weights);

	void getComplement(std::vector<bsccs::real>& weights);

private:
	int fold;
	std::vector<int> permutation;
	std::vector<int> intervalStart;
};
}
#endif /* CROSSVALIDATION_H_ */
