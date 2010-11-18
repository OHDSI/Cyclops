/*
 * CrossValidation.h
 *
 *  Created on: Sep 9, 2010
 *      Author: msuchard
 */

#ifndef CROSSVALIDATION_H_
#define CROSSVALIDATION_H_

#include <vector>
#include "InputReader.h"

#include "AbstractSelector.h"

enum CrossValidationType {
	SUBJECT = 0,
	ENTRY  = 1
};

class CrossValidationSelector : public AbstractSelector {
public:
	CrossValidationSelector(
			int inFold,
			std::vector<int>* inIds,
			CrossValidationType inType,
			long inSeed = 0);

	virtual ~CrossValidationSelector();

	void permute();

	void getWeights(int batch, std::vector<real>& weights);

	void getComplement(std::vector<real>& weights);

private:
	int fold;
	std::vector<int>* ids;
	CrossValidationType type;
	long seed;
	int K;
	int N;
	std::vector<int> permutation;
	std::vector<int> intervalStart;
	bool deterministic;
};

#endif /* CROSSVALIDATION_H_ */
