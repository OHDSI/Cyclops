/*
 * AbstractSelector.h
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#ifndef ABSTRACTSELECTOR_H_
#define ABSTRACTSELECTOR_H_

#include <vector>

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif

enum SelectorType {
	SUBJECT = 0,
	ENTRY  = 1
};

class AbstractSelector {
public:
	AbstractSelector(
			const std::vector<int>& inIds,
			SelectorType inType,
			long inSeed);

	virtual ~AbstractSelector();

	virtual void permute() = 0; // pure virtual

	virtual void getWeights(int batch, std::vector<real>& weights) = 0; // pure virtual

	virtual void getComplement(std::vector<real>& weights) = 0; // pure virtual

protected:
	const std::vector<int>& ids;
	SelectorType type;
	long seed;
	int K;
	int N;
	bool deterministic;
};

#endif /* ABSTRACTSELECTOR_H_ */
