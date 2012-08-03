/*
 * AbstractSelector.h
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#ifndef ABSTRACTSELECTOR_H_
#define ABSTRACTSELECTOR_H_

#include <vector>

// Trevor Shaddox changed from real to realTRS after namespace errors for #include <Eigen/Dense>
#ifdef DOUBLE_PRECISION
	typedef double realTRS;
#else
	typedef float realTRS;
#endif

enum SelectorType {
	SUBJECT = 0,
	ENTRY  = 1
};

class AbstractSelector {
public:
	AbstractSelector(
			std::vector<int>* inIds,
			SelectorType inType,
			long inSeed);

	virtual ~AbstractSelector();

	virtual void permute() = 0; // pure virtual

	virtual void getWeights(int batch, std::vector<realTRS>& weights) = 0; // pure virtual

	virtual void getComplement(std::vector<realTRS>& weights) = 0; // pure virtual

protected:
	std::vector<int>* ids;
	SelectorType type;
	long seed;
	int K;
	int N;
	bool deterministic;
};

#endif /* ABSTRACTSELECTOR_H_ */
