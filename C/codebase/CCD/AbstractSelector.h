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

class AbstractSelector {
public:
	AbstractSelector();

	virtual ~AbstractSelector();

	virtual void permute() = 0; // pure virtual

	virtual void getWeights(int batch, std::vector<real>& weights) = 0; // pure virtual

	virtual void getComplement(std::vector<real>& weights) = 0; // pure virtual
};

#endif /* ABSTRACTSELECTOR_H_ */
