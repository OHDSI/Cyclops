/*
 * CoxInputReader.h
 *
 *  Created on: Mar 25, 2012
 *      Author: msuchard
 */

#ifndef COXINPUTREADER_H_
#define COXINPUTREADER_H_

#include "InputReader.h"

class CoxInputReader : public InputReader {
public:
	CoxInputReader();
	virtual ~CoxInputReader();

	virtual void readFile(const char* fileName);
};

#endif /* COXINPUTREADER_H_ */
