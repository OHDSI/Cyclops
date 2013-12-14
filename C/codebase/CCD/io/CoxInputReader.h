/*
 * CoxInputReader.h
 *
 *  Created on: Mar 25, 2012
 *      Author: msuchard
 */

#ifndef COXINPUTREADER_H_
#define COXINPUTREADER_H_

#include "InputReader.h"

namespace bsccs {

class CoxInputReader : public InputReader {
public:
	CoxInputReader(DataSource* dataSource);
	virtual ~CoxInputReader();

	virtual void readFile(const char* fileName);
};

} // namespace

#endif /* COXINPUTREADER_H_ */
