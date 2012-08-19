/*
 * CCTestInputReader.h
 *
 *  Created on: Mar 31, 2012
 *      Author: msuchard
 */

#ifndef CCTESTINPUTREADER_H_
#define CCTESTINPUTREADER_H_

#include "InputReader.h"

class CCTestInputReader : public InputReader {
public:
	CCTestInputReader();
	virtual ~CCTestInputReader();

	virtual void readFile(const char* fileName);
};

#endif /* CCTESTINPUTREADER_H_ */
