/*
 * CCTestInputReader.h
 *
 *  Created on: Mar 31, 2012
 *      Author: msuchard
 */

#ifndef CCTESTINPUTREADER_H_
#define CCTESTINPUTREADER_H_

#include "InputReader.h"

namespace bsccs {

class CCTestInputReader : public InputReader {
public:
	CCTestInputReader(DataSource* dataSource);
	virtual ~CCTestInputReader();

	virtual void readFile(const char* fileName);
};

} // namespace

#endif /* CCTESTINPUTREADER_H_ */
