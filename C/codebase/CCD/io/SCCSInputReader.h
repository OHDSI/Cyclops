/*
 * InputReader.h
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#ifndef SCCSINPUTREADER_H_
#define SCCSINPUTREADER_H_

#include <iostream>
#include <fstream>

#include <vector>
#include <map>

#include "InputReader.h"

namespace bsccs {

using namespace std;

class SCCSInputReader: public InputReader {
public:
	SCCSInputReader(DataSource* dataSource);
	virtual ~SCCSInputReader();

	virtual void readFile(const char* fileName);
};

} // namespace

#endif /* SCCSINPUTREADER_H_ */
