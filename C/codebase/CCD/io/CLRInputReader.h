/*
 * CLRInputReader.h
 *
 *  Created on: April, 2012
 *      Author: msuchard
 */

#ifndef CLRINPUTREADER_H_
#define CLRINPUTREADER_H_

#include <iostream>
#include <fstream>

#include <vector>
#include <map>

using namespace std;

#include "InputReader.h"

class CLRInputReader: public InputReader {
public:
	CLRInputReader();
	virtual ~CLRInputReader();

	virtual void readFile(const char* fileName);
};

#endif /* CLRINPUTREADER_H_ */
