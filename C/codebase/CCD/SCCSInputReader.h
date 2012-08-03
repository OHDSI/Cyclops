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

using namespace std;

#include "InputReader.h"
namespace BayesianSCCS {
class SCCSInputReader: public InputReader {
public:
	SCCSInputReader();
	virtual ~SCCSInputReader();

	virtual void readFile(const char* fileName);
};
}
#endif /* SCCSINPUTREADER_H_ */
