/*
 * RTestInputReader.h
 *
 *  Created on: Mar 25, 2012
 *      Author: msuchard
 */

#ifndef RTESTINPUTREADER_H_
#define RTESTINPUTREADER_H_

#include "InputReader.h"
namespace BayesianSCCS {
class RTestInputReader : public InputReader {
public:
	RTestInputReader();
	virtual ~RTestInputReader();

	virtual void readFile(const char* fileName);
};
}
#endif /* RTESTINPUTREADER_H_ */
