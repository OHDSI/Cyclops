/*
 * NewCLRInputReader.h
 *
 *  Created on: Sep 13, 2012
 *      Author: msuchard
 */

#ifndef NEWCLRINPUTREADER_H_
#define NEWCLRINPUTREADER_H_

#include "BaseInputReader.h"

class NewCLRInputReader : public BaseInputReader<NewCLRInputReader> {
public:
	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {
		parseSingleBBROutcomeEntry<int>(ss, rowInfo);
		parseStratumEntry(ss, rowInfo);
		parseAllBBRCovariatesEntry(ss, rowInfo);
	}

	void parseHeader(ifstream& in) {
		// Do nothing
	}

};

#endif /* NEWCLRINPUTREADER_H_ */
