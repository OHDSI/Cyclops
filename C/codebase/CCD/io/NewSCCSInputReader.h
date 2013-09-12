/*
 * NewSCCSInputReader.h
 *
 *  Created on: Sep 13, 2012
 *      Author: msuchard
 */

#ifndef NEWSCCSINPUTREADER_H_
#define NEWSCCSINPUTREADER_H_

#include "BaseInputReader.h"

namespace bsccs {

class NewSCCSInputReader : public BaseInputReader<NewSCCSInputReader> {
public:
	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {
		parseConditionEntry(ss, rowInfo);
		parseStratumEntry(ss, rowInfo);
		parseSingleOutcomeEntry<int>(ss, rowInfo);
		parseOffsetEntry(ss, rowInfo);
		parseAllIndicatorCovariatesEntry(ss, rowInfo);
	}

protected:

};

} // namespace

#endif /* NEWSCCSINPUTREADER_H_ */
