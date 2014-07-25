/*
 * NewCoxInputReader.h
 *
 *  Created on: Nov 14, 2012
 *      Author: msuchard
 */

#ifndef NEWCOXINPUTREADER_H_
#define NEWCOXINPUTREADER_H_

#include "io/BaseInputReader.h"

//#define UPCAST_DENSE 

namespace bsccs {

class NewCoxInputReader : public BaseInputReader<NewCoxInputReader> {
public:

	NewCoxInputReader(
		loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error) : BaseInputReader<NewCoxInputReader>(_logger, _error) {
		// Do nothing	
	}
	
	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {		
		parseNoStratumEntry(ss, rowInfo);
		parseSingleTimeEntry<float>(ss, rowInfo);
		parseSingleOutcomeEntry<int>(ss, rowInfo);	
		parseAllBBRCovariatesEntry(ss, rowInfo, false);
	}
	
	void parseHeader(ifstream& in) {
		// Do nothing
	}

#ifdef UPCAST_DENSE
	void upcastColumns(ModelData* modelData, RowInformation& rowInfo) {	
		cerr << "Going to up-cast all columns to dense!" << endl;
		for (int i = 0; i < modelData->getNumberOfColumns(); ++i) {
			modelData->getColumn(i).convertColumnToDense(rowInfo.currentRow);
		}
	}
#endif
	
};

} // namespace

#endif /* NEWCOXINPUTREADER_H_ */
