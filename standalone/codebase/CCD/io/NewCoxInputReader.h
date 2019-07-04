/*
 * NewCoxInputReader.h
 *
 *  Created on: Nov 14, 2012
 *      Author: msuchard
 */

#ifndef NEWCOXINPUTREADER_H_
#define NEWCOXINPUTREADER_H_

#include "io/BaseInputReader.h"

#define UPCAST_DENSE

namespace bsccs {

class NewCoxInputReader : public BaseInputReader<NewCoxInputReader> {
public:

	NewCoxInputReader(
		loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error) : BaseInputReader<NewCoxInputReader>(_logger, _error) {
		// Do nothing	
	}

	void parseNoStratumEntry(stringstream& ss, RowInformation& rowInfo) {
		addEventEntry(1);
		push_back_pid(*modelData, 1);
		//modelData->pid.push_back(rowInfo.numCases);
		rowInfo.numCases++;
	}
	
	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {		
		parseNoStratumEntry(ss, rowInfo);
		parseSingleTimeEntry<double>(ss, rowInfo);
		parseSingleOutcomeEntry<int>(ss, rowInfo);	
		parseAllBBRCovariatesEntry(ss, rowInfo, false);
	}
	
	void parseHeader(ifstream& in) {
		// Do nothing
	}

#ifdef UPCAST_DENSE
	void upcastColumns(ModelData<double>* modelData, RowInformation& rowInfo) {
		cerr << "Going to up-cast all columns to dense!" << endl;
//		for (int i = 0; i < modelData->getNumberOfColumns(); ++i) {
//			modelData->convertCovariateToDense(i + 1);
//			modelData->getColumn(index).convertColumnToDense(rowInfo.currentRow);
//		}
		modelData->convertAllCovariatesToDense(rowInfo.currentRow);
	}
#endif
	
};

} // namespace

#endif /* NEWCOXINPUTREADER_H_ */
