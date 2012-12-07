/*
 * NewCoxInputReader.h
 *
 *  Created on: Nov 14, 2012
 *      Author: msuchard
 */

#ifndef NEWCOXINPUTREADER_H_
#define NEWCOXINPUTREADER_H_

#include "BaseInputReader.h"

//#define UPCAST_DENSE 

class NewCoxInputReader : public BaseInputReader<NewCoxInputReader> {
public:
	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {		
		parseNoStratumEntry(ss, rowInfo);
		parseSingleTimeEntry<float>(ss, rowInfo);
		parseSingleOutcomeEntry<int>(ss, rowInfo);	
		parseAllBBRCovariatesEntry(ss, rowInfo);
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

#endif /* NEWCOXINPUTREADER_H_ */
