/*
 * NewGenericInputReader.h
 *
 *  Created on: Dec 18, 2012
 *      Author: msuchard
 */

#ifndef NEWGENERICINPUTREADER_H_
#define NEWGENERICINPUTREADER_H_

#include "BaseInputReader.h"

class NewGenericInputReader : public BaseInputReader<NewGenericInputReader> {
public:

	NewGenericInputReader() : BaseInputReader<NewGenericInputReader>()
		, upcastToDense(false)
		, upcastToSparse(false) {
		// Do nothing
	}

	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {
		parseNoStratumEntry(ss, rowInfo);
		parseSingleOutcomeEntry<int>(ss, rowInfo);
		parseAllBBRCovariatesEntry(ss, rowInfo);
	}

	void parseHeader(ifstream& in) {
		int firstChar = in.peek();
		if (firstChar == '#') { // There is a header
			string line;
			getline(in, line);
			size_t found = line.find("dense");
			if (found != string::npos) {
				upcastToDense = true;
			}
			found = line.find("sparse");
			if (found != string::npos) {
				upcastToSparse = true;
			}
		}
	}

	void upcastColumns(ModelData* modelData, RowInformation& rowInfo) {
		if (upcastToSparse) {
			cerr << "Going to up-cast all columns to sparse!" << endl;
			for (int i = 0; i < modelData->getNumberOfColumns(); ++i) {
				modelData->getColumn(i).convertColumnToSparse();
			}
		}
		if (upcastToDense) {
			cerr << "Going to up-cast all columns to dense!" << endl;
			for (int i = 0; i < modelData->getNumberOfColumns(); ++i) {
				modelData->getColumn(i).convertColumnToDense(rowInfo.currentRow);
			}
		}
	}

private:
	bool upcastToDense;
	bool upcastToSparse;
};


#endif /* NEWGENERICINPUTREADER_H_ */
