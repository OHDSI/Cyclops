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
		, upcastToSparse(false)
		, useBBROutcome(false)
		, includeIntercept(false)
		, includeOffset(false)
		, includeRowLabel(false)
	{
		// Do nothing
	}

	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {
		if (includeRowLabel) {
			parseRowLabel(ss, rowInfo);
		}

		parseNoStratumEntry(ss, rowInfo);

		if (useBBROutcome) {
			parseSingleBBROutcomeEntry<int>(ss, rowInfo);
		} else {
			parseSingleOutcomeEntry<int>(ss, rowInfo);
		}

		if (includeOffset) {
			parseOffsetCovariateEntry(ss, rowInfo, logOffset);
		}

		if (includeIntercept) {
			modelData->getColumn(columnIntercept).add_data(rowInfo.currentRow, static_cast<real>(1.0));
		}
		parseAllBBRCovariatesEntry(ss, rowInfo);
	}

	void parseHeader(ifstream& in) {
		int firstChar = in.peek();
		if (firstChar == '#') { // There is a header
			string line;
			getline(in, line);
			upcastToDense = includesOption(line, "dense");
			upcastToSparse = includesOption(line, "sparse");
			useBBROutcome = includesOption(line, "bbr_outcome");
			includeIntercept = includesOption(line, "add_intercept");
			includeOffset = includesOption(line, "offset");
			if (includeOffset) {
				logOffset = includesOption(line, "log_offset");
			}
			includeRowLabel = includesOption(line, "row_label");
		}
	}

	void addFixedCovariateColumns(void) {
		if (includeOffset) {
			modelData->push_back(DENSE); // Column 0
			modelData->setHasOffsetCovariate(true);
		}
		if (includeIntercept) {
			modelData->push_back(DENSE); // Column 0 or 1
			columnIntercept = modelData->getNumberOfColumns() - 1;
		}
	}

	void upcastColumns(ModelData* modelData, RowInformation& rowInfo) {
		if (upcastToSparse) {
			cerr << "Going to up-cast all columns to sparse!" << endl;
			int i = includeIntercept ? 1 : 0;
			for (; i < modelData->getNumberOfColumns(); ++i) {
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

	bool includesOption(const string& line, const string& option) {
		size_t found = line.find(option);
		return found != string::npos;
	}

	bool upcastToDense;
	bool upcastToSparse;
	bool includeIntercept;
	bool includeOffset;
	bool useBBROutcome;

	int columnIntercept;
	bool logOffset;
	bool includeRowLabel;
};


#endif /* NEWGENERICINPUTREADER_H_ */
