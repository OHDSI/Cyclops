/*
 * NewGenericInputReader.h
 *
 *  Created on: Dec 18, 2012
 *      Author: msuchard
 */

#ifndef NEWGENERICINPUTREADER_H_
#define NEWGENERICINPUTREADER_H_

#include "BaseInputReader.h"
#include "Types.h"

class NewGenericInputReader : public BaseInputReader<NewGenericInputReader> {
public:

	NewGenericInputReader() : BaseInputReader<NewGenericInputReader>()
		, upcastToDense(false)
		, upcastToSparse(false)
		, useBBROutcome(false)
		, includeIntercept(false)
		, includeOffset(false)
		, includeRowLabel(false)
		, includeStratumLabel(false)
		, includeCensoredData(false)
		, includeCensoredData2(false)
		, includeWeights(false)
		, includeSCCSOffset(false)
		, indicatorOnly(false)
		, modelType(bsccs::Models::NONE)
	{
		// Do nothing
	}

	NewGenericInputReader(const bsccs::Models::ModelType model) : BaseInputReader<NewGenericInputReader>()
		, upcastToDense(false)
		, upcastToSparse(false)
		, useBBROutcome(false)
		, includeIntercept(false)
		, includeOffset(false)
		, includeRowLabel(false)
		, includeStratumLabel(false)
		, includeCensoredData(false)
		, includeCensoredData2(false)
		, includeWeights(false)
		, includeSCCSOffset(false)
		, indicatorOnly(false)
		, modelType(model)
	{
		setRequiredFlags(model);
//		std::cerr << bsccs::Models::RequiresStratumID << std::endl;
//		std::cerr << bsccs::Models::RequiresCensoredData << std::endl;
//		std::cerr << bsccs::Models::RequiresCensoredData2 << std::endl;
//		exit(-1);



	}

	void setRequiredFlags(const bsccs::Models::ModelType model) {
		using namespace bsccs::Models;
		includeStratumLabel = requiresStratumID(model);
		includeCensoredData = requiresCensoredData(model);
//		includeOffset = requiresOffset(model);
		includeSCCSOffset = requiresOffset(model);
		if (includeSCCSOffset) {
			includeOffset = false;
		}
//		std::cerr << "Model = " << model << std::endl;
//		std::cerr << includeStratumLabel << std::endl << std::endl;
	}

	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {

//		std::cerr << includeRowLabel << std::endl;
//		std::cerr << includeStratumLabel << std::endl;
//		std::cerr << includeWeights << std::endl;
//		std::cerr << includeOffset << std::endl;
//		std::cerr << ss << std::endl << std::endl;
//		exit(-1);

		if (includeRowLabel) {
			parseRowLabel(ss, rowInfo);
		}

		if (includeStratumLabel) {
			parseStratumEntry(ss,rowInfo);
		} else {
			parseNoStratumEntry(ss, rowInfo);
		}

		if (includeWeights) {
			// TODO
		}

		if (includeCensoredData) {
			parseSingleTimeEntry<real>(ss, rowInfo);
		}

		if (useBBROutcome) {
			parseSingleBBROutcomeEntry<int>(ss, rowInfo);
		} else {
			parseSingleOutcomeEntry<int>(ss, rowInfo);
		}

		if (includeSCCSOffset) {
			parseOffsetEntry(ss, rowInfo);
		} else if (includeOffset) {
			parseOffsetCovariateEntry(ss, rowInfo, offsetInLogSpace);
		}

		if (includeIntercept) {
			modelData->getColumn(columnIntercept).add_data(rowInfo.currentRow, static_cast<real>(1.0));
		}
		parseAllBBRCovariatesEntry(ss, rowInfo, indicatorOnly);
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
			indicatorOnly = includesOption(line, "indicator_only");
			if (includeOffset) {
				offsetInLogSpace = includesOption(line, "log_offset");
			}
			// Only set if true
			if (includesOption(line, "row_label")) {
				includeRowLabel = true;
			}
			if (includesOption(line, "stratum_label")) {
				includeStratumLabel = true;
			}
			if (includesOption(line, "weight")) {
				includeWeights = true;
			}
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
	bool offsetInLogSpace;
	bool includeRowLabel;
	bool includeStratumLabel;
	bool includeWeights;
	bool includeCensoredData;
	bool includeCensoredData2;
	bool includeSCCSOffset;
	bool indicatorOnly;

	bsccs::Models::ModelType modelType;
};


#endif /* NEWGENERICINPUTREADER_H_ */
