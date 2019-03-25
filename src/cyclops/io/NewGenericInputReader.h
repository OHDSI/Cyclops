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

namespace bsccs {

class NewGenericInputReader : public BaseInputReader<NewGenericInputReader> {
public:

	NewGenericInputReader(
		loggers::ProgressLoggerPtr logger,
		loggers::ErrorHandlerPtr error) : BaseInputReader<NewGenericInputReader>(logger, error)
		, upcastToDense(false)
		, upcastToSparse(false)
		, useBBROutcome(false)
		, includeIntercept(false)
		, includeOffset(false)
		, includeRowLabel(false)
		, includeStratumLabel(false)
		, includeCensoredData(false)
	//	, includeCensoredData2(false)
		, includeWeights(false)
		, includeSCCSOffset(false)
		, indicatorOnly(false)
	//	, modelType(bsccs::Models::NONE)
	{
		// Do nothing
	}

// 	NewGenericInputReader(const bsccs::ModelType model) : BaseInputReader<NewGenericInputReader>()
// 		, upcastToDense(false)
// 		, upcastToSparse(false)
// 		, useBBROutcome(false)
// 		, includeIntercept(false)
// 		, includeOffset(false)
// 		, includeRowLabel(false)
// 		, includeStratumLabel(false)
// 		, includeCensoredData(false)
// 	//	, includeCensoredData2(false)
// 		, includeWeights(false)
// 		, includeSCCSOffset(false)
// 		, indicatorOnly(false)
// 	//	, modelType(model)
// 	{
// 		setRequiredFlags(model);
// 	}

	NewGenericInputReader(const bsccs::ModelType model,
			loggers::ProgressLoggerPtr logger, loggers::ErrorHandlerPtr error) : BaseInputReader<NewGenericInputReader>(logger, error)
		, upcastToDense(false)
		, upcastToSparse(false)
		, useBBROutcome(false)
		, includeIntercept(false)
		, includeOffset(false)
		, includeRowLabel(false)
		, includeStratumLabel(false)
		, includeCensoredData(false)
	//	, includeCensoredData2(false)
		, includeWeights(false)
		, includeSCCSOffset(false)
		, indicatorOnly(false)
	//	, modelType(model)
	{
		setRequiredFlags(model);
	}

	void setRequiredFlags(const bsccs::ModelType model) {
		using namespace bsccs::Models;
		includeStratumLabel = requiresStratumID(model);
		includeCensoredData = requiresCensoredData(model);
//		includeOffset = requiresOffset(model);
		includeSCCSOffset = requiresOffset(model);
		if (includeSCCSOffset) {
			includeOffset = false;
		}
	}

	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {

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
			parseSingleTimeEntry<double>(ss, rowInfo);
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
			modelData->getX().getColumn(columnIntercept).add_data(rowInfo.currentRow, static_cast<double>(1.0));
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
			if (!includeSCCSOffset) {
				includeOffset = includesOption(line, "offset");
			} else {
				includeOffset = false;
			}
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
			modelData->getX().push_back(DENSE); // Column 0
			modelData->setHasOffsetCovariate(true);
			modelData->getX().getColumn(0).add_label(-1);
		}
		if (includeIntercept) {
			modelData->getX().push_back(DENSE); // Column 0 or 1
			modelData->setHasInterceptCovariate(true);
			columnIntercept = modelData->getNumberOfColumns() - 1;
		}
	}

	void upcastColumns(ModelData<double>* modelData, RowInformation& rowInfo) {
		if (upcastToSparse) {
			std::ostringstream stream;
			stream << "Going to up-cast all columns to sparse!";
			logger->writeLine(stream);
			size_t i = includeIntercept ? 1 : 0;
			for (; i < modelData->getNumberOfColumns(); ++i) {
				modelData->getX().getColumn(i).convertColumnToSparse();
			}
		}
		if (upcastToDense) {
			std::ostringstream stream;
			stream << "Going to up-cast all columns to dense!";
			logger->writeLine(stream);
			for (size_t i = 0; i < modelData->getNumberOfColumns(); ++i) {
				modelData->getX().getColumn(i).convertColumnToDense(rowInfo.currentRow);
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
	bool useBBROutcome;
	bool includeIntercept;
	bool includeOffset;
	bool includeRowLabel;
	bool includeStratumLabel;
	bool includeCensoredData;
//	bool includeCensoredData2;
	bool includeWeights;
	bool includeSCCSOffset;
	bool indicatorOnly;

//	bsccs::Models::ModelType modelType;

	int columnIntercept;
	bool offsetInLogSpace;
};

} // namespace

#endif /* NEWGENERICINPUTREADER_H_ */
