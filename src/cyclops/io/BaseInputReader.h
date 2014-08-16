/*
 * GenericSparseReader.h
 *
 *  Created on: Sep 11, 2012
 *      Author: msuchard
 */

#ifndef GENERICSPARSEREADER_H_
#define GENERICSPARSEREADER_H_

#include <vector>

#include "InputReader.h"
#include "SparseIndexer.h"
#include "io/ProgressLogger.h"
 
#define MAX_ENTRIES		1000000000
#define MISSING_STRING	"NA"

namespace bsccs {

using std::string;
//using std::cerr;
//using std::cout;
using std::endl;
using std::ifstream;

typedef std::vector<std::string> string_vector;

struct RowInformation {
	int currentRow;
	int numCases;
	int numEvents;
	string outcomeId;
	string currentPid;
	SparseIndexer indexer;
	string_vector scratch;

	RowInformation(int _currentRow, int _numCases, int _numEvents,
			string _outcomeId, string _currentPid,
			SparseIndexer _indexer) : currentRow(_currentRow), numCases(_numCases),
			numEvents(_numEvents), outcomeId(_outcomeId), currentPid(_currentPid),
			indexer(_indexer) {
		// Do nothing
	}
};

struct NoMissingPolicy {
	// No API at moment
	bool isMissing(real value) {
		return false;
	}

	void hook1() {
		// Do nothing
	}

	void hook2() {
		// Do nothing
	}
};

template <typename DerivedFormat, typename Missing = NoMissingPolicy>
class BaseInputReader : public InputReader, Missing {
public:
	typedef std::vector<int> int_vector;
	typedef std::vector<string> string_vector;

//	BaseInputReader() : InputReader(
//		bsccs::make_shared<loggers::CoutLogger>(),
//	 	bsccs::make_shared<loggers::CerrErrorHandler>()), innerDelimitor(":") {
//		// Do nothing		
//	}
	
	BaseInputReader(
		loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error) : InputReader(_logger, _error), innerDelimitor(":") {
		// Do nothing	
	}

	virtual ~BaseInputReader() {
		// Do nothing
	}

	virtual void readFile(const char* fileName) {

		ifstream in(fileName);
		if (!in) {		
			std::ostringstream stream;
			stream << "Unable to open " << fileName;			
			error->throwError(stream);
		}
		
// 		in.close();
// 		return;

		// Initial values
		RowInformation rowInfo(0,0,0, MISSING_STRING, MISSING_STRING, *modelData);
		string line;

		try {
			static_cast<DerivedFormat*>(this)->parseHeader(in);

			static_cast<DerivedFormat*>(this)->addFixedCovariateColumns();

			while (getline(in, line) && (rowInfo.currentRow < MAX_ENTRIES)) {
				if (!line.empty()) {
					stringstream ss(line.c_str()); // Tokenize
					static_cast<DerivedFormat*>(this)->parseRow(ss, rowInfo);
					rowInfo.currentRow++;
				}
			}
			addEventEntry(rowInfo.numEvents); // Save last patient

		} catch (...) {
			std::ostringstream stream;
			stream << "Exception while trying to read " << fileName;
			in.close();
			error->throwError(stream);			
		}
		
		static_cast<DerivedFormat*>(this)->upcastColumns(modelData, rowInfo);

		doSort(); // Override for sort criterion or no sorting

		std::ostringstream stream;
		stream << "Number of rows: " << rowInfo.currentRow << " read from " << fileName << endl;
		stream << "Number of cases: " << rowInfo.numCases << endl;
		stream << "Number of covariates: " <<  modelData->getNumberOfColumns();
		logger->writeLine(stream);

		modelData->nPatients = rowInfo.numCases;
		modelData->nRows = rowInfo.currentRow;
		modelData->conditionId = rowInfo.outcomeId;
		
		in.close();
	}	 
	
protected:
	void parseHeader(ifstream& in) {
		string line;
		getline(in, line); // Read header
	}
	
	void upcastColumns(ModelData* modelData, RowInformation& rowInfo) {
		// Do nothing
	}

	void parseAllBBRCovariatesEntry(stringstream& ss, RowInformation& rowInfo, bool indicatorOnly) {
		string entry;
//		int count = 0;
		while (ss >> entry) {
			rowInfo.scratch.clear();
			IdType drug;
			real value;
			if (indicatorOnly) {
				drug = atoi(entry.c_str());
				value = static_cast<real>(1);
			} else {
				split(rowInfo.scratch, entry, getInnerDelimitor());
				drug = atoi(rowInfo.scratch[0].c_str());
				value = atof(rowInfo.scratch[1].c_str());
			}
			if (!rowInfo.indexer.hasColumn(drug)) {
				// Add new column
				rowInfo.indexer.addColumn(drug, INDICATOR);
				Missing::hook1(); // Handle allocation if necessary
			}

			CompressedDataColumn& column = rowInfo.indexer.getColumn(drug);
			if (value != static_cast<real>(1) && value != static_cast<real>(0)) {
				if (column.getFormatType() == INDICATOR) {
					std::ostringstream stream;
					stream << "Up-casting covariate " << column.getLabel() << " to sparse!";
					logger->writeLine(stream);
					column.convertColumnToSparse();
				}
			}

			if (Missing::isMissing(value)) {
				// Handle missing values
				Missing::hook2();
			} else {
				// Add to storage
				bool valid = column.add_data(rowInfo.currentRow, value);
				if (!valid) {
					std::ostringstream stream;
					stream << "Warning: repeated sparse entries in data row: "
							<< (rowInfo.currentRow + 1)
							<< ", column: " << column.getLabel();
					logger->writeLine(stream);
				}
			}
		}
	}		

	void doSort() {
		// Easy to sort columns now in AOS format
		modelData->sortColumns(CompressedDataColumn::sortNumerically);
	}

	void addFixedCovariateColumns(void) {
		// Do nothing
	}

	void parseConditionEntry(stringstream& ss,
			RowInformation& rowInfo) {
		string currentOutcomeId;
		ss >> currentOutcomeId;
		if (rowInfo.outcomeId == MISSING_STRING) {
			rowInfo.outcomeId = currentOutcomeId;
		} else if (currentOutcomeId != rowInfo.outcomeId) {
			std::ostringstream stream;
			stream << "More than one condition ID in input file";
			error->throwError(stream);			
		}
	}

	void parseNoStratumEntry(stringstream& ss, RowInformation& rowInfo) {
		addEventEntry(1);
		modelData->pid.push_back(rowInfo.numCases);
		rowInfo.numCases++;
	}

	void parseRowLabel(stringstream& ss, RowInformation& rowInfo) {
		string label;
		ss >> label;
		modelData->labels.push_back(label);
	}

	void parseStratumEntry(stringstream& ss, RowInformation& rowInfo) {
		string unmappedPid;
		ss >> unmappedPid;
		if (unmappedPid != rowInfo.currentPid) { // New patient, ASSUMES these are sorted
			if (rowInfo.currentPid != MISSING_STRING) { // Skip first switch
				addEventEntry(rowInfo.numEvents);
				rowInfo.numEvents = 0;
			}
			rowInfo.currentPid = unmappedPid;
			rowInfo.numCases++;
		}
		modelData->pid.push_back(rowInfo.numCases - 1);
	}

	template <typename T>
	void parseSingleOutcomeEntry(stringstream& ss, RowInformation& rowInfo) {
		T thisY;
		ss >> thisY;
		rowInfo.numEvents += thisY;
		modelData->y.push_back(thisY);
	}
	
	template <typename T>
	void parseSingleTimeEntry(stringstream& ss, RowInformation& rowInfo) {
		T thisY;
		ss >> thisY;		
		modelData->z.push_back(thisY);
	}	

	template <typename T>
	void parseSingleBBROutcomeEntry(stringstream& ss, RowInformation& rowInfo) {
		T thisY;
		ss >> thisY;
		if (thisY < static_cast<T>(0)) { // BBR uses +1 / -1, BSCCS uses 1 / 0.
			thisY = static_cast<T>(0);
		}
		rowInfo.numEvents += thisY;
		modelData->y.push_back(thisY);
	}

	void parseOffsetCovariateEntry(stringstream& ss, RowInformation& rowInfo, bool inLogSpace) {
		real thisOffset;
		ss >> thisOffset;
		if (!inLogSpace) {
			thisOffset = std::log(thisOffset);
		}
		modelData->getColumn(0).add_data(rowInfo.currentRow, thisOffset);
	}

	void parseOffsetEntry(stringstream& ss, RowInformation&) {
		real thisOffs;
		ss >> thisOffs;
		modelData->offs.push_back(thisOffs);
	}

	void parseAllIndicatorCovariatesEntry(stringstream& ss, RowInformation& rowInfo) {
		IdType drug;
		while (ss >> drug) {
			if (drug == 0) { // No drug
				// Do nothing
			} else {
				if (!rowInfo.indexer.hasColumn(drug)) {
					// Add new column
					rowInfo.indexer.addColumn(drug, INDICATOR);
				}
				// Add to CSC storage
				rowInfo.indexer.getColumn(drug).add_data(rowInfo.currentRow, 1.0);
			}
		}
	}
	
	// Inlined functions
	const string& getInnerDelimitor() {
		return innerDelimitor;
	}

	void setInnerDelimitor(const string& d) { innerDelimitor = d; }

	void addEventEntry(int numEvents) {
		modelData->nevents.push_back(numEvents);
	}
	
// 	loggers::ProgressLoggerPtr logger;
// 	loggers::ErrorHandlerPtr error;

private:
	string innerDelimitor;
};

} // namespace

#endif /* GENERICSPARSEREADER_H_ */
