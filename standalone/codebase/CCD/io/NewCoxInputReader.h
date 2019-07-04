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
		loggers::ErrorHandlerPtr _error) : BaseInputReader<NewCoxInputReader>(_logger, _error),
		        hasWeights(false) {
		// Do nothing	
	}

	void parseNoStratumEntry(stringstream& ss, RowInformation& rowInfo) {
		addEventEntry(1);
		push_back_pid(*modelData, 1);
		//modelData->pid.push_back(rowInfo.numCases);
		rowInfo.numCases++;
	}

	template <typename T>
	inline void parseWeightEntry(stringstream& ss, RowInformation& rowInfo) {
        T thisY;
        ss >> thisY;
        push_back_z(*modelData, thisY);
	}

	inline void parseRow(stringstream& ss, RowInformation& rowInfo) {		
		parseNoStratumEntry(ss, rowInfo);
		parseSingleTimeEntry<double>(ss, rowInfo);
		parseSingleOutcomeEntry<int>(ss, rowInfo);
		if (hasWeights) {
		    parseWeightEntry<double>(ss, rowInfo);
		}
		parseAllBBRCovariatesEntry(ss, rowInfo, false);
	}
	
	void parseHeader(ifstream& in) {
        string line;
        getline(in, line); // Read header
        auto index = line.find("has_weights");
        if (index != string::npos) {
            hasWeights = true;
        }
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

private:
    bool hasWeights;
	
};

} // namespace

#endif /* NEWCOXINPUTREADER_H_ */
