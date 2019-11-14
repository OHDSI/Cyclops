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

    template <typename RealType>
class NewCoxInputReader : public BaseInputReader<RealType, NewCoxInputReader<RealType>> {
public:

    using BaseInputReader<RealType, NewCoxInputReader<RealType>>::addEventEntry;
    using BaseInputReader<RealType, NewCoxInputReader<RealType>>::modelData;
    using BaseInputReader<RealType, NewCoxInputReader<RealType>>::parseSingleTimeEntry;
    using BaseInputReader<RealType, NewCoxInputReader<RealType>>::parseSingleOutcomeEntry;
    using BaseInputReader<RealType, NewCoxInputReader<RealType>>::parseAllBBRCovariatesEntry;

	NewCoxInputReader(
		loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error) : BaseInputReader<RealType, NewCoxInputReader<RealType>>(_logger, _error),
                                           hasWeights(false), castToDense(false) {
		// Do nothing	
	}

    void parseNoStratumEntry(stringstream& ss, RowInformation<RealType>& rowInfo) {
        addEventEntry(1);
        push_back_pid(*modelData, 1);
        //modelData->pid.push_back(rowInfo.numCases);
        rowInfo.numCases++;
    }

    template <typename T>
    inline void parseWeightEntry(stringstream& ss, RowInformation<RealType>& rowInfo) {
        T thisY;
        ss >> thisY;
        push_back_z(*modelData, thisY);
    }

    inline void parseRow(stringstream& ss, RowInformation<RealType>& rowInfo) {
//	    std::cout << "parseRow in newcoxreader" << '\n';
        parseNoStratumEntry(ss, rowInfo); // push_back_nevents & push_back_pid
        parseSingleTimeEntry(ss, rowInfo); // push_back_offs
        parseSingleOutcomeEntry(ss, rowInfo); // push_back_y
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
        index = line.find("dense");
        if (index != string::npos) {
            castToDense = true;
        }
    }

    void upcastColumns(ModelData<RealType>* modelData, RowInformation<RealType>& rowInfo) {
        if (castToDense) {
            cerr << "Going to up-cast all columns to dense!" << endl;
            modelData->convertAllCovariatesToDense(rowInfo.currentRow);
//            modelData->convertCovariateToDense(rowInfo.currentRow);
        }
    }

private:
    bool hasWeights;
    bool castToDense;

};

} // namespace

#endif /* NEWCOXINPUTREADER_H_ */