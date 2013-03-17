/*
 * OutputWriter.h
 *
 *  Created on: Feb 15, 2013
 *      Author: msuchard
 */

#ifndef OUTPUTWRITER_H_
#define OUTPUTWRITER_H_


//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <vector>
//#include <map>
//#include <cstdlib>
//#include <cstring>
//
//#include <fstream>
//#include <iostream>
#include "../CyclicCoordinateDescent.h"
#include "../ModelData.h"

namespace bsccs {

// Define interface

class OutputWriter {
public:

	enum DisplayMask {
		ALL, WEIGHTED, UNWEIGHTED
	};

	OutputWriter() { }
	virtual ~OutputWriter() { }

	virtual void openFile(const char* fileName) = 0;
	virtual void writeFile() = 0;
	virtual void closeFile() = 0;
	virtual void setDisplayMask(DisplayMask use) = 0;
};

namespace OutputHelper {
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

struct RowInformation {
	int currentRow;

	RowInformation(int _currentRow) : currentRow(_currentRow) {
		// Do nothing
	}
};
}

// Define base class

template <typename DerivedFormat, typename Missing = OutputHelper::NoMissingPolicy>
class BaseOutputWriter : public OutputWriter, Missing {
public:
	BaseOutputWriter(const CyclicCoordinateDescent& _ccd, const ModelData& _data) :
		OutputWriter(), ccd(_ccd), data(_data), delimitor("\t") {
		setDisplayMask(ALL);
	}
	virtual ~BaseOutputWriter() {
		// Do nothing
	}

	void openFile(const char* fileName) {
		out.open(fileName, ios::out);
	}

	void closeFile() {
		out.close();
	}

	void writeFile() {
		if (out) {
			writeFile(out);
		}
	}

	void setDisplayMask(DisplayMask usage) {
		switch (usage) {
			case WEIGHTED :
				useWeightAsMask = true;
				useComplementWeight = false;
				break;
			case UNWEIGHTED :
				useWeightAsMask = true;
				useComplementWeight = true;
				break;
			default :
				useWeightAsMask = false;
				break;
		}
	}

protected:
	void writeFile(ofstream& out) {

		static_cast<DerivedFormat*>(this)->preprocessAllRows();

		OutputHelper::RowInformation rowInformation(0);
		while(rowInformation.currentRow < data.getNumberOfRows()) {
			if (displayRow(rowInformation.currentRow)) {
				static_cast<DerivedFormat*>(this)->writeRow(out, rowInformation);
			}
			++(rowInformation.currentRow);
		}
	}

	const CyclicCoordinateDescent& ccd;
	const ModelData& data;
	string delimitor;
	ofstream out;

private:
	bool displayRow(int row) {
		if (!useWeightAsMask) {
			return true;
		}
		return useComplementWeight == (ccd.getWeight(row) == 0.0);
	}

	bool useWeightAsMask;
	bool useComplementWeight;
};

class PredictionOutputWriter : public BaseOutputWriter<PredictionOutputWriter> {
public:
	PredictionOutputWriter(const CyclicCoordinateDescent& ccd, const ModelData& data) :
		BaseOutputWriter<PredictionOutputWriter>(ccd, data) {
		// Do nothing
	}
	virtual ~PredictionOutputWriter() {
		// Do nothing
	}

	void preprocessAllRows() {
		predictions.resize(ccd.getPredictionSize());
		ccd.getPredictiveEstimates(&predictions[0], NULL);
	}

	void writeRow(ofstream& out, OutputHelper::RowInformation& rowInfo) {
		if (data.getHasRowLobels()) {
			out << data.getRowLabel(rowInfo.currentRow) << delimitor;
		}
		out << predictions[rowInfo.currentRow];
		out << endl; // TODO This causes a flush; may be faster to stream several lines before flush
	}

private:
	std::vector<real> predictions; // TODO Should be double
};

}

#endif /* OUTPUTWRITER_H_ */
