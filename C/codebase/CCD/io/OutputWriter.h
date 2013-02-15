/*
 * OutputWriter.h
 *
 *  Created on: Feb 15, 2013
 *      Author: msuchard
 */

#ifndef OUTPUTWRITER_H_
#define OUTPUTWRITER_H_

#include "../CyclicCoordinateDescent.h"
#include "../ModelData.h"

namespace bsccs {

// Define interface

class OutputWriter {
public:
	OutputWriter() { }
	virtual ~OutputWriter() { }

	virtual void writeFile(const char* fileName) = 0;
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
		// Do nothing
	}
	virtual ~BaseOutputWriter() {
		// Do nothing
	}

	virtual void writeFile(const char* fileName) {
		ofstream out;
		out.open(fileName, ios::out);
		writeFile(out);

	}

protected:
	void writeFile(ofstream& out) {
		out << "Hello world!" << endl;

		static_cast<DerivedFormat*>(this)->preprocessAllRows();

		OutputHelper::RowInformation rowInformation(0);
		while(rowInformation.currentRow < data.getNumberOfRows()) {
			static_cast<DerivedFormat*>(this)->writeRow(out, rowInformation);
			++(rowInformation.currentRow);
		}
	}

	const CyclicCoordinateDescent& ccd; // TODO Should be const
	const ModelData& data;
	string delimitor;
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
