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
	BaseOutputWriter(CyclicCoordinateDescent& _ccd, const ModelData& _data) :
		OutputWriter(), ccd(_ccd), data(_data), delimitor(",") {
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

	virtual void preprocessAllRows() {
		// Default: do nothing
	}

	virtual int getNumberOfRows() = 0; // pure virtual

	virtual void writeHeader(ofstream& out) {
		// Default: do nothing
	}

	void writeFile(ofstream& out) {
//		static_cast<DerivedFormat*>(this)->
				preprocessAllRows();

//		static_cast<DerivedFormat*>(this)->
				writeHeader(out);

		OutputHelper::RowInformation rowInformation(0);
		int numRows = //static_cast<DerivedFormat*>(this)->
				getNumberOfRows();

		while(rowInformation.currentRow < numRows) {
			static_cast<DerivedFormat*>(this)->writeRow(out, rowInformation);
			++(rowInformation.currentRow);
		}
	}

	CyclicCoordinateDescent& ccd;
	const ModelData& data;
	string delimitor;
};

typedef std::pair<std::string,double> ExtraInformation;
typedef std::vector<ExtraInformation> ExtraInformationVector;

class DiagnosticsOutputWriter : public BaseOutputWriter<DiagnosticsOutputWriter> {
public:
	DiagnosticsOutputWriter(CyclicCoordinateDescent& ccd, const ModelData& data) :
		BaseOutputWriter<DiagnosticsOutputWriter>(ccd, data) {
		// Do nothing
	}
	virtual ~DiagnosticsOutputWriter() {
		// Do nothing
	}

	std::string returnFlagString(UpdateReturnFlags flag) {
		switch (flag) {
			case SUCCESS : return "SUCCESS";
			case MAX_ITERATIONS : return "MAX_ITERATIONS";
			case ILLCONDITIONED : return "ILLCONDITIONED";
			default : return "FAILED";
		}
	}

	void addExtraInformation(ExtraInformationVector info) {
		extraInfoVector.insert(extraInfoVector.end(), info.begin(), info.end());
	}

	void writeHeader(ofstream& out) {
		// Do work
		double hyperParameter = ccd.getHyperprior();
		double logLikelihood = ccd.getLogLikelihood();
		double logPrior = ccd.getLogPrior();
		UpdateReturnFlags returnFlag = ccd.getUpdateReturnFlag();
		int iterations = ccd.getIterationCount();
		string priorInfo = ccd.getPriorInfo();

		out << "key" << delimitor << "value" << endl;
		out << "log_likelihood" << delimitor << logLikelihood << endl;
		out << "log_prior" << delimitor << logPrior << endl;
		out << "return_flag" << delimitor << returnFlagString(returnFlag) << endl;
		out << "iterations" << delimitor << iterations << endl;
		out << "prior_info" << delimitor << priorInfo << endl;
		out << "variance" << delimitor << hyperParameter << endl;

		for (ExtraInformationVector::const_iterator it = extraInfoVector.begin();
			it != extraInfoVector.end(); ++it) {
			out << (*it).first << delimitor << (*it).second << endl;
		}
	}

	void writeRow(ofstream& out, OutputHelper::RowInformation& rowInfo) {
		// Do nothing;
	}

	int getNumberOfRows() { return 0; }

private:
	ExtraInformationVector extraInfoVector;
};

class PredictionOutputWriter : public BaseOutputWriter<PredictionOutputWriter> {
public:
	PredictionOutputWriter(CyclicCoordinateDescent& ccd, const ModelData& data) :
		BaseOutputWriter<PredictionOutputWriter>(ccd, data) {
		// Do nothing
	}
	virtual ~PredictionOutputWriter() {
		// Do nothing
	}

	int getNumberOfRows() { return data.getNumberOfRows(); }

	void preprocessAllRows() {
		predictions.resize(ccd.getPredictionSize());
		ccd.getPredictiveEstimates(&predictions[0], NULL);
	}

	void writeHeader(ofstream& out) {
		if (data.getHasRowLobels()) {
			out << "row_label" << delimitor;
		}
		out << "prediction" << endl;
		// Default: do nothing
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
