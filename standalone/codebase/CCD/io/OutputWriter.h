/*
 * OutputWriter.h
 *
 *  Created on: Feb 15, 2013
 *      Author: msuchard
 */

#ifndef OUTPUTWRITER_H_
#define OUTPUTWRITER_H_

#include "CyclicCoordinateDescent.h"
#include "ModelData.h"

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
		out.open(fileName, std::ios::out);
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

// typedef std::pair<std::string,double> ExtraInformation;
// typedef std::vector<ExtraInformation> ExtraInformationVector;

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
			case MISSING_COVARIATES : return "MISSING_COVARIATES";
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
		int covariateCount = ccd.getBetaSize();

		if (covariateCount == 0) {
			returnFlag = MISSING_COVARIATES;
		}

		out << "key" << delimitor << "value" << endl;
		out << "log_likelihood" << delimitor << logLikelihood << endl;
		out << "log_prior" << delimitor << logPrior << endl;
		out << "return_flag" << delimitor << returnFlagString(returnFlag) << endl;
		out << "iterations" << delimitor << iterations << endl;
		out << "prior_info" << delimitor << priorInfo << endl;
		out << "variance" << delimitor << hyperParameter << endl;
		out << "covariate_count" << delimitor << covariateCount << endl;

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

// struct ProfileInformation {
// 	bool defined;
// 	double lower95Bound;
// 	double upper95Bound;
// 
// 	ProfileInformation() : defined(false), lower95Bound(0.0), upper95Bound(0.0) { }
// 	ProfileInformation(double lower, double upper) : defined(true), lower95Bound(lower),
// 			upper95Bound(upper) { }
// };
// 
// typedef std::map<int, ProfileInformation> ProfileInformationMap;
// typedef std::vector<ProfileInformation> ProfileInformationList;

class EstimationOutputWriter : public BaseOutputWriter<EstimationOutputWriter> {
public:
	EstimationOutputWriter(CyclicCoordinateDescent& ccd, const ModelData& data) :
		BaseOutputWriter<EstimationOutputWriter>(ccd, data), withProfileBounds(false) {
		// Do nothing
	}
	virtual ~EstimationOutputWriter() {
		// Do nothing
	}

	int getNumberOfRows() { return ccd.getBetaSize(); }

	void preprocessAllRows() {
		// Set up profile information vector
		informationList.resize(ccd.getBetaSize());
		for (ProfileInformationMap::iterator it = informationMap.begin();
				it != informationMap.end(); ++it) {
			informationList[it->first] = it->second;
		}
		withProfileBounds = !informationMap.empty();
	}
	
	void addBoundInformation(int column, ProfileInformation information) {
		information.defined = true;
		informationMap.insert(std::pair<int, ProfileInformation>(column, information));
	}	

	void addBoundInformation(const ProfileInformationMap& map) {
		informationMap.insert(map.begin(), map.end());
	}

	void writeHeader(ofstream& out) {
		out << "column_label" << delimitor << "estimate";
		if (withProfileBounds) {
			out << delimitor << "lower" << delimitor << "upper";
		}
		out << endl;		
	}

	void writeRow(ofstream& out, OutputHelper::RowInformation& rowInfo) {
		out << data.getColumn(rowInfo.currentRow).getLabel();
		out << delimitor;
		out << ccd.getBeta(rowInfo.currentRow);
		if (withProfileBounds && informationList[rowInfo.currentRow].defined) {
			// TODO Delegate to friend of ProfileInformation
			out << delimitor;
			out << informationList[rowInfo.currentRow].lower95Bound;
			out << delimitor;
			out << informationList[rowInfo.currentRow].upper95Bound;
		}
		out << endl; // TODO This causes a flush; may be faster to stream several lines before flush
	}

private:
	bool withProfileBounds;
	ProfileInformationMap informationMap;
	ProfileInformationList informationList;
};


}

#endif /* OUTPUTWRITER_H_ */
