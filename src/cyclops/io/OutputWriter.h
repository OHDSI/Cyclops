/*
 * OutputWriter.h
 *
 *  Created on: Feb 15, 2013
 *      Author: msuchard
 */

#ifndef OUTPUTWRITER_H_
#define OUTPUTWRITER_H_

#include <iterator>

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

class OFStream : public std::ofstream {
public:	

	OFStream(std::string _delimitor) : delimitor(_delimitor) { }

	template <typename T>
	OFStream& addText(const T& t) {
		*this << t;
		return *this;
	}
	
	template <typename T>
	OFStream& addText(const std::vector<T>& v) {
	    std::copy(v.begin(), v.end(),
	        std::ostream_iterator<T>(*this, " "));
	    return *this;
	}	
		
	OFStream& addDelimitor() { return addText(delimitor); }
	
	OFStream& addEndl() {
		*this << std::endl;
		return *this;
	}
	
	template <typename T> 
	OFStream& addHeader(const T& t) { return addText(t); }
		
	template <typename T>
	OFStream& addMetaKey(const T& t) { return addText(t).addDelimitor(); }
	
	template <typename T>
	OFStream& addMetaValue(const T& t) { return addText(t).addEndl(); }
	
	template <typename T>
	OFStream& addValue(const T& t) { return addText(t); }
	
	OFStream& endTable(const char* t) { return *this; }
	
	bool includeLabels() { return true; }
	
private:
	const std::string delimitor;
};

class CoutStream {
public:	

	CoutStream(std::string _delimitor) : delimitor(_delimitor) { }

	template <typename T>
	CoutStream& addText(const T& t) {
		std::cout << t;
		return *this;
	}
	
	template <typename T>
	CoutStream& addText(const std::vector<T>& v) {
	    std::copy(v.begin(), v.end(),
	        std::ostream_iterator<T>(std::cout, " "));
	    return *this;
	}
		
	CoutStream& addDelimitor() { return addText(delimitor); }
	
	CoutStream& addEndl() {
		std::cout << std::endl;
		return *this;
	}
	
	template <typename T> 
	CoutStream& addHeader(const T& t) { return addText(t); }
		
	template <typename T>
	CoutStream& addMetaKey(const T& t) { return addText(t).addDelimitor(); }
	
	template <typename T>
	CoutStream& addMetaValue(const T& t) { return addText(t).addEndl(); }
	
	template <typename T>
	CoutStream& addValue(const T& t) { return addText(t); }
	
	CoutStream& endTable(const char* t) { return *this; }
	
private:
	const std::string delimitor;
};

}

// Define base class

template <typename DerivedFormat, typename Missing = OutputHelper::NoMissingPolicy>
class BaseOutputWriter : public OutputWriter, Missing {
public:
	BaseOutputWriter(CyclicCoordinateDescent& _ccd, const ModelData& _data) :
		OutputWriter(), ccd(_ccd), data(_data), delimitor(","), endl("\n") {
		// Do nothing
	}
	virtual ~BaseOutputWriter() {
		// Do nothing
	}

	virtual void writeFile(const char* fileName) {
		OutputHelper::OFStream out(delimitor);
		out.open(fileName, std::ios::out);
		writeFile(out);
	}
	
	template <typename Stream>
	void writeStream(Stream& stream) {
	    writeFile(stream);
	}

protected:

	virtual void preprocessAllRows() {
		// Default: do nothing
	}

	virtual int getNumberOfRows() = 0; // pure virtual

//	virtual void writeHeader(ofstream& out) {
//		// Default: do nothing
//	}

	template <typename Stream>
	void writeFile(Stream& out) {
//		static_cast<DerivedFormat*>(this)->
				preprocessAllRows();

		static_cast<DerivedFormat*>(this)->
				writeHeader(out);
				
		static_cast<DerivedFormat*>(this)->
				writeMetaData(out);

		OutputHelper::RowInformation rowInformation(0);
		int numRows = //static_cast<DerivedFormat*>(this)->
				getNumberOfRows();

		while(rowInformation.currentRow < numRows) {
			static_cast<DerivedFormat*>(this)->writeRow(out, rowInformation);
			++(rowInformation.currentRow);
		}
		
		static_cast<DerivedFormat*>(this)->
			writeEndTable(out);		
	}
	
	template <typename Stream>
	void writeMetaData(Stream& out) {
		// Do nothing
	}	
	
	template <typename Stream>
	void writeEndTable(Stream& out) {
		// Do nothing
	}		

	CyclicCoordinateDescent& ccd;
	const ModelData& data;
	string delimitor;
	string endl;
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
	
	template <typename Stream>
	void writeEndTable(Stream& out) {
		out.endTable("diagonstics");
	}	

	template <typename Stream>
	void writeHeader(Stream& out) {
		out.addHeader("key").addDelimitor().addHeader("value").addEndl();
	}
	
	template <typename Stream>
	void writeMetaData(Stream& out) {
		// Do work
		std::vector<double> hyperParameter = ccd.getHyperprior();
		double logLikelihood = ccd.getLogLikelihood();
		double logPrior = ccd.getLogPrior();
		UpdateReturnFlags returnFlag = ccd.getUpdateReturnFlag();
		int iterations = ccd.getIterationCount();
		string priorInfo = ccd.getPriorInfo();
		int covariateCount = ccd.getBetaSize();

		if (covariateCount == 0) {
			returnFlag = MISSING_COVARIATES;
		}	
	
		out.addMetaKey("log_likelihood").addMetaValue(logLikelihood);		
		out.addMetaKey("log_prior").addMetaValue(logPrior);
		out.addMetaKey("return_flag").addMetaValue(returnFlagString(returnFlag));
		out.addMetaKey("iterations").addMetaValue(iterations);
		out.addMetaKey("prior_info").addMetaValue(priorInfo);
		out.addMetaKey("variance").addMetaValue(hyperParameter);
		out.addMetaKey("covariate_count").addMetaValue(covariateCount);

		for (ExtraInformationVector::const_iterator it = extraInfoVector.begin();
			it != extraInfoVector.end(); ++it) {
			out.addMetaKey(it->first).addMetaValue(it->second);
			//out << (*it).first << delimitor << (*it).second << endl;
		}
	}

	template <typename Stream>
	void writeRow(Stream& out, OutputHelper::RowInformation& rowInfo) {
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
	
	template <typename Stream>
	void writeEndTable(Stream& out) {
		out.endTable("prediction");
	}		

	int getNumberOfRows() { return data.getNumberOfRows(); }

	void preprocessAllRows() {
		predictions.resize(ccd.getPredictionSize());
		ccd.getPredictiveEstimates(&predictions[0], NULL);
	}

	template <typename Stream>
	void writeHeader(Stream& out) {
		if (data.getHasRowLabels() && out.includeLabels()) {
			out.addHeader("row_label").addDelimitor();
		}
  	out.addHeader("prediction").addEndl();		
	}
	
	template <typename Stream>
	void writeRow(Stream& out, OutputHelper::RowInformation& rowInfo) {
		if (data.getHasRowLabels() && out.includeLabels()) {
			out.addValue(data.getRowLabel(rowInfo.currentRow)).addDelimitor();
		}
  	out.addValue(predictions[rowInfo.currentRow]).addEndl();
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
	
	template <typename Stream>
	void writeEndTable(Stream& out) {
		out.endTable("estimation");
	}		
	
	void addBoundInformation(int column, ProfileInformation information) {
		information.defined = true;
		informationMap.insert(std::pair<int, ProfileInformation>(column, information));
	}	

	void addBoundInformation(const ProfileInformationMap& map) {
		informationMap.insert(map.begin(), map.end());
	}

  template <typename Stream>
	void writeHeader(Stream& out) {
		out.addHeader("column_label").addDelimitor().addHeader("estimate");
		if (withProfileBounds) {
            out.addDelimitor().addHeader("lower").addDelimitor().addHeader("upper");
		}
		out.addEndl();		
	}

	template <typename Stream>
	void writeRow(Stream& out, OutputHelper::RowInformation& rowInfo) {	
	    if (rowInfo.currentRow > 0 || !data.getHasOffsetCovariate()) {
            out.addValue(data.getColumn(rowInfo.currentRow).getNumericalLabel()).addDelimitor();
            out.addValue(ccd.getBeta(rowInfo.currentRow));
            if (withProfileBounds && informationList[rowInfo.currentRow].defined) {
                // TODO Delegate to friend of ProfileInformation
                out.addDelimitor();
                out.addValue(informationList[rowInfo.currentRow].lower95Bound);
                out.addDelimitor();
                out.addValue(informationList[rowInfo.currentRow].upper95Bound);
            }
            out.addEndl();
		}
	}

private:
	bool withProfileBounds;
	ProfileInformationMap informationMap;
	ProfileInformationList informationList;
};

}

#endif /* OUTPUTWRITER_H_ */
