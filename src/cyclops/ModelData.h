/*
 * ModelData.h
 *
 *  Created on: August, 2012
 *      Author: msuchard
 */

#ifndef MODELDATA_H_
#define MODELDATA_H_

#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <map>

// using std::map;
// using std::string;
// using std::vector;
// using std::stringstream;

#include "CompressedDataMatrix.h"
#include "io/ProgressLogger.h"
#include "io/SparseIndexer.h"

//#define USE_DRUG_STRING

namespace bsccs {

template <class T> void reindexVector(std::vector<T>& vec, std::vector<int> ind) {
	int n = (int) vec.size();
	std::vector<T> temp = vec;
	for(int i = 0; i < n; i++){
		vec[i] = temp[ind[i]];
	}
}

class ModelData : public CompressedDataMatrix {
public:
//	ModelData();
	
	ModelData(
    	ModelType modelType, 
        loggers::ProgressLoggerPtr log,
        loggers::ErrorHandlerPtr error
    );

//	pid.begin(), pid.end(),
//	y.begin(), y.end(),
//	z.begin(), z.end(),
//	offs.begin(), offs.end(),
//	xip.begin(), xii.end()

	template <typename IntegerVector, typename RealVector>
	ModelData(
	        ModelType _modelType,
			const IntegerVector& _pid,
			const RealVector& _y,
			const RealVector& _z,
			const RealVector& _offs,
            loggers::ProgressLoggerPtr _log,
            loggers::ErrorHandlerPtr _error			
			) :
		modelType(_modelType), nPatients(0), nStrata(0), hasOffsetCovariate(false), hasInterceptCovariate(false), isFinalized(false)
		, pid(_pid.begin(), _pid.end()) // copy
		, y(_y.begin(), _y.end()) // copy
		, z(_z.begin(), _z.end()) // copy
		, offs(_offs.begin(), _offs.end()) // copy
		, sparseIndexer(*this)
		, log(_log), error(_error)
		{

	}

	virtual ~ModelData();
	
	size_t append(
        const std::vector<IdType>& oStratumId,
        const std::vector<IdType>& oRowId,
        const std::vector<double>& oY,
        const std::vector<double>& oTime,
        const std::vector<IdType>& cRowId,
        const std::vector<IdType>& cCovariateId,
        const std::vector<double>& cCovariateValue        
    );		

	int* getPidVector();
	real* getYVector();
	void setYVector(std::vector<real> y_);
	int* getNEventVector();
	real* getOffsetVector();
//	map<int, IdType> getDrugNameMap();
	int getNumberOfPatients();
	std::string getConditionId();
	std::vector<int>* getPidVectorSTL();

	const std::vector<real>& getZVectorRef() const {
		return z;
	}

	const std::vector<real>& getYVectorRef() const {
		return y;
	}

	const std::vector<int>& getPidVectorRef() const { // Not const because PIDs can get renumbered
		return pid;
	}
	
	std::vector<real>& getZVectorRef() {
		return z;
	}

	std::vector<real>& getYVectorRef() {
		return y;
	}

	std::vector<int>& getPidVectorRef() { // Not const because PIDs can get renumbered
		return pid;
	}

//	const std::vector<int>& getNEventsVectorRef() const {
//		return nevents;
//	}

	bool getHasOffsetCovariate() const {
		return hasOffsetCovariate;
	}

	void setHasOffsetCovariate(bool b) {
		hasOffsetCovariate = b;
	}

	bool getHasInterceptCovariate() const {
		return hasInterceptCovariate;
	}

	void setHasInterceptCovariate(bool b) {
		hasInterceptCovariate = b;
	}

	bool getHasRowLabels() const {
		return (labels.size() == getNumberOfRows());
	}
	
	bool getIsFinalized() const {
	    return isFinalized;
	}
	
	void setIsFinalized(bool b) {
	    isFinalized = b;
	}

	void sortDataColumns(std::vector<int> sortedInds);
	
	double getSquaredNorm() const;

	double getNormalBasedDefaultVar() const;

	int getNumberOfVariableColumns() const;
	
	size_t getNumberOfStrata() const;
	
    size_t getColumnIndex(const IdType covariate) const;	

	const std::string& getRowLabel(size_t i) const {
		if (i >= labels.size()) {
			return missing;
		} else {
			return labels[i];
		}
	}

	// TODO Improve encapsulation
	friend class SCCSInputReader;
	friend class CLRInputReader;
	friend class RTestInputReader;
	friend class CoxInputReader;
	friend class CCTestInputReader;
	friend class GenericSparseReader;

	template <class FormatType, class MissingPolicy> friend class BaseInputReader;
	template <class ImputationPolicy> friend class BBRInputReader;
	template <class ImputationPolicy> friend class CSVInputReader;
	
protected:
    ModelType modelType;
    
	mutable int nPatients;
	mutable size_t nStrata;
	
	bool hasOffsetCovariate;
	bool hasInterceptCovariate;
	bool isFinalized;
		
	std::vector<int> pid;
	std::vector<real> y; // TODO How to load these directly from Rcpp::NumericVector
	std::vector<real> z;
	std::vector<real> offs; // TODO Rename to 'time'
	std::vector<int> nevents; // TODO Where are these used?
	std::string conditionId;
	std::vector<std::string> labels; // TODO Change back to 'long'
	
private:
	// Disable copy-constructors and copy-assignment
	ModelData(const ModelData&);
	ModelData& operator = (const ModelData&);
			
	static const std::string missing;
		
    std::pair<IdType,int> lastStratumMap;
    
    SparseIndexer sparseIndexer;
	
	protected:
    loggers::ProgressLoggerPtr log;
    loggers::ErrorHandlerPtr error;
        
};

} // namespace

#endif /* MODELDATA_H_ */
