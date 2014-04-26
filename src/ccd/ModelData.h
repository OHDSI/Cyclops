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
	ModelData();

//	pid.begin(), pid.end(),
//	y.begin(), y.end(),
//	z.begin(), z.end(),
//	offs.begin(), offs.end(),
//	xip.begin(), xii.end()

	template <typename IntegerVector, typename RealVector>
	ModelData(
			const IntegerVector& _pid,
			const RealVector& _y,
			const RealVector& _z,
			const RealVector& _offs
			) :
		hasOffsetCovariate(false), hasInterceptCovariate(false), nPatients(0)
		, pid(_pid.begin(), _pid.end()) // copy
		, y(_y.begin(), _y.end()) // copy
		, z(_z.begin(), _z.end()) // copy
		, offs(_offs.begin(), _offs.end()) // copy
		{

	}

	virtual ~ModelData();

	int* getPidVector();
	real* getYVector();
	void setYVector(std::vector<real> y_);
	int* getNEventVector();
	real* getOffsetVector();
//	map<int, DrugIdType> getDrugNameMap();
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

	bool getHasRowLobels() const {
		return (labels.size() == getNumberOfRows());
	}

	void sortDataColumns(std::vector<int> sortedInds);
	
	double getSquaredNorm() const;

	double getNormalBasedDefaultVar() const;

	int getNumberOfVariableColumns() const;

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

private:
	// Disable copy-constructors and copy-assignment
	ModelData(const ModelData&);
	ModelData& operator = (const ModelData&);

	int nPatients; // TODO Where are these used?
	std::vector<int> pid;
	std::vector<real> y; // TODO How to load these directly from Rcpp::NumericVector
	std::vector<real> z;
	std::vector<real> offs;
	std::vector<int> nevents; // TODO Where are these used?
	std::string conditionId;
	bool hasOffsetCovariate;
	bool hasInterceptCovariate;
	std::vector<std::string> labels;
	static const std::string missing;
};

} // namespace

#endif /* MODELDATA_H_ */
