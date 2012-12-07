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

using std::map;
using std::string;
using std::vector;
using std::stringstream;

#include "CompressedDataMatrix.h"

//#define USE_DRUG_STRING

#ifdef USE_DRUG_STRING
	typedef string DrugIdType; // TODO Strings do not get sorted in numerical order
#else
	typedef int DrugIdType;
#endif

template <class T> void reindexVector(vector<T>& vec, vector<int> ind) {
	int n = (int) vec.size();
	vector<T> temp = vec;
	for(int i = 0; i < n; i++){
		vec[i] = temp[ind[i]];
	}
}

class ModelData : public CompressedDataMatrix {
public:
	ModelData();
	virtual ~ModelData();

	int* getPidVector();
	real* getYVector();
	void setYVector(vector<real> y_);
	int* getNEventVector();
	int* getOffsetVector();
//	map<int, DrugIdType> getDrugNameMap();
	int getNumberOfPatients();
	string getConditionId();
	std::vector<int>* getPidVectorSTL();

	const std::vector<real>& getZVectorRef() const {
		return z;
	}

	const std::vector<real>& getYVectorRef() const {
		return y;
	}

	const std::vector<int>& getPidVectorRef() const {
		return pid;
	}
	
	const std::vector<int>& getNEventsVectorRef() const {
		return nevents;
	}

	bool getHasOffsetCovariate() const {
		return hasOffsetCovariate;
	}

	void sortDataColumns(vector<int> sortedInds);
	
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
	// Disable copy-constructors
	ModelData(const ModelData&);

	template <class T>
	T* makeDeepCopy(T *original, unsigned int length) {
		T *copy = (T *) malloc(length * sizeof(T));
		memcpy(copy, original, length * sizeof(T));
		return copy;
	}

	int nPatients;
	vector<int> pid;
	vector<real> y;
	vector<real> z;
	vector<int> offs;
	vector<int> nevents;
	string conditionId;
	bool hasOffsetCovariate;
};

#endif /* MODELDATA_H_ */
