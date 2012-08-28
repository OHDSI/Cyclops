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

class ModelData : public CompressedDataMatrix {
public:
	ModelData();
	virtual ~ModelData();

	int* getPidVector();
	real* getYVector();
	int* getNEventVector();
	int* getOffsetVector();
	map<int, DrugIdType> getDrugNameMap();
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
	
	bool hasCovariate(DrugIdType covariate) const {
		return drugMap.count(covariate) != 0;
	}
	
	
	void addCovariate(DrugIdType covariate, FormatType type) {
		drugMap.insert(std::make_pair(covariate,getNumberOfColumns()));
		push_back(type);
	}
	
	void addLabel(DrugIdType covariate, std::string label) {
		add_label(drugMap[covariate], label);
	}
	
	void addDatum(DrugIdType covariate, int row, real value) {
		add_data(drugMap[covariate], row, value);		
	}

//#ifdef DATA_AOS
//	CompressedDataColumn& getColumn(DrugIdType covariate) {
//		return allColumns[drugMap[covariate]];
//	}
//#endif
	
	// TODO Improve encapsulation
	friend class SCCSInputReader;
	friend class CLRInputReader;
	friend class RTestInputReader;
	friend class CoxInputReader;
	friend class CCTestInputReader;

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
	map<DrugIdType, int> drugMap;
	map<int, DrugIdType> indexToDrugIdMap;
	string conditionId;
};

#endif /* MODELDATA_H_ */
