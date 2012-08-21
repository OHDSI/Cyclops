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
	
private:

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
