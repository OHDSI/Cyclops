/*
 * InputReader.h
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#ifndef INPUTREADER_H_
#define INPUTREADER_H_

#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <map>

using std::map;
using std::string;
using std::vector;
using std::stringstream;

#ifdef MY_RCPP_FLAG
	#include <R.h>
//// For OSX 10.6, R is built with 4.2.1 which has a bug in stringstream
//stringstream& operator>> (stringstream &in, int &out) {
//	string entry;
//	in >> entry;
//	out = atoi(entry.c_str());
//	return in;
//}
#endif

#include "ModelData.h"

////#define USE_DRUG_STRING
//
//#ifdef USE_DRUG_STRING
//	typedef string DrugIdType; // TODO Strings do not get sorted in numerical order
//#else
//	typedef int DrugIdType;
//#endif

class InputReader {
public:
	InputReader();
	virtual ~InputReader();

//	int* getPidVector();
//	real* getYVector();
//	int* getNEventVector();
//	int* getOffsetVector();
//	map<int, DrugIdType> getDrugNameMap();
//	int getNumberOfPatients();
//	string getConditionId();
//	std::vector<int>* getPidVectorSTL();
//
//	const std::vector<real>& getZVectorRef() const {
//		return z;
//	}
//
//	const std::vector<real>& getYVectorRef() const {
//		return y;
//	}
//
//	const std::vector<int>& getPidVectorRef() const {
//		return pid;
//	}

	virtual void readFile(const char* fileName) = 0;

	ModelData* getModelData() {
		deleteModelData = false;
		return modelData;
	}

protected:

//	template <class T>
//	T* makeDeepCopy(T *original, unsigned int length) {
//		T *copy = (T *) malloc(length * sizeof(T));
//		memcpy(copy, original, length * sizeof(T));
//		return copy;
//	}

	bool listContains(const vector<DrugIdType>& list, DrugIdType value);

	void split( vector<string> & theStringVector,
	       const  string  & theString,
	       const  string  & theDelimiter);

//	int nPatients;
//	vector<int> pid;
//	vector<real> y;
//	vector<real> z;
//	vector<int> offs;
//	vector<int> nevents;
//	map<DrugIdType, int> drugMap;
//	map<int, DrugIdType> indexToDrugIdMap;
//	string conditionId;

	ModelData* modelData;
	bool deleteModelData;
};

#endif /* INPUTREADER_H_ */
