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

#include <vector>
#include <map>

using namespace std;

#include "CompressedIndicatorMatrix.h"

//#define USE_DRUG_STRING

#ifdef USE_DRUG_STRING
	typedef string DrugIdType; // TODO String do not get sorted in numerical order
#else
	typedef int DrugIdType;
#endif

class InputReader: public CompressedIndicatorMatrix {
public:
	InputReader();

	InputReader(const char* fileName);
//	InputReader(const ifstream& in);

	virtual ~InputReader();

	int* getPidVector();
	int* getEtaVector();
	int* getNEventVector();
	int* getOffsetVector();
	map<int, DrugIdType> getDrugNameMap();
	int getNumberOfPatients();
	string getConditionId();

private:
	
	int* makeDeepCopy(int *original, unsigned int length);

	bool listContains(const vector<DrugIdType>& list, DrugIdType value);

	int nPatients;
	vector<int> pid;
	vector<int> eta;
	vector<int> offs;
	vector<int> nevents;
	map<DrugIdType, int> drugMap;
	map<int, DrugIdType> indexToDrugIdMap;
	string conditionId;
};

#endif /* INPUTREADER_H_ */
