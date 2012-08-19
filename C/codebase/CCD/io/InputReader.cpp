/*
 * InputReader.cpp
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "InputReader.h"

#define MAX_ENTRIES		1000000000

#define FORMAT_MATCH_1	"CONDITION_CONCEPT_ID"
#define MATCH_LENGTH_1	20
#define FORMAT_MATCH_2	"PID"
#define MATCH_LENGTH_2	3

#define MISSING_STRING	"NA"

#ifdef USE_DRUG_STRING
	#define NO_DRUG			"0"
#else
	#define	NO_DRUG			0
#endif

using namespace std;

InputReader::InputReader() {
	// Do nothing
}

bool InputReader::listContains(const vector<DrugIdType>& list, DrugIdType value) {
	return (find(list.begin(), list.end(), value)
				!=  list.end());
}

int InputReader::getNumberOfPatients() {
	return nPatients;
}

string InputReader::getConditionId() {
	return conditionId;
}

InputReader::~InputReader() {
	// Do nothing
}

void InputReader::split( vector<string> & theStringVector,  /* Altered/returned value */
       const  string  & theString,
       const  string  & theDelimiter) {
    size_t  start = 0, end = 0;

    while ( end != string::npos)
    {
        end = theString.find( theDelimiter, start);

        // If at end, use length=maxLength.  Else use length=end-start.
        theStringVector.push_back( theString.substr( start,
                       (end == string::npos) ? string::npos : end - start));

        // If at end, use start=maxSize.  Else use start=end+delimiter.
        start = (   ( end > (string::npos - theDelimiter.size()) )
                  ?  string::npos  :  end + theDelimiter.size());
    }
}

int* InputReader::getPidVector() {	
	//return &pid[0];
	return makeDeepCopy(&pid[0], pid.size());
}

std::vector<int>* InputReader::getPidVectorSTL() {
	return new std::vector<int>(pid);
}

real* InputReader::getYVector() {
	//return &eta[0];
	return makeDeepCopy(&y[0], y.size());
}

int* InputReader::getNEventVector() {
	//return &nevents[0];
	return makeDeepCopy(&nevents[0], nevents.size());
}

int* InputReader::getOffsetVector() {
	//return &offs[0];
	return makeDeepCopy(&offs[0], offs.size());
}

map<int, DrugIdType> InputReader::getDrugNameMap() {
//	return drugMap;
	return indexToDrugIdMap;
}
