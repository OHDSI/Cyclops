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

//InputReader::InputReader(const char* fileName) {
//	readFile(fileName); // TODO Throw error
//}

//	ifstream in(fileName);
//	if (!in) {
//		cerr << "Unable to open " << fileName << endl;
//		exit(-1);
//	}
//
//	string line;
//	getline(in, line); // Read header
//
//	if ((line.compare(0, MATCH_LENGTH_1, FORMAT_MATCH_1) != 0) &&
//			(line.compare(0, MATCH_LENGTH_2, FORMAT_MATCH_2) != 0)) {
//		cerr << "Unrecognized file type" << endl;
//		exit(-1);
//	}
//
//	bool hasConditionId = true;
//	if (line.compare(0, MATCH_LENGTH_2, FORMAT_MATCH_2) == 0) {
//		hasConditionId = false; // Original data format style
//	}
//
//	vector<int_vector*> unorderColumns = vector<int_vector*>();
//
//	int numPatients = 0;
//	int numDrugs = 0;
//	string currentPid = MISSING_STRING;
//	int numEvents = 0;
//	string outcomeId = MISSING_STRING;
//	DrugIdType noDrug = NO_DRUG;
//
//	int currentEntry = 0;
//	while (getline(in, line) && (currentEntry < MAX_ENTRIES)) {
//		if (!line.empty()) {
//
//			stringstream ss(line.c_str()); // Tokenize
//
//			// Parse first entry
//			if (hasConditionId) {
//				string currentOutcomeId;
//				ss >> currentOutcomeId;
//				if (outcomeId == MISSING_STRING) {
//					outcomeId = currentOutcomeId;
//				} else if (currentOutcomeId != outcomeId) {
//					cerr << "More than one condition ID in input file" << endl;
//					exit(-1);
//				}
//			}
//
//			// Parse second entry
//			string unmappedPid;
//			ss >> unmappedPid;
//			if (unmappedPid != currentPid) { // New patient, ASSUMES these are sorted
//				if (currentPid != MISSING_STRING) { // Skip first switch
//					nevents.push_back(numEvents);
//					numEvents = 0;
//				}
//				currentPid = unmappedPid;
//				numPatients++;
//			}
//			pid.push_back(numPatients - 1);
//
//			// Parse third entry
//			int thisEta;
//			ss >> thisEta;
//			numEvents += thisEta;
//			eta.push_back(thisEta);
//
//			// Parse fourth entry
//			int thisOffs;
//			ss >> thisOffs;
//			offs.push_back(thisOffs);
//
//			// Parse remaining (variable-length) entries
//			DrugIdType drug;
//			vector<DrugIdType> uniqueDrugsForEntry;
//			while (ss >> drug) {
//				if (drug == noDrug) { // No drug
//					// Do nothing
//				} else {
//					if (drugMap.count(drug) == 0) {
//						drugMap.insert(make_pair(drug,numDrugs));
//						unorderColumns.push_back(new int_vector());
//						numDrugs++;
//					}
//					if (!listContains(uniqueDrugsForEntry, drug)) {
//						// Add to CSC storage
//						unorderColumns[drugMap[drug]]->push_back(currentEntry);
//						uniqueDrugsForEntry.push_back(drug);
//					}
//				}
//			}
//
//			currentEntry++;
//		}
//	}
//
//	nevents.push_back(numEvents); // Save last patient
//
////	columns = vector<int_vector>(unorderColumns.size());
//	columns.resize(unorderColumns.size());
//	formatType.resize(unorderColumns.size(), INDICATOR);
//
//	// Sort drugs numerically
//	int index = 0;
//	for (map<DrugIdType,int>::iterator ii = drugMap.begin(); ii != drugMap.end(); ii++) {
//		if (columns[index]) {
//			delete columns[index];
//		}
//	   	columns[index] = unorderColumns[(*ii).second];
//	   	drugMap[(*ii).first] = index;
//	   	indexToDrugIdMap.insert(make_pair(index, (*ii).first));
//	   	index++;
//	}
//
//	cout << "Read " << currentEntry << " data lines from " << fileName << endl;
////	cout << "Number of patients: " << numPatients << endl;
////	cout << "Number of drugs: " << numDrugs << endl;
//
//	nPatients = numPatients;
//	nCols = columns.size();
//	nRows = currentEntry;
//	conditionId = outcomeId;
//
//#if 0
//	cout << "Converting first column to dense format" << endl;
//	convertColumnToDense(0);
//#endif
//
//#if 0
//	const int count = 20;
//	cout << "Converting some columns to dense format" << endl;
//	for (int j = 0; j < std::min(count, nCols); ++j) {
//		convertColumnToDense(j);
//	}
//	nCols = std::min(count, nCols);
//#endif
//
//#if 0
//	const int count = 20;
//	cout << "Converting some columns to sparse format" << endl;
//	for (int j = 0; j < std::min(count, nCols); ++j) {
//		convertColumnToSparse(j);
//	}
//	nCols = std::min(count, nCols);
//#endif
//
//#if 0
//	cout << "Converting all columns to dense format" << endl;
//	for (int j = 0; j < nCols; ++j) {
//		convertColumnToDense(j);
//	}
//#endif
//}

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


int* InputReader::makeDeepCopy(int *original, unsigned int length) {
	int *copy = (int *) malloc(length * sizeof(int));
	memcpy(copy, original, length * sizeof(int));
	return copy;
}


void
InputReader::split( vector<string> & theStringVector,  /* Altered/returned value */
       const  string  & theString,
       const  string  & theDelimiter)
{
//    UASSERT( theDelimiter.size(), >, 0); // My own ASSERT macro.

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

int* InputReader::getEtaVector() {
	//return &eta[0];
	return makeDeepCopy(&eta[0], eta.size());
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
