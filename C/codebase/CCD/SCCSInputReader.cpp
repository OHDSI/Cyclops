/*
 * SCCSInputReader.cpp
 *
 *  Created on: April, 2012
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

#include "SCCSInputReader.h"
namespace bsccs {
#ifdef MY_RCPP_FLAG
// For OSX 10.6, R is built with 4.2.1 which has a bug in stringstream
stringstream& operator>> (stringstream &in, int &out) {
	string entry;
	in >> entry;
	out = atoi(entry.c_str());
	return in;
}
#endif

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

SCCSInputReader::SCCSInputReader() : InputReader() {
	// Do nothing
}

SCCSInputReader::~SCCSInputReader() {
	// Do nothing
}

void SCCSInputReader::readFile(const char* fileName) {
	ifstream in(fileName);
	if (!in) {
		cerr << "Unable to open " << fileName << endl;
		exit(-1);
	}

	string line;
	getline(in, line); // Read header

	if ((line.compare(0, MATCH_LENGTH_1, FORMAT_MATCH_1) != 0) &&
			(line.compare(0, MATCH_LENGTH_2, FORMAT_MATCH_2) != 0)) {
		cerr << "Unrecognized file type" << endl;
		exit(-1);
	}

	bool hasConditionId = true;
	if (line.compare(0, MATCH_LENGTH_2, FORMAT_MATCH_2) == 0) {
		hasConditionId = false; // Original data format style
	}

	vector<int_vector*> unorderColumns = vector<int_vector*>();

	int numPatients = 0;
	int numDrugs = 0;
	string currentPid = MISSING_STRING;
	int numEvents = 0;
	string outcomeId = MISSING_STRING;
	DrugIdType noDrug = NO_DRUG;

	int currentEntry = 0;
	while (getline(in, line) && (currentEntry < MAX_ENTRIES)) {	
		if (!line.empty()) {

			stringstream ss(line.c_str()); // Tokenize

			// Parse first entry
			if (hasConditionId) {
				string currentOutcomeId;
				ss >> currentOutcomeId;
				if (outcomeId == MISSING_STRING) {
					outcomeId = currentOutcomeId;
				} else if (currentOutcomeId != outcomeId) {
					cerr << "More than one condition ID in input file" << endl;
					exit(-1);
				}
			}

			// Parse second entry
			string unmappedPid;
			ss >> unmappedPid;
			if (unmappedPid != currentPid) { // New patient, ASSUMES these are sorted
				if (currentPid != MISSING_STRING) { // Skip first switch
					nevents.push_back(numEvents);
					numEvents = 0;
				}
				currentPid = unmappedPid;
				numPatients++;
			}
			pid.push_back(numPatients - 1);

			// Parse third entry
			int thisEta;
			ss >> thisEta;
			numEvents += thisEta;
			eta.push_back(thisEta);

			// Parse fourth entry
			int thisOffs;
			ss >> thisOffs;
			offs.push_back(thisOffs);

			// Parse remaining (variable-length) entries
			DrugIdType drug;
			vector<DrugIdType> uniqueDrugsForEntry;
			while (ss >> drug) {
				if (drug == noDrug) { // No drug
					// Do nothing
				} else {
					if (drugMap.count(drug) == 0) {
						drugMap.insert(make_pair(drug,numDrugs));
						unorderColumns.push_back(new int_vector());
						numDrugs++;
					}
					if (!listContains(uniqueDrugsForEntry, drug)) {
						// Add to CSC storage
						unorderColumns[drugMap[drug]]->push_back(currentEntry);
						uniqueDrugsForEntry.push_back(drug);
					}
				}
			}

			currentEntry++;
		}
	}

	nevents.push_back(numEvents); // Save last patient

//	columns = vector<int_vector>(unorderColumns.size());
	columns.resize(unorderColumns.size());
	formatType.resize(unorderColumns.size(), INDICATOR);

	// Sort drugs numerically
	int index = 0;
	for (map<DrugIdType,int>::iterator ii = drugMap.begin(); ii != drugMap.end(); ii++) {
		if (columns[index]) {
			delete columns[index];
		}
	   	columns[index] = unorderColumns[(*ii).second];
	   	drugMap[(*ii).first] = index;
	   	indexToDrugIdMap.insert(make_pair(index, (*ii).first));
	   	index++;
	}

#ifndef MY_RCPP_FLAG
	cout << "Read " << currentEntry << " data lines from " << fileName << endl;
#endif
//	Rprintf("Read %d data lines from %s\n", currentEntry, fileName);
//	Rprintf("Number of drugs: %d\n", numDrugs);
//	cout << "Number of patients: " << numPatients << endl;
//	cout << "Number of drugs: " << numDrugs << endl;

	nPatients = numPatients;
	nCols = columns.size();
	nRows = currentEntry;
	conditionId = outcomeId;

#if 0
	cout << "Converting first column to dense format" << endl;
	convertColumnToDense(0);
#endif

#if 0
	const int count = 20;
	cout << "Converting some columns to dense format" << endl;
	for (int j = 0; j < std::min(count, nCols); ++j) {
		convertColumnToDense(j);
	}
	nCols = std::min(count, nCols);
#endif

#if 0
	const int count = 20;
	cout << "Converting some columns to sparse format" << endl;
	for (int j = 0; j < std::min(count, nCols); ++j) {
		convertColumnToSparse(j);
	}
	nCols = std::min(count, nCols);
#endif

#if 0
	cout << "Converting all columns to dense format" << endl;
	for (int j = 0; j < nCols; ++j) {
		convertColumnToDense(j);
	}
#endif

}
}
