/*
 * CLRInputReader.cpp
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

#include "CLRInputReader.h"

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

CLRInputReader::CLRInputReader() : InputReader() {
	// Do nothing
}

CLRInputReader::~CLRInputReader() {
	// Do nothing
}

void CLRInputReader::readFile(const char* fileName) {
	ifstream in(fileName);
	if (!in) {
		cerr << "Unable to open " << fileName << endl;
		exit(-1);
	}

	string line;
#ifdef HAS_HEADER
	getline(in, line); // Read header
#endif

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

	// Set-up fixed columns of covariates
	int_vector* gender = new int_vector();
	push_back(gender, NULL, INDICATOR);

	real_vector* age = new real_vector();
	push_back(NULL, age, DENSE);

	real_vector* drugCount = new real_vector();
	push_back(NULL, drugCount, DENSE);

	real_vector* days = new real_vector();
	push_back(NULL, days, DENSE);

	vector<int_vector*> unorderColumns = vector<int_vector*>();
	vector<real_vector*> unorderData = vector<real_vector*>();

	int numCases = 0;
	int numDrugs = 0;
	string currentPid = MISSING_STRING;
	int numEvents = 0;
	string outcomeId = MISSING_STRING;
	DrugIdType noDrug = NO_DRUG;

	vector<string> strVector;

	string outerDelimiter(",");

	bool hasConditionId = true;

	int currentEntry = 0;
	while (getline(in, line) && (currentEntry < MAX_ENTRIES)) {	
		if (!line.empty()) {

			stringstream ss(line.c_str()); // Tokenize
			strVector.clear();
			split(strVector, line, outerDelimiter);
//			cerr << "entries = " << strVector.size() << endl;
//			exit(-1);

			// Parse condition entry
			if (hasConditionId) {
				string currentOutcomeId = strVector[1];
				if (outcomeId == MISSING_STRING) {
					outcomeId = currentOutcomeId;
				} else if (currentOutcomeId != outcomeId) {
					cerr << "More than one condition ID in input file" << endl;
					exit(-1);
				}
			}

//			cerr << "Line read" << endl;

			// Parse case ID (pid) entry
			string unmappedPid = strVector[0];
//			ss >> unmappedPid;
			if (unmappedPid != currentPid) { // New patient, ASSUMES these are sorted
				if (currentPid != MISSING_STRING) { // Skip first switch
					nevents.push_back(numEvents);
//					cerr << "Saved " << numEvents << " events in CC" << endl;
					numEvents = 0;
				}
				currentPid = unmappedPid;
				numCases++;
				cerr << "Added new case!" << endl;
			}
			pid.push_back(numCases - 1);

			// Parse outcome entry
			int thisEta;
//			ss >> thisEta;
			istringstream(strVector[2]) >> thisEta;
 			numEvents += thisEta;
			eta.push_back(thisEta);

			// Fix offs for CLR
//			int thisOffs;
//			ss >> thisOffs;
			offs.push_back(1);

			// Parse gender entry; F = 0; M = 1
			if (strVector[3] == "F") {
//				cerr << "Female " << currentEntry << endl;
			} else {
//				cerr << "Male " << currentEntry << endl;
				gender->push_back(currentEntry); // Keep track of males
			}

			// Parse age entry
			int thisAge;
			istringstream(strVector[4]) >> thisAge;
			age->push_back(thisAge);

			// Parse drugCount entry
			int thisCount;
			istringstream(strVector[5]) >> thisCount;
			drugCount->push_back(thisCount);

			// Parse days entry
			int thisDays;
			istringstream(strVector[6]) >> thisDays;
			days->push_back(thisDays);

			// Parse variable-length entries
			DrugIdType drug;
			vector<DrugIdType> uniqueDrugsForEntry;

			vector<string> drugs;
			if (strVector[7] != "") {
				split(drugs, strVector[7], "+");
				for (int i = 0; i < drugs.size(); ++i) {
					int drug;
					vector<string> pair;
					split(pair, drugs[i], ":");
					istringstream(pair[0]) >> drug;
					int thisQuantity;
//					cerr  << drug << ":" << thisQuantity << " ";
					istringstream(pair[1]) >> thisQuantity;
					if (drugMap.count(drug) == 0) {
						drugMap.insert(make_pair(drug, numDrugs));
						unorderColumns.push_back(new int_vector());
						unorderData.push_back(new real_vector());
						numDrugs++;
					}
					if (!listContains(uniqueDrugsForEntry, drug)) {
						// Add to CSC storage
						unorderColumns[drugMap[drug]]->push_back(currentEntry);
						unorderData[drugMap[drug]]->push_back(thisQuantity);
					}
				}
//				cerr << endl;
			}

//			cerr << "#drugs = " << drugs.size();
//			if (drugs.size() > 0) {
//				cerr << " " << drugs[0];
//			}
//			cerr << endl;

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

			currentEntry++;
		}
	}

	nevents.push_back(numEvents); // Save last patient
	int index = columns.size();

//	columns = vector<int_vector>(unorderColumns.size());
	for (int i = 0; i < unorderColumns.size(); ++i) {
		columns.push_back(NULL);
		data.push_back(NULL);
		formatType.push_back(SPARSE);
	}
//	columns.resize(unorderColumns.size());
//	formatType.resize(unorderColumns.size(), INDICATOR);

	// Sort drugs numerically
	for (map<DrugIdType,int>::iterator ii = drugMap.begin(); ii != drugMap.end(); ii++) {
		if (columns[index]) {
			delete columns[index];
		}
		if (data[index]) {
			delete data[index];
		}
	   	columns[index] = unorderColumns[(*ii).second];
	   	data[index] = unorderData[(*ii).second];
	   	drugMap[(*ii).first] = index;
	   	indexToDrugIdMap.insert(make_pair(index, (*ii).first));
	   	index++;
	}

	cout << "Read " << currentEntry << " data lines from " << fileName << endl;
//	cout << "Number of patients: " << numCases << endl;
//	cout << "Number of drugs: " << numDrugs << endl;

//	cout << "Number"

#if 1
	erase(0);
	erase(0);
	printColumn(0);
	printColumn(1);
	printColumn(2);
	printColumn(3);
	printColumn(4);
#endif

	nPatients = numCases;
	nCols = columns.size();
	nRows = currentEntry;
	conditionId = outcomeId;

#if 1
	nCols = 1;
#endif
//	exit(-1);

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

}



