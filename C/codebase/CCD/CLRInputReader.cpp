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
#define HAS_HEADER

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

namespace bsccs {

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

	const bool useGender = false;
	const bool useAge = false;
	const bool useDrugCount = true;
	const bool useDays = true;

	const int drugColumn = 9 - 1; // beforeIndex_DaysSinceUse
	const int maxDaysOnDrug = 0; // -1 = any
	const bool useDrugIndicator = true;

	// Set-up fixed columns of covariates
	int_vector* gender = new int_vector();
	if (useGender) {
		push_back(gender, NULL, INDICATOR);
	}

	real_vector* age = new real_vector();
	if (useAge) {
		push_back(NULL, age, DENSE);
	}

	real_vector* drugCount = new real_vector();
	if (useDrugCount) {
		push_back(NULL, drugCount, DENSE);
	}

	real_vector* days = new real_vector();
	if (useDays) {
		push_back(NULL, days, DENSE);
	}

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

			// Parse case ID (pid) entry
			string unmappedPid = strVector[0];
			if (unmappedPid != currentPid) { // New patient, ASSUMES these are sorted
				if (currentPid != MISSING_STRING) { // Skip first switch
					nevents.push_back(numEvents);
					numEvents = 0;
				}
				currentPid = unmappedPid;
				numCases++;
//				cerr << "Added new case!" << endl;
			}
			pid.push_back(numCases - 1);

			// Parse outcome entry
			int thisEta;
			istringstream(strVector[2]) >> thisEta;
 			numEvents += thisEta;
			eta.push_back(thisEta);

			// Fix offs for CLR
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
			if (strVector[drugColumn] != "") {
				split(drugs, strVector[drugColumn], "+");
				for (int i = 0; i < drugs.size(); ++i) {
					int drug;
					vector<string> pair;
					split(pair, drugs[i], ":");
					istringstream(pair[0]) >> drug;
					int thisQuantity;
					istringstream(pair[1]) >> thisQuantity;
//					cerr  << drug << ":" << thisQuantity << " ";
					if (maxDaysOnDrug == -1 || thisQuantity <= maxDaysOnDrug) {  // Only add drug:0 pairs
//						cerr << "add ";
						if (drugMap.count(drug) == 0) {
							drugMap.insert(make_pair(drug, numDrugs));
							unorderColumns.push_back(new int_vector());
							if (useDrugIndicator) {
								unorderData.push_back(NULL);
							} else {
								unorderData.push_back(new real_vector());
							}
							numDrugs++;
						}
						if (!listContains(uniqueDrugsForEntry, drug)) {
							// Add to CSC storage
							unorderColumns[drugMap[drug]]->push_back(currentEntry);
							if (!useDrugIndicator) {
								unorderData[drugMap[drug]]->push_back(thisQuantity);
							}
						}
					}
				}
//				cerr << endl;
			}
			currentEntry++;
		}
	}

	nevents.push_back(numEvents); // Save last patient
	int index = columns.size();

	for (int i = 0; i < unorderColumns.size(); ++i) {
		columns.push_back(NULL);
		data.push_back(NULL);
		formatType.push_back(useDrugIndicator ?  INDICATOR : SPARSE);
	}

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
	cout << "Number of patients: " << numCases << endl;
	cout << "Number of drugs: " << numDrugs << " out of " << columns.size() << endl;


	nPatients = numCases;
	nCols = columns.size();
	nRows = currentEntry;
	conditionId = outcomeId;

#if 0
//	erase(0);
//	erase(0);
	printColumn(0);
	cerr << "Sum = " << sumColumn(0) << endl;
//	printColumn(1);
//	printColumn(2);
//	printColumn(3);
//	printColumn(4);
	exit(-1);
#endif

	// TODO If !useGender, etc., then delete memory

}
}



