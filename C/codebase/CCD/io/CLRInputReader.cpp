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
#include "SparseIndexer.h"

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

namespace bsccs {

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

	const bool useGender = false;
	const bool useAge = false;
	const bool useDrugCount = true;
	const bool useDays = true;

	const int drugColumn = 9 - 1; // beforeIndex_DaysSinceUse
	const int maxDaysOnDrug = 0; // -1 = any
	const bool useDrugIndicator = true;

	SparseIndexer indexer(*modelData);

#define GENDER	-10
#define AGE		-9
#define DRUG	-8
#define DAYS	-7

	// Set-up fixed columns of covariates
	int_vector* gender = new int_vector();
	if (useGender) {
		indexer.addColumn(GENDER, INDICATOR);
	}

	if (useAge) {
		indexer.addColumn(AGE, DENSE);
	}

	if (useDrugCount) {
		indexer.addColumn(DRUG, DENSE);
	}

	if (useDays) {
		indexer.addColumn(DAYS, DENSE);
	}

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
					modelData->nevents.push_back(numEvents);
					numEvents = 0;
				}
				currentPid = unmappedPid;
				numCases++;
//				cerr << "Added new case!" << endl;
			}
			modelData->pid.push_back(numCases - 1);

			// Parse outcome entry
			int thisY;
			istringstream(strVector[2]) >> thisY;
 			numEvents += thisY;
 			modelData->y.push_back(thisY);

			// Fix offs for CLR
 			modelData->offs.push_back(1);

			// Parse gender entry; F = 0; M = 1
			if (strVector[3] == "F") {
//				cerr << "Female " << currentEntry << endl;
			} else {
//				cerr << "Male " << currentEntry << endl;
				indexer.getColumn(GENDER).add_data(currentEntry, 1.0);
			}

			// Parse age entry
			int thisAge;
			istringstream(strVector[4]) >> thisAge;
			indexer.getColumn(AGE).add_data(currentEntry, thisAge);

			// Parse drugCount entry
			int thisCount;
			istringstream(strVector[5]) >> thisCount;
			indexer.getColumn(DRUG).add_data(currentEntry, thisCount);

			// Parse days entry
			int thisDays;
			istringstream(strVector[6]) >> thisDays;
			indexer.getColumn(DAYS).add_data(currentEntry, thisDays);

			// Parse variable-length entries
			DrugIdType drug;

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

					if (maxDaysOnDrug == -1 || thisQuantity <= maxDaysOnDrug) {  // Only add drug:0 pairs
						if (!indexer.hasColumn(drug)) {
							// Add new column
							if (useDrugIndicator) {
								indexer.addColumn(drug, INDICATOR);
							} else {
								indexer.addColumn(drug, SPARSE);
							}
						}
						// Add to CSC storage
						real value = useDrugIndicator ? 1.0 : thisQuantity;
						indexer.getColumn(drug).add_data(currentEntry, value);
					}
				}
			}
			currentEntry++;
		}
	}

	modelData->nevents.push_back(numEvents); // Save last patient

	// Easy to sort columns now in AOS format
	modelData->sortColumns(CompressedDataColumn::sortNumerically);

	cout << "Read " << currentEntry << " data lines from " << fileName << endl;
	cout << "Number of patients: " << numCases << endl;
	cout << "Number of drugs: " << numDrugs << " out of " << modelData->getNumberOfColumns() << endl;


	// TODO Code duplication
	modelData->nPatients = numCases;
	modelData->nRows = currentEntry;
	modelData->conditionId = outcomeId;

}

} // namespace
