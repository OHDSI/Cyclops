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
#include <numeric>
#include <algorithm>

#include "SCCSInputReader.h"
#include "io/SparseIndexer.h"

#include "io/CmdLineProgressLogger.h"

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

SCCSInputReader::SCCSInputReader() : InputReader(
	bsccs::make_shared<loggers::CoutLogger>(),
	bsccs::make_shared<loggers::CerrErrorHandler>()) {
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

	vector<IntVector*> unorderColumns = vector<IntVector*>();

	SparseIndexer<double> indexer(modelData->getX());

	int numPatients = 0;
	int numDrugs = 0;
	string currentPid = MISSING_STRING;
	int numEvents = 0;
	string outcomeId = MISSING_STRING;
	IdType noDrug = NO_DRUG;

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
					//modelData->nevents.push_back(numEvents);
					push_back_nevents(*modelData, numEvents);
					
					numEvents = 0;
				}
				currentPid = unmappedPid;
				numPatients++;
			}
			//modelData->pid.push_back(numPatients - 1);
			push_back_pid(*modelData, numPatients - 1);

			// Parse third entry
			int thisY;
			ss >> thisY;
			numEvents += thisY;
			//modelData->y.push_back(thisY);
			push_back_y(*modelData, thisY);

			// Parse fourth entry
			int thisOffs;
			ss >> thisOffs;
			//modelData->offs.push_back(thisOffs);
			push_back_offs(*modelData, thisOffs);

			// Parse remaining (variable-length) entries
			IdType drug;		
			while (ss >> drug) {
				if (drug == noDrug) { // No drug
					// Do nothing
				} else {
					if (!indexer.hasColumn(drug)) {
						// Add new column
						indexer.addColumn(drug, INDICATOR);
					}		
					// Add to CSC storage
					bool valid = indexer.getColumn(drug).add_data(currentEntry, 1.0);
					if (!valid) {
//						std::cerr << "Repeated entry!" << std::endl;
//						exit(-1);
					}
				}
			}
			currentEntry++;
		}
	}

	//modelData->nevents.push_back(numEvents); // Save last patient
	push_back_nevents(*modelData, numEvents);

	// Easy to sort columns now in AOS format
	//Columns(CompressedDataColumn::sortNumerically);
	
#ifndef MY_RCPP_FLAG
	cout << "Read " << currentEntry << " data lines from " << fileName << endl;
#endif
//	Rprintf("Read %d data lines from %s\n", currentEntry, fileName);
//	Rprintf("Number of drugs: %d\n", numDrugs);
//	cout << "Number of patients: " << numPatients << endl;
//	cout << "Number of drugs: " << modelData->getNumberOfColumns() << endl;

	//modelData->nPatients = numPatients;
	setNumberPatients(*modelData, numPatients);	
	//modelData->nRows = currentEntry;
	setNumberRows(*modelData, currentEntry);	
	//modelData->conditionId = outcomeId;
	setConditionId(*modelData, outcomeId);

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

} // namespace
