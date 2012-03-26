/*
 * RTestInputReader.cpp
 *
 *  Created on: Mar 25, 2012
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

#include "RTestInputReader.h"

#define MAX_ENTRIES		1000000000

#define FORMAT_MATCH_1	"CONDITION_CONCEPT_ID"
#define MATCH_LENGTH_1	20
#define FORMAT_MATCH_2	"PID"
#define MATCH_LENGTH_2	3
#define MISSING_STRING	"NA"
#define MISSING_LENGTH	-1
#define DELIMITER		","

using namespace std;

RTestInputReader::RTestInputReader() : InputReader() { }

RTestInputReader::~RTestInputReader() { }

/**
 * Reads in a dense CSV data file with format:
 * Stratum,Outcome,X1 ...
 *
 * Assumes that file is sorted by 'Stratum'
 */
void RTestInputReader::readFile(const char* fileName) {
	ifstream in(fileName);
	if (!in) {
		cerr << "Unable to open " << fileName << endl;
		exit(-1);
	}

	string line;
	getline(in, line); // Read header and ignore

	int numCases = 0;
	int numCovariates = MISSING_LENGTH;
	string currentStratum = MISSING_STRING;
	int numEvents = 0;

	vector<string> strVector;
	string outerDelimiter(DELIMITER);

	int currentRow = 0;
	while (getline(in, line) && (currentRow < MAX_ENTRIES)) {
		if (!line.empty()) {

			strVector.clear();
			split(strVector, line, outerDelimiter);

			// Make columns
			if (numCovariates == MISSING_LENGTH) {
				numCovariates = strVector.size() - 2;
				for (int i = 0; i < numCovariates; ++i) {
					real_vector* thisColumn = new real_vector();
					push_back(NULL, thisColumn, DENSE);
				}
			} else if (numCovariates != strVector.size() - 2) {
				cerr << "All rows must be the same length" << endl;
				exit(-1);
			}

			// Parse stratum (pid)
			string unmappedStratum = strVector[0];
			if (unmappedStratum != currentStratum) { // New stratum, ASSUMES these are sorted
				if (currentStratum != MISSING_STRING) { // Skip first switch
					nevents.push_back(1);
					numEvents = 0;
				}
				currentStratum = unmappedStratum;
				numCases++;
			}
			pid.push_back(numCases - 1);

			// Parse outcome entry
			int thisEta;
			istringstream(strVector[1]) >> thisEta;
 			numEvents += thisEta;
			eta.push_back(thisEta);

			// Fix offs for CLR
			offs.push_back(1);

			// Parse covariates
			for (int i = 0; i < numCovariates; ++i) {
				real value;
				istringstream(strVector[2 + i]) >> value;
				data[i]->push_back(value);
			}

			currentRow++;
		}
	}
	nevents.push_back(1); // Save last patient

	int index = columns.size();

	cout << "RTestInputReader" << endl;
	cout << "Read " << currentRow << " data lines from " << fileName << endl;
	cout << "Number of stratum: " << numCases << endl;
	cout << "Number of covariates: " << numCovariates << endl;

	nPatients = numCases;
	nCols = columns.size();
	nRows = currentRow;
	conditionId = "0";

	for (int i = 0; i < nCols; ++i) {
		printColumn(i);
	}

#if 1
	cerr << "PIDs ";
	printVector(&(pid[0]), static_cast<int>(pid.size()));

	cerr << "Ys ";
	printVector(eta.data(), eta.size());

	cerr << "nEvents ";
	printVector(nevents.data(), nevents.size());
#endif
}
