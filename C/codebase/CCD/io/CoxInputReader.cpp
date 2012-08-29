/*
 * CoxInputReader.cpp
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

#include "CoxInputReader.h"

#define MAX_ENTRIES		1000000000

#define FORMAT_MATCH_1	"CONDITION_CONCEPT_ID"
#define MATCH_LENGTH_1	20
#define FORMAT_MATCH_2	"PID"
#define MATCH_LENGTH_2	3
#define MISSING_STRING	"NA"
#define MISSING_LENGTH	-1
#define DELIMITER		","

using namespace std;

CoxInputReader::CoxInputReader() : InputReader() { }

CoxInputReader::~CoxInputReader() { }

/**
 * Reads in a dense CSV data file with format:
 * Stratum,Outcome,X1 ...
 *
 * Assumes that file is sorted by 'Stratum'
 */
void CoxInputReader::readFile(const char* fileName) {

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
					modelData->push_back(DENSE);
				}
			} else if (numCovariates != strVector.size() - 2) {
				cerr << "All rows must be the same length" << endl;
				exit(-1);
			}

			numCases++; // Each row is separate case
			modelData->pid.push_back(numCases - 1);

			// Parse outcome entry
			real thisZ = static_cast<real>(atof(strVector[0].c_str()));
 			real thisY = static_cast<real>(atof(strVector[1].c_str()));
 			modelData->y.push_back(thisY);
 			modelData->z.push_back(thisZ);

			// Fix offs for CLR
 			modelData->offs.push_back(1);

			// Parse covariates
			for (int i = 0; i < numCovariates; ++i) {
				real value = static_cast<real>(atof(strVector[2 + i].c_str()));
				modelData->getColumn(i).add_data(currentRow, value);
			}

			currentRow++;
		}
	}
	modelData->nevents.push_back(1); // Save last patient

#ifndef MY_RCPP_FLAG
	cout << "CoxInputReader" << endl;
	cout << "Read " << currentRow << " data lines from " << fileName << endl;
	cout << "Number of cases: " << numCases << endl;
	cout << "Number of covariates: " << numCovariates << endl;
#endif

	// TODO Code duplication below
	modelData->nPatients = numCases;
	modelData->nRows = currentRow;
	modelData->conditionId = "0";
}
