/*
 * CCTestInputReader.cpp
 *
 *  Created on: Mar 31, 2012
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
#include <numeric>

#include "CCTestInputReader.h"

#define MAX_ENTRIES		1000000000

#define FORMAT_MATCH_1	"CONDITION_CONCEPT_ID"
#define MATCH_LENGTH_1	20
#define FORMAT_MATCH_2	"PID"
#define MATCH_LENGTH_2	3
#define MISSING_STRING	"NA"
#define MISSING_LENGTH	-1
#define DELIMITER		","

namespace bsccs {

using namespace std;

CCTestInputReader::CCTestInputReader() : InputReader() { }

CCTestInputReader::~CCTestInputReader() { }

/**
 * Reads in a dense CSV data file with format:
 *
 * CaseSetID,DrugCount,DaysInCohort,Exposed1,Exposed2,Exposed3,Event
 *
 * from Martijn Schuemie's Jerboa
 *
 * Assumes that file is sorted by 'CaseSetID' (Stratum)
 */
void CCTestInputReader::readFile(const char* fileName) {

	const int numCovariates = 2;
	const int colStratum = 0;
	const int colOutcome = 6;
	const int colCount = 1;
	const FormatType colCountFormat = DENSE;
	const int colExposed = 3;
	const FormatType colExposedFormat = DENSE;
	const bool makeIndicator = true;

	ifstream in(fileName);
	if (!in) {
		cerr << "Unable to open " << fileName << endl;
		exit(-1);
	}

	string line;
	getline(in, line); // Read header and ignore

	int numCases = 0;
	string currentStratum = MISSING_STRING;
	int numEvents = 0;

	vector<string> strVector;
	string outerDelimiter(DELIMITER);

	modelData->push_back(INDICATOR); // Exposed
	modelData->push_back(SPARSE); // DrugCount

	int currentRow = 0;
	while (getline(in, line) && (currentRow < MAX_ENTRIES)) {
		if (!line.empty()) {

			strVector.clear();
			split(strVector, line, outerDelimiter);

			// Parse stratum (pid)
			string unmappedStratum = strVector[colStratum];
			if (unmappedStratum != currentStratum) { // New stratum, ASSUMES these are sorted
				if (currentStratum != MISSING_STRING) { // Skip first switch
					modelData->nevents.push_back(1);
					numEvents = 0;
				}
				currentStratum = unmappedStratum;
				numCases++;
			}
			modelData->pid.push_back(numCases - 1);

			// Parse outcome entry
			int thisY;
			istringstream(strVector[colOutcome]) >> thisY;
 			numEvents += thisY;
 			modelData->y.push_back(thisY);

			// Fix offs for CLR
 			modelData->offs.push_back(1);

			// Parse covariates
			int value = 0;
			istringstream(strVector[colExposed]) >> value;
			if (makeIndicator) {
				if (value != 0) {
					value = 1;
				}

			}
			modelData->getColumn(0).add_data(currentRow, value);

			value = 0;
			istringstream(strVector[colCount]) >> value;
			modelData->getColumn(1).add_data(currentRow, value);

			currentRow++;
		}
	}
	modelData->nevents.push_back(1); // Save last patient

	cout << "CCTestInputReader" << endl;
	cout << "Read " << currentRow << " data lines from " << fileName << endl;
	cout << "Number of stratum: " << numCases << endl;
	cout << "Number of covariates: " << numCovariates << endl;

//	cout << "Sum of exposed: " << std::accumulate(data[0]->begin(), data[0]->end(), static_cast<real>(0.0)) << endl;
//	cout << "Sum of count  : " << std::accumulate(data[1]->begin(), data[1]->end(), static_cast<real>(0.0)) << endl;

	modelData->nPatients = numCases;
	modelData->nRows = currentRow;
	modelData->conditionId = "0";
}

} // namespace
