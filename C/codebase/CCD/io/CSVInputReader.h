/*
* CSVInputReader.h
*
*  Created on: Aug 13, 2012
*      Author: Sushil Mittal
*/

#ifndef CSVINPUTREADER_H_
#define CSVINPUTREADER_H_

#include "InputReader.h"
#include "imputation/ImputationPolicy.h"

using namespace std;

#define MAX_ENTRIES		1000000000

#define FORMAT_MATCH_1	"CONDITION_CONCEPT_ID"
#define MATCH_LENGTH_1	20
#define FORMAT_MATCH_2	"PID"
#define MATCH_LENGTH_2	3
#define MISSING_STRING_1	"NA"
#define MISSING_STRING_2	"NULL"
#define MISSING_LENGTH	-1
#define DELIMITER		" "

template <typename ImputationPolicy>
class CSVInputReader : public InputReader{
public:
	CSVInputReader() : InputReader() {
		imputePolicy = new ImputationPolicy();
	}
	virtual ~CSVInputReader() {}

	virtual void readFile(const char* fileName) {	
		// Currently supports only DENSE columns
		ifstream in(fileName);
		if (!in) {
			cerr << "Unable to open " << fileName << endl;
			exit(-1);
		}

		string line;
		getline(in, line); // Read header and ignore

		int numCases = 0;
		int numCovariates = MISSING_LENGTH;
		string currentStratum = MISSING_STRING_1;
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
						int_vector* nullVector = new int_vector();
						imputePolicy->push_back(nullVector,0);
					}
				} else if (numCovariates != strVector.size() - 2) {
					cerr << "All rows must be the same length" << endl;
					exit(-1);
				}

				// Parse stratum (pid)
				string unmappedStratum = strVector[0];
				if (unmappedStratum != currentStratum) { // New stratum, ASSUMES these are sorted
					if (currentStratum != MISSING_STRING_1 && currentStratum != MISSING_STRING_2) { // Skip first switch
						modelData->nevents.push_back(1);
						numEvents = 0;
					}
					currentStratum = unmappedStratum;
					numCases++;
				}
				modelData->pid.push_back(numCases - 1);

				// Parse outcome entry. Also handling missingness in Y now.
				real thisY;
				if(strVector[1] == MISSING_STRING_1 || strVector[1] == MISSING_STRING_2){
					imputePolicy->push_backY(currentRow);
					thisY = 0.0;
				}
				else{
					thisY = static_cast<real>(atof(strVector[1].c_str()));
				}
				modelData->y.push_back(thisY);
				numEvents += thisY;

				// Fix offs for CLR
				modelData->offs.push_back(1);

				// Parse covariates
				for (int i = 0; i < numCovariates; ++i) {
					if(strVector[2 + i] == MISSING_STRING_1 || strVector[2 + i] == MISSING_STRING_2){
						modelData->getColumn(i).add_data(currentRow, 0.0);
						imputePolicy->push_back(i,currentRow);
					}
					else{
						real value = static_cast<real>(atof(strVector[2 + i].c_str()));
						modelData->getColumn(i).add_data(currentRow, value);
					}
				}
				currentRow++;
			}
		}

		modelData->nPatients = numCases;
		modelData->nCols = modelData->getNumberOfColumns();
		modelData->nRows = currentRow;
		modelData->conditionId = "0";

/*
		for(int j = 0; j < modelData->nCols; j++){
			if(modelData->formatType[j] == INDICATOR){
				if(modelData->columns[j])
					modelData->columns[j]->clear();
				modelData->columns[j] = new int_vector();
				for(int i = 0; i < modelData->nRows; i++)
				{
					if((int)modelData->data[j]->at(i) == 1)
						modelData->columns[j]->push_back(i);
				}
				modelData->data[j]->clear();
			}
		}

		int index = modelData->columns.size();
*/
#ifndef MY_RCPP_FLAG
		cout << "ImputeInputReader" << endl;
		cout << "Read " << currentRow << " data lines from " << fileName << endl;
		cout << "Number of stratum: " << numCases << endl;
		cout << "Number of covariates: " << numCovariates << endl;
#endif
		
	}

	ImputationPolicy* getImputationPolicy(){
		return imputePolicy;
	}

protected:
	ImputationPolicy* imputePolicy;
};

#endif /* CSVINPUTREADER_H_ */
