/*
* BBRInputReader.h
*
*  Created on: Aug 13, 2012
*      Author: Sushil Mittal
*/

#ifndef BBRINPUTREADER_H_
#define BBRINPUTREADER_H_

#include <numeric>

#include "io/InputReader.h"
#include "imputation/ImputationPolicy.h"
#include "io/SparseIndexer.h"
#include "io/CmdLineProgressLogger.h"

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

namespace bsccs {

template <class ImputationPolicy>
class BBRInputReader : public InputReader{
public:
	BBRInputReader() : InputReader(
	bsccs::make_shared<loggers::CoutLogger>(),
	bsccs::make_shared<loggers::CerrErrorHandler>()) {
		imputePolicy = new ImputationPolicy();
	}
	virtual ~BBRInputReader() {}

	virtual void readFile(const char* fileName){
		// Currently supports only INDICATOR and DENSE columns
		ifstream in(fileName);
		if (!in) {
			cerr << "Unable to open " << fileName << endl;
			exit(-1);
		}

		string line;
		//	getline(in, line); // Read header and ignore

		int numCases = 0;
		vector<string> strVector;
		string outerDelimiter(DELIMITER);

		SparseIndexer<double> indexer(modelData->getX());

		// Allocate a column for intercept
//		IntVector* nullVector = new IntVector();
//		imputePolicy->push_back(nullVector,0);
//		indexer.addColumn(0, DENSE);

		int currentRow = 0;		
// 		modelData->nRows = 0;
		setNumberRows(*modelData, 0);
		int maxCol = 0;
		while (getline(in, line) && (currentRow < MAX_ENTRIES)) {
			if (!line.empty()) {
				strVector.clear();
				split(strVector, line, outerDelimiter);

				modelData->nevents.push_back(1);
				numCases++;
				modelData->pid.push_back(numCases - 1);

				vector<string> thisCovariate;
				split(thisCovariate, strVector[0], ":");

				// Parse outcome and censoring (if any). Also handling missingness in Y now.
				real thisY;
				int yIndex;
				if(thisCovariate.size() == 2){
					yIndex = 1;
					real thisZ = static_cast<real>(atof(thisCovariate[0].c_str()));
					modelData->z.push_back(thisZ);
				}
				else{
					yIndex = 0;
				}
				if(thisCovariate[yIndex] == MISSING_STRING_1 || thisCovariate[yIndex] == MISSING_STRING_2){
					imputePolicy->push_backY(currentRow);
					thisY = 0.0;
				}
				else{
					thisY = static_cast<real>(atof(thisCovariate[yIndex].c_str()));
				}
				modelData->y.push_back(thisY);

				/*
				real thisY;
				if(thisCovariate[0] == MISSING_STRING_1 || thisCovariate[0] == MISSING_STRING_2){
					imputePolicy->push_backY(currentRow);
					thisY = 0.0;
				}
				else{
					thisY = static_cast<real>(atof(thisCovariate[0].c_str()));
				}
				modelData->y.push_back(thisY);

				// Parse censoring index entry if any
				if(thisCovariate.size() == 2){
					real thisZ = static_cast<real>(atof(thisCovariate[1].c_str()));
					modelData->z.push_back(thisZ);
				}
				*/
				// Fix offs for CLR
				modelData->offs.push_back(1);

				//Fill intercept
//				modelData->getColumn(0).add_data(currentRow, 1.0);

				// Parse covariates
				for (int i = 0; i < (int)strVector.size() - 1; ++i){
					thisCovariate.clear();
					split(thisCovariate, strVector[i + 1], ":");
					if((int)thisCovariate.size() == 2){
						DrugIdType drug = static_cast<DrugIdType>(atof(thisCovariate[0].c_str()));
						if (!indexer.hasColumn(drug)) {
							indexer.addColumn(drug, INDICATOR);
							IntVector* nullVector = new IntVector();
							imputePolicy->push_back(nullVector,0);
						}
						int col = indexer.getIndex(drug);
						if(thisCovariate[1] == MISSING_STRING_1 || thisCovariate[1] == MISSING_STRING_2){
							if(modelData->getX().getFormatType(col) == DENSE){
								modelData->getX().getColumn(col).add_data(currentRow, 0.0);
							}
							imputePolicy->push_back(col,currentRow);
						}
						else{
							real value = static_cast<real>(atof(thisCovariate[1].c_str()));
							if(modelData->getX().getFormatType(col) == DENSE){
								modelData->getX().getColumn(col).add_data(currentRow, value);
							}
							else if(modelData->getX().getFormatType(col) == INDICATOR){
								if(value != 1.0 && value != 0.0){
										modelData->getX().convertColumnToDense(col);
									modelData->getX().getColumn(col).add_data(currentRow,value);
								}
								else{
									if(value == 1.0){
										modelData->getX().getColumn(col).add_data(currentRow,1.0);
									}
								}
							}
						}
					}
				}
				currentRow++;
				modelData->getX().nRows++;
			}
		}

		for(int col = 0; col < modelData->getX().nCols; col++) {
			if(modelData->getX().getFormatType(col) == DENSE) {
				if(modelData->getX().getColumn(col).getDataVectorLength() < currentRow) {
					modelData->getX().getColumn(col).add_data(currentRow-1,0.0);
				}
			}
		}

#ifndef MY_RCPP_FLAG
		cout << "RTestInputReader" << endl;
		cout << "Read " << currentRow << " data lines from " << fileName << endl;
		cout << "Number of stratum: " << numCases << endl;
		cout << "Number of covariates: " << modelData->getX().nCols << endl;
#endif

		//modelData->nPatients = numCases;
		setNumberPatients(*modelData, numCases);
		modelData->getX().nRows = currentRow;
		//modelData->getX().conditionId = "0";
		setConditionId(*modelData, "0");
		
		
		//cerr << "Total 0: " << modelData->getX().getColumn(0).sumColumn(currentRow) << endl;
		//cerr << "Total 1: " << modelData->getX().getColumn(1).sumColumn(currentRow) << endl;
		//cerr << "Y: " << std::accumulate(modelData->y.begin(), modelData->y.end(), 0.0) << endl;
		//cerr << "Z: " << std::accumulate(modelData->z.begin(), modelData->z.end(), 0.0) << endl;

//		cerr << "Total 2: " << modelData->getColumn(2).sumColumn(currentRow) << endl;
	}

	ImputationPolicy* getImputationPolicy(){
		return imputePolicy;
	}

protected:
	ImputationPolicy* imputePolicy;
};

}

#endif /* BBRINPUTREADER_H_ */
