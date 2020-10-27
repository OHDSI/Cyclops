/*
 * ImputeVaraibles.h
 *
 *  Created on: Jul 28, 2012
 *      Author: Sushil Mittal
 */

#ifndef IMPUTEVARIABLES_H_
#define IMPUTEVARIABLES_H_

#include "ImputationPolicy.h"
#include "ccd.h"

namespace bsccs {

class ImputeVariables {
public:
	ImputeVariables();
	~ImputeVariables();
	void initialize(CCDArguments args, int numberOfImputations, bool inclY);
	void impute();
	void imputeColumn(int col);
	void getColumnToImpute(int col, real* y);
	void initializeCCDModel(int col);
	void randomizeImputationsLR(vector<real> yPred, vector<real> weights, int col);
	void randomizeImputationsLS(vector<real> yPred, vector<real> weights, int col);
	void writeImputedData(int imputationNumber);
	void resetModelData();
	void includeYVector();
	void excludeYVector();
private:
	CCDArguments arguments;
	CyclicCoordinateDescent*  ccd;
	AbstractModelSpecifics* model;
	InputReader* reader;
	ModelData* modelData;
	ImputationHelper* imputeHelper;
	int nImputations;
	bool includeY;
	FormatType formatTypeY;
};

} // namespace

#endif
