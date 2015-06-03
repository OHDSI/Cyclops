/*
 * ImputeVaraibles.cpp
 *
 *  Created on: Jul 28, 2012
 *      Author: Sushil Mittal
 */

#include "ImputeVariables.h"
#include "io/CSVInputReader.h"
#include "io/BBRInputReader.h"
#include "io/BBROutputWriter.h"
#include "GridSearchCrossValidationDriver.h"
#include "CrossValidationSelector.h"
#include <time.h>

namespace bsccs {

double randn_notrig(double mu=0.0, double sigma=1.0) {
	static bool deviateAvailable=false;        //        flag
	static float storedDeviate;                        //        deviate from previous calculation
	double polar, rsquared, var1, var2;

	//        If no deviate has been stored, the polar Box-Muller transformation is
	//        performed, producing two independent normally-distributed random
	//        deviates.  One is stored for the next round, and one is returned.
	if (!deviateAvailable) {

		//        choose pairs of uniformly distributed deviates, discarding those
		//        that don't fall within the unit circle
		do {
			var1=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
			var2=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
			rsquared=var1*var1+var2*var2;
		} while ( rsquared>=1.0 || rsquared == 0.0);

		//        calculate polar tranformation for each deviate
		polar=sqrt(-2.0*log(rsquared)/rsquared);

		//        store first deviate and set flag
		storedDeviate=var1*polar;
		deviateAvailable=true;

		//        return second deviate
		return var2*polar*sigma + mu;
	}

	//        If a deviate is available from a previous call to this function, it is
	//        returned, and the flag is set to false.
	else {
		deviateAvailable=false;
		return storedDeviate*sigma + mu;
	}
}

void getComplement(vector<real>& weights){
	for(std::vector<real>::iterator it = weights.begin(); it != weights.end(); it++) {
		*it = 1 - *it;
	}
}

ImputeVariables::ImputeVariables(){
	ccd = NULL;
	model = NULL;
}

ImputeVariables::~ImputeVariables(){
	if (modelData)
		delete modelData;
}

void ImputeVariables::initialize(CCDArguments args, int numberOfImputations, bool inclY){
	arguments = args;
	nImputations = numberOfImputations;
	includeY = inclY;
	if(arguments.fileFormat == "csv"){
		reader = new CSVInputReader<ImputationHelper>();
		static_cast<CSVInputReader<ImputationHelper>*>(reader)->readFile(arguments.inFileName.c_str());
		imputeHelper = static_cast<CSVInputReader<ImputationHelper>*>(reader)->getImputationPolicy();
		modelData = static_cast<CSVInputReader<ImputationHelper>*>(reader)->getModelData();
	}
	else if(arguments.fileFormat == "bbr"){
		reader = new BBRInputReader<ImputationHelper>();
		static_cast<BBRInputReader<ImputationHelper>*>(reader)->readFile(arguments.inFileName.c_str());
		imputeHelper = static_cast<BBRInputReader<ImputationHelper>*>(reader)->getImputationPolicy();
		modelData = static_cast<BBRInputReader<ImputationHelper>*>(reader)->getModelData();
	}
	else{
		cerr << "Invalid file format." << endl;
		exit(-1);
	}
	imputeHelper->saveOrigYVector(modelData->getYVector(), modelData->getNumberOfRows());
	srand(time(NULL));
}

void ImputeVariables::impute(){
	
	for(int i = 0; i < nImputations; i++){
		if(includeY){
			includeYVector();
		}
		imputeHelper->sortColumns();
		vector<int> sortedColIndices = imputeHelper->getSortedColIndices();
		modelData->sortDataColumns(sortedColIndices);

		vector<int> nMissingPerColumn = imputeHelper->getnMissingPerColumn();
		int nColsToImpute = 0;
		for(int j = 0; j < (int)nMissingPerColumn.size(); j++){
			if(nMissingPerColumn[j] > 0){
				nColsToImpute++;
			}
		}

		cout << "Total columns to impute = " << nColsToImpute << endl;

		for(int j = 0; j < (int)nMissingPerColumn.size(); j++){
			if(nMissingPerColumn[j] > 0){
				cout << endl << "Imputing column " << j - ((int)nMissingPerColumn.size() - nColsToImpute) << endl;
				imputeColumn(j);
			}
		}
		modelData->setNumberOfColumns(imputeHelper->getOrigNumberOfColumns());
		modelData->sortDataColumns(imputeHelper->getReverseColIndices());
		imputeHelper->resortColumns();
		
		if(includeY){
			excludeYVector();
		}
		else{
			modelData->setYVector(imputeHelper->getOrigYVector());
			modelData->setNumberOfColumns(imputeHelper->getOrigNumberOfColumns());
		}

		writeImputedData(i);
		resetModelData();
	}
}

void ImputeVariables::imputeColumn(int col){
	
	int nRows = modelData->getNumberOfRows();

	vector<real> y(nRows,0.0);
	getColumnToImpute(col, &y[0]);

	modelData->setYVector(y);
	modelData->setNumberOfColumns(col);

	vector<real> weights(nRows,1.0);
	imputeHelper->setWeightsForImputation(col,weights,nRows);

	vector<real> weightsMissing = weights;
	getComplement(weightsMissing);

	initializeCCDModel(col);
	// Do cross validation for finding optimum hyperparameter value
	CrossValidationSelector selector(arguments.fold, modelData->getPidVectorSTL(), SUBJECT, rand(), &weightsMissing);
	GridSearchCrossValidationDriver driver(arguments.gridSteps, arguments.lowerLimit, arguments.upperLimit, &weightsMissing);
	driver.drive(*ccd, selector, arguments);
//	driver.logResults(arguments);
	driver.resetForOptimal(*ccd, selector, arguments);

	//initializeCCDModel(col,weights);
	// No need to initialize ccd again, just reset weights.
	ccd->setWeights(&weights[0]);

	ccd->update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);
	getComplement(weights);

	vector<real> yPred(nRows,0.0);

	vector<real> allOnes(nRows,1.0);

	ccd->getPredictiveEstimates(&yPred[0], &allOnes[0]);

	srand(time(NULL));
	if(modelData->getFormatType(col) == DENSE)
		randomizeImputationsLS(yPred, weights, col);
	else
		randomizeImputationsLR(yPred, weights, col);

	if (ccd)
		delete ccd;
	if (model)
		delete model;
}

void ImputeVariables::getColumnToImpute(int col, real* y){
	int nRows = modelData->getNumberOfRows();
	if(modelData->getFormatType(col) == DENSE){
		real* dataVec = modelData->getDataVector(col);
		for(int i = 0; i < nRows; i++)
			y[i] = dataVec[i];
	}
	else{
		int* columnVec = modelData->getCompressedColumnVector(col);
		for(int i = 0; i < modelData->getNumberOfEntries(col); i++)
			y[columnVec[i]] = 1.0;
	}
}

void ImputeVariables::initializeCCDModel(int col){
	if(modelData->getFormatType(col) == DENSE)	//Least Squares
		model = new ModelSpecifics<LeastSquares<real>,real>(*modelData);
	else										//Logistic Regression
		model = new ModelSpecifics<LogisticRegression<real>,real>(*modelData);

	using namespace bsccs::priors;
	PriorPtr covariatePrior;
	if (arguments.useNormalPrior) {
		covariatePrior = std::make_shared<NormalPrior>();
	} else {
		covariatePrior = std::make_shared<LaplacePrior>();
	}

	covariatePrior->setVariance(arguments.hyperprior);

	JointPriorPtr prior(new priors::FullyExchangeableJointPrior(covariatePrior));
	ccd = new CyclicCoordinateDescent(modelData, *model, prior);
}

void ImputeVariables::randomizeImputationsLR(vector<real> yPred, vector<real> weights, int col){
	if(modelData->getFormatType(col) == INDICATOR){
		int nRows = modelData->getNumberOfRows();
		vector<int> y;
		for(int i = 0; i < nRows; i++){
			if(weights[i]){
				real r = (real)rand()/RAND_MAX;
				if(r < yPred[i])
					y.push_back(i);
			}
		}
		modelData->addToColumnVector(col,y);
	}
	else{
		real* y = modelData->getDataVector(col);
		for(int i = 0; i < modelData->getNumberOfRows(); i++){
			if(weights[i]){
				real r = (real)rand()/RAND_MAX;
				if(r < yPred[i])
					y[i] = 1.0;
				else
					y[i] = 0.0;
			}
		}
	}
}

void ImputeVariables::randomizeImputationsLS(vector<real> yPred, vector<real> weights, int col){
	
	int n = 0;
	
	int nRows = modelData->getNumberOfRows();
	vector<real> dist(nRows,0.0);

	vector<real> xMean;
	vector<real> xVar;

	for(int j = 0; j < col; j++){
		real* dataVec;
		int* columnVec;
		int nEntries = 0;
		FormatType formatType = modelData->getFormatType(j);
		if(formatType == DENSE){
			dataVec = modelData->getDataVector(j);
		}
		else{
			columnVec = modelData->getCompressedColumnVector(j);
			nEntries = modelData->getNumberOfEntries(j);
		}
		real mean,var;
		imputeHelper->getSampleMeanVariance(col,mean,var,dataVec,columnVec,formatType,nRows,nEntries);
		xMean.push_back(mean);
		xVar.push_back(var);
	}

	real* y = modelData->getDataVector(col);

	real sigma = 0.0;
	for(int i = 0; i < nRows; i++){
		if(!weights[i]){
			sigma += (y[i] - yPred[i]) * (y[i] - yPred[i]);
			n++;
		}
		else{
			vector<real> x(col);
			modelData->getDataRow(i,&x[0]);
			for(int j = 0; j < col; j++){
				if(xVar[j] > 1e-10)
					dist[i] += (x[j] - xMean[j]) * (x[j] - xMean[j]) / xVar[j] ;
			}
		}
	}
	sigma = sqrt(sigma/(n-2));

	for(int i = 0; i < nRows; i++){
		if(weights[i]){
			double predInterval = 1.96 * sigma * sqrt(1 + 1/n + dist[i]/(n-1));
			y[i] = randn_notrig(yPred[i],predInterval);
//			y[i] = yPred[i];
		}
	}
}

void ImputeVariables::writeImputedData(int imputationNumber){
	string outFileName = arguments.inFileName;
	int dotind = outFileName.find_last_of('.');
	std::ostringstream addendum;
	addendum << "_imputed_" << imputationNumber;
	outFileName.insert(dotind,addendum.str());

	if(arguments.fileFormat == "bbr"){
		BBROutputWriter* bbrout = new BBROutputWriter();
		bbrout->writeFile(outFileName.c_str(),modelData);
	}
}

void ImputeVariables::resetModelData(){
	int nCols = modelData->getNumberOfColumns();
	for(int j = 0; j < nCols; j++){
		if(modelData->getFormatType(j) == INDICATOR){
			vector<int> missing;
			imputeHelper->getMissingEntries(j,missing);
			modelData->removeFromColumnVector(j,missing);
		}
	}
}

void ImputeVariables::includeYVector(){
	imputeHelper->includeYVector();

	vector<real> y = imputeHelper->getOrigYVector();
	modelData->setYVector(y);

	formatTypeY = INDICATOR;
	for(int i = 0; i < modelData->getNumberOfRows(); i++){
		if(y[i] != 0.0 && y[i] != 1.0){
			formatTypeY = DENSE;
			break;
		}
	}
	modelData->push_back(formatTypeY);

	int colY = modelData->getNumberOfColumns();
	if(formatTypeY == DENSE){
		for(int i = 0; i < modelData->getNumberOfRows(); i++){
			modelData->getColumn(colY-1).add_data(i,y[i]);
		}
	}
	else{
		for(int i = 0; i < modelData->getNumberOfRows(); i++){
			if(y[i] ==  1.0){
				modelData->getColumn(colY-1).add_data(i,1.0);
			}
		}
	}
}

void ImputeVariables::excludeYVector(){
	int colY = imputeHelper->getOrigNumberOfColumns()-1;
	vector<real> y(modelData->getNumberOfRows(),0.0);
	if(formatTypeY == DENSE){
		real* y_real = modelData->getDataVector(colY);
		for(int i = 0; i < modelData->getNumberOfRows(); i++){
			y[i] = y_real[i];
		}
	}
	else{
		int* y_int = modelData->getCompressedColumnVector(colY);
		for(int i = 0; i < modelData->getNumberOfEntries(colY); i++){
			y[y_int[i]] = 1.0;
		}
	}
	modelData->setYVector(y);
	modelData->erase(colY);
	imputeHelper->pop_back();
}

} // namespace
