/*
 * CyclicCoordinateDescent.cpp
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>


#include "CyclicCoordinateDescent.h"
#include "InputReader.h"
#include "Iterators.h"
#include "SparseRowVector.h"


//#ifdef MY_RCPP_FLAG
//	#include <R.h>
//#else
//	void Rprintf(...) {
//		// Do nothing
//	}
//#endif

#ifndef MY_RCPP_FLAG
#define PI	3.14159265358979323851280895940618620443274267017841339111328125
#else
//#include "Rcpp.h"
#endif

//using namespace Eigen::MatrixX;
//using namespace std;

void compareIntVector(int* vec0, int* vec1, int dim, const char* name) {
	for (int i = 0; i < dim; i++) {
		if (vec0[i] != vec1[i]) {
			cerr << "Error at " << name << "[" << i << "]: ";
			cerr << vec0[i] << " != " << vec1[i] << endl;
			exit(0);
		}
	}
}

CyclicCoordinateDescent::CyclicCoordinateDescent() {
	// Do nothing
}

CyclicCoordinateDescent::CyclicCoordinateDescent(
			InputReader* reader
		) {
	N = reader->getNumberOfPatients();
	K = reader->getNumberOfRows();
	J = reader->getNumberOfColumns();

	hXI = reader;
	hEta = reader->getEtaVector();
	hOffs = reader->getOffsetVector();
	hNEvents = NULL;
	hPid = reader->getPidVector();

	conditionId = reader->getConditionId();
	denomNullValue = static_cast<realTRS>(0.0);

	updateCount = 0;
	likelihoodCount = 0;

	/*
	SparseRowVector test;
	test.fillSparseRowVector(reader);

	cout << "Printing Trans by Col" << endl;
	cout << "Trans Rows = " << test.getNumberOfRows() << endl;
	cout << "Trans Cols = " << test.getNumberOfColumns() << endl;

	for (int j = 0; j < test.getNumberOfColumns(); j++) {
		test.printRow(j);
	}


	cout << "Printing Non-trans by Col!" << endl;

	cout << "n Rows = " << hXI->getNumberOfRows() << endl;
	cout << "n Cols = " << hXI->getNumberOfColumns() << endl;
	for (int i = 0; i < hXI->getNumberOfColumns(); i++) {
		hXI->printColumn(i);
	}
*/
	init();
}

//CyclicCoordinateDescent::CyclicCoordinateDescent(
//		int inN,
//		CompressedIndicatorMatrix* inX,
//		int* inEta,
//		int* inOffs,
//		int* inNEvents,
//		int* inPid) :
//	N(inN), hXI(inX), hEta(inEta), hOffs(inOffs), hNEvents(inNEvents), hPid(inPid) {
//
//	K = hXI->getNumberOfRows();
//	J = hXI->getNumberOfColumns();
//
//	init();
//}

CyclicCoordinateDescent::~CyclicCoordinateDescent(void) {

	free(hPid);
	free(hNEvents);
	free(hEta);
	free(hOffs);
	
	free(hXBeta);
	free(hXBetaSave);
	free(hDelta);
	
#ifdef TEST_ROW_INDEX
	for (int j = 0; j < J; ++j) {
		if (hXColumnRowIndicators[j]) {
			free(hXColumnRowIndicators[j]);
		}
	}
	free(hXColumnRowIndicators);
#endif

	free(hXjEta);
	free(offsExpXBeta);
	free(xOffsExpXBeta);
//	free(denomPid);  // Nested in denomPid allocation
	free(numerPid);
//	free(t1);
	
#ifdef NO_FUSE
	free(wPid);
#endif
	
	if (hWeights) {
		free(hWeights);
	}

#ifdef SPARSE_PRODUCT
	for (std::vector<std::vector<int>* >::iterator it = sparseIndices.begin(); it != sparseIndices.end(); ++it) {
		if (*it) {
			delete *it;
		}
	}
#endif
}

string CyclicCoordinateDescent::getPriorInfo() {
#ifdef MY_RCPP_FLAG
	return "prior"; // Rcpp error with stringstream
#else
	stringstream priorInfo;
	if (priorType == LAPLACE) {
		priorInfo << "Laplace(";
		priorInfo << lambda;
	} else if (priorType == NORMAL) {
		priorInfo << "Normal(";
		priorInfo << sigma2Beta;
	}
	priorInfo << ")";
	return priorInfo.str();
#endif
}

void CyclicCoordinateDescent::resetBounds() {
	for (int j = 0; j < J; j++) {
		hDelta[j] = 2.0;
	}
}

void CyclicCoordinateDescent::init() {
	
	// Set parameters and statistics space
	hDelta = (realTRS*) malloc(J * sizeof(realTRS));
//	for (int j = 0; j < J; j++) {
//		hDelta[j] = 2.0;
//	}

	hBeta = (realTRS*) calloc(J, sizeof(realTRS)); // Fixed starting state
	hXBeta = (realTRS*) calloc(K, sizeof(realTRS));
	hXBetaSave = (realTRS*) calloc(K, sizeof(realTRS));

	// Set prior
	priorType = LAPLACE;
	sigma2Beta = 1000;
	lambda = sqrt(2.0/20.0);
	
	// Recode patient ids
	int currentNewId = 0;
	int currentOldId = hPid[0];
	
	for(int i = 0; i < K; i++) {
		if (hPid[i] != currentOldId) {			
			currentOldId = hPid[i];
			currentNewId++;
		}
		hPid[i] = currentNewId;
	}
		
	// Init temporary variables
	offsExpXBeta = (realTRS*) malloc(sizeof(realTRS) * K);
	xOffsExpXBeta = (realTRS*) malloc(sizeof(realTRS) * K);

	// Put numer and denom in single memory block, with first entries on 16-word boundary
	int alignedLength = getAlignedLength(N);

	numerPid = (realTRS*) malloc(sizeof(realTRS) * 3 * alignedLength);
//	denomPid = (realTRS*) malloc(sizeof(realTRS) * N);
	denomPid = numerPid + alignedLength; // Nested in denomPid allocation
	numerPid2 = numerPid + 2 * alignedLength;
//	t1 = (realTRS*) malloc(sizeof(realTRS) * N);
	hNEvents = (int*) malloc(sizeof(int) * N);
	hXjEta = (realTRS*) malloc(sizeof(realTRS) * J);
	hWeights = NULL;
	
#ifdef NO_FUSE
	wPid = (realTRS*) malloc(sizeof(realTRS) * alignedLength);
#endif

	for (int j = 0; j < J; ++j) {
		if (hXI->getFormatType(j) == DENSE) {
			sparseIndices.push_back(NULL);
		} else {
			std::set<int> unique;
			const int n = hXI->getNumberOfEntries(j);
			const int* indicators = hXI->getCompressedColumnVector(j);
			for (int j = 0; j < n; j++) { // Loop through non-zero entries only
				const int k = indicators[j];
				const int i = hPid[k];
				unique.insert(i);
			}
			std::vector<int>* indices = new std::vector<int>(unique.begin(),
					unique.end());
			sparseIndices.push_back(indices);
		}
	}

	useCrossValidation = false;
	validWeights = false;
	sufficientStatisticsKnown = false;
	xBetaKnown = true;
	doLogisticRegression = false;

#ifdef DEBUG	
#ifndef MY_RCPP_FLAG
	cerr << "Number of patients = " << N << endl;
	cerr << "Number of exposure levels = " << K << endl;
	cerr << "Number of drugs = " << J << endl;
#endif
#endif          

//#ifdef MY_RCPP_FLAG
//	Rprintf("Number of patients = %d\n", N);
//	Rprintf("Number of exposure levels = %d\n", K);
//	Rprintf("Number of drugs = %d\n", J);
//#endif

}

int CyclicCoordinateDescent::getAlignedLength(int N) {
	return (N / 16) * 16 + (N % 16 == 0 ? 0 : 16);
}

void CyclicCoordinateDescent::computeNEvents() {
	zeroVector(hNEvents, N);
	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			int events = doLogisticRegression ? 1 : hEta[i];
			hNEvents[hPid[i]] += events * int(hWeights[i]); // TODO Consider using only integer weights
		}
	} else {
		for (int i = 0; i < K; i++) {
			int events = doLogisticRegression ? 1 : hEta[i];
			hNEvents[hPid[i]] += events;
		}
	}
	validWeights = true;
}

void CyclicCoordinateDescent::resetBeta(void) {
	for (int j = 0; j < J; j++) {
		hBeta[j] = 0.0;
	}
	for (int k = 0; k < K; k++) {
		hXBeta[k] = 0.0;
	}
	sufficientStatisticsKnown = false;
}

//void double getHessianComponent(int i, int j) {
//	return 0.0;
//}

void CyclicCoordinateDescent::logResults(const char* fileName) {

	ofstream outLog(fileName);
	if (!outLog) {
		cerr << "Unable to open log file: " << fileName << endl;
		exit(-1);
	}

	InputReader* reader = dynamic_cast<InputReader*>(hXI);
	map<int, DrugIdType> drugMap = reader->getDrugNameMap();

	string sep(","); // TODO Make option

	outLog << "Drug_concept_id" << sep << "Condition_concept_id" << sep << "score" << endl;

	for (int i = 0; i < J; i++) {
		outLog << drugMap[i] << sep <<
		conditionId << sep << hBeta[i] << endl;
	}
	outLog.flush();
	outLog.close();
}

double CyclicCoordinateDescent::getPredictiveLogLikelihood(realTRS* weights) {

	if (!xBetaKnown) {
		computeXBeta();
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Remove index????
	}

	getDenominators();

	double logLikelihood = 0;

	for (int i = 0; i < K; i++) {
		logLikelihood += hEta[i] * weights[i] * (hXBeta[i] - log(denomPid[hPid[i]]));
	}

	return logLikelihood;
}

int CyclicCoordinateDescent::getBetaSize(void) {
	return J;
}

realTRS CyclicCoordinateDescent::getBeta(int i) {
	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, i);
	}
	return hBeta[i];
}

double CyclicCoordinateDescent::getLogLikelihood(void) {

	if (!xBetaKnown) {
		computeXBeta();
	}

	if (!validWeights) {
		computeNEvents();
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Check index?
	}

//	double sum1 = 0.0;
//	double sum2 = 0.0;

	getDenominators();

	realTRS logLikelihood = 0;

	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			logLikelihood += hEta[i] * hXBeta[i] * hWeights[i];
		}
	} else {
		for (int i = 0; i < K; i++) {
			logLikelihood += hEta[i] * hXBeta[i];
//			sum1 += hEta[i];
		}
	}

	for (int i = 0; i < N; i++) {
		logLikelihood -= hNEvents[i] * log(denomPid[i]); // Weights modified in computeNEvents()
//		sum2 += hNEvents[i];
	}

//#ifdef MY_RCPP_FLAG
//	Rprintf("xbeta[0] = %f\n", hXBeta[0]);
//	Rprintf("beta[0]  = %f\n", hBeta[0]);
//	Rprintf("eta = %f\n", sum1);
//	Rprintf("events = %f\n", sum2);
//
////	typedef Rcpp::Rout cerr;
////	cerr << "hello" << endl;
//#else
//	cerr << "xbeta[0] = " << hXBeta[0] << endl;
//	cerr << "eta = " << sum1 << endl;
//	cerr << "events = " << sum2 << endl;
//#endif

	likelihoodCount += 1;
	return static_cast<double>(logLikelihood);
}

void CyclicCoordinateDescent::getDenominators() {
	// Do nothing
}

double convertVarianceToHyperparameter(double value) {
	return sqrt(2.0 / value);
}

void CyclicCoordinateDescent::setHyperprior(double value) {
	sigma2Beta = value;
	lambda = convertVarianceToHyperparameter(value);
}

void CyclicCoordinateDescent::setLogisticRegression(bool idoLR) {
	std::cerr << "Setting LR to " << idoLR << std::endl;
	doLogisticRegression = idoLR;
	denomNullValue = static_cast<realTRS>(1.0);
	validWeights = false;
	sufficientStatisticsKnown = false;
}

void CyclicCoordinateDescent::getHessianForCholesky_GSL() {
	//naming of variables consistent with Suchard write up page 23 Appendix A

	vector<int> giVector(N, 0); // get Gi given patient i

	vector<vector<int> > kValues;

	for (int j = 0; j < K; j++) {
		giVector[hPid[j]] ++;
	}

	for (int n = 0; n < N; n++) {
		vector<int> temp;
		kValues.push_back(temp);
	}

	for (int j = 0; j < K; j++) {
		kValues[hPid[j]].push_back(j);
	}

	gsl_matrix * HessianMatrix = gsl_matrix_alloc(J,J);
	gsl_matrix_set_zero(HessianMatrix);

	for (int i = 0; i < N; i ++) { // sum over patients
		gsl_matrix * FirstTerm = gsl_matrix_alloc(J, J);
		gsl_matrix * SecondTerm = gsl_matrix_alloc(J, J);
		int ni = hNEvents[i];

		for (int j = 0; j < J; j++) {
			for (int j2 = 0; j2 < J; j2 ++) {
				computeNumeratorForGradient(j);
				double firstNumer = numerPid[i];
				computeNumeratorForGradient(j2);
				double secondNumer = numerPid[i];
				double FirstTermjj2 = firstNumer*secondNumer / (denomPid[i]*denomPid[i]);
				gsl_matrix_set(FirstTerm, j, j2, FirstTermjj2);
				realTRS * columnVectorj = hXI->getDataVector(j);
				realTRS * columnVectorj2 = hXI->getDataVector(j2);
				double SecondTermjj2 = 0;
				for (int g = 0; g < kValues[i].size(); g++) {
					SecondTermjj2 += offsExpXBeta[kValues[i][g]]*columnVectorj[kValues[i][g]]*columnVectorj2[kValues[i][g]] / (denomPid[i]);
				}
				gsl_matrix_set(SecondTerm, j, j2, SecondTermjj2);

			}
		}

		gsl_matrix_scale(FirstTerm, -ni);
		gsl_matrix_scale(SecondTerm, -ni);
		gsl_matrix_add(HessianMatrix, FirstTerm);
		gsl_matrix_sub(HessianMatrix, SecondTerm);
	}

	int test = gsl_linalg_cholesky_decomp(HessianMatrix);


}


void CyclicCoordinateDescent::getHessianForCholesky_Eigen(Eigen::MatrixXf* ReturnHessian) {
	//naming of variables consistent with Suchard write up page 23 Appendix A

	vector<int> giVector(N, 0); // get Gi given patient i

	vector<vector<int> > kValues;

	for (int j = 0; j < K; j++) {
		giVector[hPid[j]] ++;
	}

	for (int n = 0; n < N; n++) {
		vector<int> temp;
		kValues.push_back(temp);
	}

	for (int j = 0; j < K; j++) {
		kValues[hPid[j]].push_back(j);
	}

	Eigen::MatrixXf HessianMatrix_Eigen(J, J);
	HessianMatrix_Eigen.setZero(J, J);

	for (int i = 0; i < N; i ++) { // sum over patients
		Eigen::MatrixXf FirstTerm_Eigen(J, J);
		Eigen::MatrixXf SecondTerm_Eigen(J, J);

		int ni = hNEvents[i];

		for (int j = 0; j < J; j++) {
			for (int j2 = 0; j2 < J; j2 ++) {
				computeNumeratorForGradient(j);
				double firstNumer = numerPid[i];
				computeNumeratorForGradient(j2);
				double secondNumer = numerPid[i];
				double FirstTermjj2 = firstNumer*secondNumer / (denomPid[i]*denomPid[i]);
				FirstTerm_Eigen(j, j2) = FirstTermjj2;
				realTRS * columnVectorj = hXI->getDataVector(j);
				realTRS * columnVectorj2 = hXI->getDataVector(j2);
				double SecondTermjj2 = 0;
				for (int g = 0; g < kValues[i].size(); g++) {
					SecondTermjj2 += offsExpXBeta[kValues[i][g]]*columnVectorj[kValues[i][g]]*columnVectorj2[kValues[i][g]] / (denomPid[i]);
				}
				SecondTerm_Eigen(j, j2) = SecondTermjj2;
			}
		}

		FirstTerm_Eigen *= -ni;
		SecondTerm_Eigen *= -ni;
		HessianMatrix_Eigen += FirstTerm_Eigen;

		HessianMatrix_Eigen -= SecondTerm_Eigen;
	}

	Eigen::MatrixXf CholeskyDecomp(J, J);
	Eigen::LLT<Eigen::MatrixXf> CholDecom(HessianMatrix_Eigen);
	CholeskyDecomp = CholDecom.matrixL();
	*ReturnHessian = HessianMatrix_Eigen; //CholDecom.matrixL();

}

void CyclicCoordinateDescent::getCholeskyFromHessian(Eigen::MatrixXf* HessianMatrix, Eigen::MatrixXf* CholeskyDecomp){
	Eigen::MatrixXf CholeskyDecompL(J, J);
	Eigen::LLT<Eigen::MatrixXf> CholDecom(*HessianMatrix);
	CholeskyDecompL = CholDecom.matrixL();
	*CholeskyDecomp = CholDecom.matrixL();
}


void CyclicCoordinateDescent::setPriorType(int iPriorType) {
	if (iPriorType != LAPLACE && iPriorType != NORMAL) {
		cerr << "Unknown prior type" << endl;
		exit(-1);
	}
	priorType = iPriorType;
}

//template <typename T>
void CyclicCoordinateDescent::setBeta(const std::vector<double>& beta) {
	for (int j = 0; j < J; ++j) {
		hBeta[j] = static_cast<realTRS>(beta[j]);
	}
	xBetaKnown = false;
}

void CyclicCoordinateDescent::setWeights(realTRS* iWeights) {

	if (iWeights == NULL) {
		std::cerr << "Turning off weights!" << std::endl;
		// Turn off weights
		useCrossValidation = false;
		validWeights = false;
		sufficientStatisticsKnown = false;
		return;
	}

	if (hWeights == NULL) {
		hWeights = (realTRS*) malloc(sizeof(realTRS) * K);
	}
	for (int i = 0; i < K; ++i) {
		hWeights[i] = iWeights[i];
	}
	useCrossValidation = true;
	validWeights = false;
	sufficientStatisticsKnown = false;
}
	
double CyclicCoordinateDescent::getLogPrior(void) {
	if (priorType == LAPLACE) {
		return J * log(0.5 * lambda) - lambda * oneNorm(hBeta, J);
	} else {
		return -0.5 * J * log(2.0 * PI * sigma2Beta) - 0.5 * twoNormSquared(hBeta, J) / sigma2Beta;		
	}
}

double CyclicCoordinateDescent::getObjectiveFunction(void) {	
//	return getLogLikelihood() + getLogPrior(); // This is LANGE
	realTRS criterion = 0;
	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			criterion += hXBeta[i] * hEta[i] * hWeights[i];
		}
	} else {
		for (int i = 0; i < K; i++) {
			criterion += hXBeta[i] * hEta[i];
		}
	}
	return static_cast<double>(criterion);
}

double CyclicCoordinateDescent::computeZhangOlesConvergenceCriterion(void) {
	double sumAbsDiffs = 0;
	double sumAbsResiduals = 0;
	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			sumAbsDiffs += abs(hXBeta[i] - hXBetaSave[i]) * hEta[i] * hWeights[i];
			sumAbsResiduals += abs(hXBeta[i]) * hEta[i] * hWeights[i];
		}
	} else {
		for (int i = 0; i < K; i++) {
			sumAbsDiffs += abs(hXBeta[i] - hXBetaSave[i]) * hEta[i];
			sumAbsResiduals += abs(hXBeta[i]) * hEta[i];
		}
	}
	return sumAbsDiffs / (1.0 + sumAbsResiduals);
}

void CyclicCoordinateDescent::saveXBeta(void) {
	memcpy(hXBetaSave, hXBeta, K * sizeof(realTRS));
}

void CyclicCoordinateDescent::update(
		int maxIterations,
		int convergenceType,
		double epsilon
		) {


	if (convergenceType != LANGE && convergenceType != ZHANG_OLES) {
		cerr << "Unknown convergence criterion" << endl;
		exit(-1);
	}

	if (!validWeights) {
		computeXjEta();
		computeNEvents();
	}

	if (!xBetaKnown) {
		computeXBeta();
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Check index?
	}

	resetBounds();


	bool done = false;
	int iteration = 0;
	double lastObjFunc;

	if (convergenceType == LANGE) {
		lastObjFunc = getObjectiveFunction();
	} else { // ZHANG_OLES
		saveXBeta();
	}
	


	while (!done) {
	
		// Do a complete cycle
		for(int index = 0; index < J; index++) {
		
			double delta = ccdUpdateBeta(index);
			delta = applyBounds(delta, index);
			if (delta != 0.0) {
				sufficientStatisticsKnown = false;
				updateSufficientStatistics(delta, index);
			}
			
			if ((index+1) % 100 == 0) {
				cout << "Finished variable " << (index+1) << endl;
			}
			
		}
		
		iteration++;
		bool checkConvergence = (iteration % J == 0 || iteration == maxIterations);
		//bool checkConvergence = true; // Check after each complete cycle

		if (checkConvergence) {

			double conv;
			if (convergenceType == LANGE) {
				double thisObjFunc = getObjectiveFunction();
				conv = computeConvergenceCriterion(thisObjFunc, lastObjFunc);
				lastObjFunc = thisObjFunc;
			} else { // ZHANG_OLES
				conv = computeZhangOlesConvergenceCriterion();
				saveXBeta();
			} // Necessary to call getObjFxn or computeZO before getLogLikelihood,
			  // since these copy over XBeta

			double thisLogLikelihood = getLogLikelihood();
			double thisLogPrior = getLogPrior();
			double thisLogPost = thisLogLikelihood + thisLogPrior;
			cout << endl;
			printVector(hBeta, J, cout);
			cout << endl;
			cout << "log post: " << thisLogPost
				 << " (" << thisLogLikelihood << " + " << thisLogPrior
			     << ") (iter:" << iteration << ") ";

			if (epsilon > 0 && conv < epsilon) {
				cout << "Reached convergence criterion" << endl;
				done = true;
			} else if (iteration == maxIterations) {
				cout << "Reached maximum iterations" << endl;
				done = true;
			} else {
				cout << endl;
			}
		}				
	}
	updateCount += 1;
}

/**
 * Computationally heavy functions
 */

void CyclicCoordinateDescent::computeGradientAndHessian(int index, double *ogradient,
		double *ohessian) {
	// Run-time dispatch
	switch (hXI->getFormatType(index)) {
	case INDICATOR :
//		computeGradientAndHessianImplHand(index, ogradient, ohessian);
		computeGradientAndHessianImpl<IndicatorIterator>(index, ogradient, ohessian);
		break;
	case SPARSE :
		computeGradientAndHessianImpl<SparseIterator>(index, ogradient, ohessian);
		break;
	case DENSE :
		computeGradientAndHessianImpl<DenseIterator>(index, ogradient, ohessian);
		break;
	}
}

//void CyclicCoordinateDescent::computeGradientAndHessianImplHand(int index, double *ogradient,
//		double *ohessian) {
//	realTRS gradient = 0;
//	realTRS hessian = 0;
//	for (int k = 0; k < N; ++k) {
//		const realTRS t = numerPid[k] / denomPid[k];
//		const realTRS g = hNEvents[k] * t;
//		gradient += g;
//		hessian += g * (static_cast<realTRS>(1.0) - t); // TODO Update for !indicators
//	}
//
//	gradient -= hXjEta[index];
//	*ogradient = static_cast<double>(gradient);
//	*ohessian = static_cast<double>(hessian);
//}

//#define LR

//template <>
//inline void CyclicCoordinateDescent::incrementGradientAndHessian<IndicatorIterator>(
//		realTRS* gradient, realTRS* hessian,
//		realTRS numer, realTRS numer2, realTRS denom, int nEvents) {
//	const realTRS t = numer / denom;
//	const realTRS g = nEvents * t;
//	*gradient += g;
//	*hessian += g * (static_cast<realTRS>(1.0) - t);
//}

template <class IteratorType>
inline void CyclicCoordinateDescent::incrementGradientAndHessian(
		realTRS* gradient, realTRS* hessian,
		realTRS numer, realTRS numer2, realTRS denom, int nEvents) {

	const realTRS t = numer / denom;
	const realTRS g = nEvents * t;
	*gradient += g;
	if (IteratorType::isIndicator) {
		*hessian += g * (static_cast<realTRS>(1.0) - t);
	} else {
		*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
	}
}

//template <class IteratorType>
//inline void CyclicCoordinateDescent::incrementGradientAndHessian(
//		realTRS* gradient, realTRS* hessian,
//		realTRS numer, realTRS numer2, realTRS denom, int nEvents) {
//
//	const realTRS t = numer / denom;
//	const realTRS g = nEvents * t;
//	*gradient += g;
//	if (IteratorType::isIndicator) {
//		*hessian += g * (static_cast<realTRS>(1.0) - t);
//	} else {
//		*hessian += nEvents * (numer2 / denom - t * t); // Bounded by x_j^2
//	}
//}

//template <>
//inline void CyclicCoordinateDescent::incrementGradientAndHessian<SparseIterator>(
//		realTRS* gradient, realTRS* hessian,
//		realTRS numer, realTRS numer2, realTRS denom, int nEvents) {
//	const realTRS t = numer / denom;
//	const realTRS g = nEvents * t;
//	*gradient += g;
////	const realTRS h1 = nEvents * (numer2 * denom - numer * numer) / (denom * denom);
//	const realTRS h1 = nEvents * (numer2 / denom - t * t);
//	const realTRS h2 =  g * (static_cast<realTRS>(1.0) - t);
//	*hessian += g * (static_cast<realTRS>(1.0) - t);
//	cerr << "Sparse it! " << h1 << " " << h2 << endl;
//	exit(-1);
//}
//
//template <>
//inline void CyclicCoordinateDescent::incrementGradientAndHessian<DenseIterator>(
//		realTRS* gradient, realTRS* hessian,
//		realTRS numer, realTRS numer2, realTRS denom, int nEvents) {
//	const realTRS t = numer / denom;
//	const realTRS g = nEvents * t;
//	*gradient += g;
//	const realTRS h1 = nEvents * (numer2 * denom - numer * numer) / (denom * denom);
//	const realTRS h2 =  g * (static_cast<realTRS>(1.0) - t);
//	*hessian += g * (static_cast<realTRS>(1.0) - t);
//	cerr << "Dense it! " << h1 << " " << h2 << endl;
////	exit(-1);
//}

void CyclicCoordinateDescent::computeGradientAndHessianImplHand(int index, double *ogradient,
		double *ohessian) {
	realTRS gradient = 0;
	realTRS hessian = 0;

	std::vector<int>::iterator it = sparseIndices[index]->begin();
	const std::vector<int>::iterator end = sparseIndices[index]->end();

	for (; it != end; ++it) {

		const int k = *it;
		incrementGradientAndHessian<IndicatorIterator>(
				&gradient, &hessian,
				numerPid[k], numerPid2[k], denomPid[k], hNEvents[k]
		);
	}
	gradient -= hXjEta[index];

	*ogradient = static_cast<double>(gradient);
	*ohessian = static_cast<double>(hessian);
}

template <class IteratorType>
void CyclicCoordinateDescent::computeGradientAndHessianImpl(int index, double *ogradient,
		double *ohessian) {
	realTRS gradient = 0;
	realTRS hessian = 0;
	
	IteratorType it(*sparseIndices[index], N); // TODO How to create with different constructor signatures?
	for (; it; ++it) {
		const int k = it.index();
		incrementGradientAndHessian<IteratorType>(
				&gradient, &hessian,
				numerPid[k], numerPid2[k], denomPid[k], hNEvents[k]
		);
	}
	gradient -= hXjEta[index];

	*ogradient = static_cast<double>(gradient);
	*ohessian = static_cast<double>(hessian);
}

//template <class IteratorType>
//void CyclicCoordinateDescent::computeHessianImpl(int index_i, int index_j, double *ohessian) {
//	realTRS hessian = 0;
//	IteratorType it = IteratorType::intersection(*sparseIndices[index_i], *sparseIndices[index_j], N);
//	for (; it; ++it) {
//		const int k = it.index();
//		incrementHessian<IteratorType(
//				&hessian,
//				numerPid[k], numerPid2[k], denomPid[k], hNEvents[k]
//		);
//	}
//	*ohessian = static_cast<double>(hessian);
//}

//#define MD

void CyclicCoordinateDescent::computeNumeratorForGradient(int index) {
	// Run-time delegation
	switch (hXI->getFormatType(index)) {
		case INDICATOR : {
			IndicatorIterator it(*sparseIndices[index]);
			for (; it; ++it) { // Only affected entries
				numerPid[it.index()] = static_cast<realTRS>(0.0);
			}
			incrementNumeratorForGradientImpl<IndicatorIterator>(index);
//			incrementNumeratorForGradientImplHand(index);
			}
			break;
		case DENSE :
			zeroVector(numerPid, N);
			zeroVector(numerPid2, N);
			incrementNumeratorForGradientImpl<DenseIterator>(index);
			break;
		case SPARSE : {
			IndicatorIterator it(*sparseIndices[index]);
			for (; it; ++it) { // Only affected entries
				numerPid[it.index()] = static_cast<realTRS>(0.0);
				numerPid2[it.index()] = static_cast<realTRS>(0.0); // TODO Does this invalid the cache line too much?
			}
			incrementNumeratorForGradientImpl<SparseIterator>(index); }
			break;
		default :
			// throw error
			exit(-1);
	}
}

void CyclicCoordinateDescent::incrementNumeratorForGradientImplHand(int index) {
	IndicatorIterator it(*hXI, index);
	const int n = hXI->getNumberOfEntries(index);
	int* indices = hXI->getCompressedColumnVector(index);
	for (int i = 0; i < n; ++i) {
		const int k = indices[i];
		numerPid[hPid[k]] += offsExpXBeta[k] * it.value();
	}
}

template <class IteratorType>
void CyclicCoordinateDescent::incrementNumeratorForGradientImpl(int index) {
	IteratorType it(*hXI, index);

	//cout << "incrementNumeratorForGradientImpl" << endl;
	//cout << "Index = " << index << endl;
	for (; it; ++it) {
		//cout << "it.index() = " << it.index() << endl;
		const int k = it.index();
		numerPid[hPid[k]] += offsExpXBeta[k] * it.value();
		//cout << "offsExpXBeta[k] = " << offsExpXBeta[k] << endl;
		//cout << "it.value = " << it.value() << endl;
		//cout << "hPid[k] = " << hPid[k] << endl;
		//cout << "numerPid[hPid[K]] = " << numerPid[hPid[k]] << endl;
		//cout << "denomPid[hPid[k]] = " << denomPid[hPid[k]] << endl;
		//cout << "hNevents[k] = " << hNEvents[hPid[k]] << endl;
		if (!IteratorType::isIndicator) {
			numerPid2[hPid[k]] += offsExpXBeta[k] * it.value() * it.value();
		}
	}
}

//void CyclicCoordinateDescent::computeRatio(int index) {
//	const int n = hXI->getNumberOfEntries(index);
//
//#ifdef BETTER_LOOPS
//	realTRS* t = t1;
//	const realTRS* end = t + N;
//	realTRS* num = numerPid;
//	realTRS* denom = denomPid;
//	for (; t != end; ++t, ++num, ++denom) {
//		*t = *num / *denom;
//	}
//#else
//	for (int i = 0; i < N; i++) {
//		t1[i] = numerPid[i] / denomPid[i];
//	}
//#endif
//}

void CyclicCoordinateDescent::computeRatiosForGradientAndHessian(int index) {
//	computeNumeratorForGradient(index);
//#ifndef MERGE_TRANSFORMATION
//	computeRatio(index);
//#endif
	cerr << "Error!" << endl;
	exit(-1);
}

double CyclicCoordinateDescent::ccdUpdateBeta(int index) {

	double delta;


	if (!sufficientStatisticsKnown) {
		cerr << "Error in state synchronization." << endl;
		exit(0);		
	}
	

	computeNumeratorForGradient(index);
	

	double g_d1;
	double g_d2;
					
	computeGradientAndHessian(index, &g_d1, &g_d2);

//	if (g_d2 <= 0.0) {
//		cerr << "Not positive-definite! Hessian = " << g_d2 << endl;
//		exit(-1);
//	}
		
	// Move into separate delegate-function (below)
	if (priorType == NORMAL) {
		
		delta = - (g_d1 + (hBeta[index] / sigma2Beta)) /
				  (g_d2 + (1.0 / sigma2Beta));
		
	} else {
					
		double neg_update = - (g_d1 - lambda) / g_d2;
		double pos_update = - (g_d1 + lambda) / g_d2;
		
		int signBetaIndex = sign(hBeta[index]);
		
		if (signBetaIndex == 0) {
						
			if (neg_update < 0) {
				delta = neg_update;
			} else if (pos_update > 0) {
				delta = pos_update;
			} else {
				delta = 0;
			}
		} else { // non-zero entry
			
			if (signBetaIndex < 0) {
				delta = neg_update;
			} else {
				delta = pos_update;			
			}
			
			if ( sign(hBeta[index] + delta) != signBetaIndex ) {
				delta = - hBeta[index];
			}			
		}
	}
	
	return delta;
}

template <class IteratorType>
void CyclicCoordinateDescent::axpy(realTRS* y, const realTRS alpha, const int index) {
	IteratorType it(*hXI, index);
	for (; it; ++it) {
		const int k = it.index();
		y[k] += alpha * it.value();
	}
}

void CyclicCoordinateDescent::computeXBeta(void) {
	// Separate function for benchmarking
//	cerr << "Computing X %*% beta" << endl;

	//	hXBeta = hX * hBeta;

	// Note: X is current stored in (sparse) column-major format, which is
	// inefficient for forming X\beta.
	// TODO Make row-major version of X

	// clear X\beta
	zeroVector(hXBeta, K);

	// Update one column at a time (poor cache locality)
	for (int j = 0; j < J; ++j) {
		const realTRS beta = hBeta[j];
		switch(hXI->getFormatType(j)) {
			case INDICATOR :
				axpy<IndicatorIterator>(hXBeta, beta, j);
				break;
			case DENSE :
				axpy<DenseIterator>(hXBeta, beta, j);
				break;
			case SPARSE :
				axpy<SparseIterator>(hXBeta, beta, j);
				break;
			default :
				// throw error
				exit(-1);
		}
	}

	xBetaKnown = true;
	sufficientStatisticsKnown = false;
}

template <class IteratorType>
void CyclicCoordinateDescent::updateXBetaImpl(realTRS realDelta, int index) {
	IteratorType it(*hXI, index);
	for (; it; ++it) {
		const int k = it.index();
		hXBeta[k] += realDelta * it.value();
		// Update denominators as well
		realTRS oldEntry = offsExpXBeta[k];
		realTRS newEntry = offsExpXBeta[k] = hOffs[k] * exp(hXBeta[k]);
		denomPid[hPid[k]] += (newEntry - oldEntry);
	}
}

void CyclicCoordinateDescent::updateXBetaImplHand(realTRS realDelta, int index) {


	realTRS* data = hXI->getDataVector(index);
	realTRS* xBeta = hXBeta;
	realTRS* offsEXB = offsExpXBeta;
	int* offs = hOffs;
	int* pid = hPid;

	for (int k = 0; k < K; k++) {
		*xBeta += realDelta * *data;
		realTRS oldEntry = *offsEXB;
		realTRS newEntry = *offsEXB = *offs * exp(*xBeta);
		denomPid[*pid] += (newEntry - oldEntry);
		data++; xBeta++; offsEXB++; offs++; pid++;
	}
}

void CyclicCoordinateDescent::updateXBeta(double delta, int index) {
	realTRS realDelta = static_cast<realTRS>(delta);
	hBeta[index] += realDelta;

	// Run-time dispatch to implementation depending on covariate FormatType
	switch(hXI->getFormatType(index)) {
		case INDICATOR :
			updateXBetaImpl<IndicatorIterator>(realDelta, index);
			break;
		case DENSE :
			updateXBetaImpl<DenseIterator>(realDelta, index);
//			updateXBetaImplHand(realDelta, index); // Oddly slower!
			break;
		case SPARSE :
			updateXBetaImpl<SparseIterator>(realDelta, index);
			break;
		default :
			// throw error
			exit(-1);
	}
}

void CyclicCoordinateDescent::updateSufficientStatistics(double delta, int index) {
	updateXBeta(delta, index);
	sufficientStatisticsKnown = true;
//	computeRemainingStatistics(false);
}

void CyclicCoordinateDescent::computeRemainingStatistics(bool allStats, int index) { // TODO Rename
	// Separate function for benchmarking

	if (allStats) {
		fillVector(denomPid, N, denomNullValue);
		for (int i = 0; i < K; i++) {
			offsExpXBeta[i] = hOffs[i] * exp(hXBeta[i]);
			denomPid[hPid[i]] += offsExpXBeta[i];
		}
//		cerr << "den[0] = " << denomPid[0] << endl;
//		exit(-1);
	}
	sufficientStatisticsKnown = true;
}

double CyclicCoordinateDescent::oneNorm(realTRS* vector, const int length) {
	double norm = 0;
	for (int i = 0; i < length; i++) {
		norm += abs(vector[i]);
	}
	return norm;
}

double CyclicCoordinateDescent::twoNormSquared(realTRS * vector, const int length) {
	double norm = 0;
	for (int i = 0; i < length; i++) {
		norm += vector[i] * vector[i];
	}
	return norm;
}

/**
 * Updating and convergence functions
 */

double CyclicCoordinateDescent::computeConvergenceCriterion(double newObjFxn, double oldObjFxn) {	
	// This is the stopping criterion that Ken Lange generally uses
	return abs(newObjFxn - oldObjFxn) / (abs(newObjFxn) + 1.0);
}

double CyclicCoordinateDescent::applyBounds(double inDelta, int index) {
	double delta = inDelta;
	if (delta < -hDelta[index]) {
		delta = -hDelta[index];
	} else if (delta > hDelta[index]) {
		delta = hDelta[index];
	}

	hDelta[index] = std::max(2.0 * abs(delta), 0.5 * hDelta[index]); //made std::max vs just max post Eigen use
	return delta;
}

/**
 * Utility functions
 */

void CyclicCoordinateDescent::computeXjEta(void) {

//	cerr << "YXj";
	for (int drug = 0; drug < J; drug++) {
		hXjEta[drug] = 0;
		GenericIterator it(*hXI, drug);

		if (useCrossValidation) {
			for (; it; ++it) {
				const int k = it.index();
				hXjEta[drug] += it.value() * hEta[k] * hWeights[k];
			}
		} else {
			for (; it; ++it) {
				const int k = it.index();
				hXjEta[drug] += it.value() * hEta[k];
			}
		}
//		cerr << " " << hXjEta[drug];
	}
//	cerr << endl;
//	exit(1);
}

template <class T>
void CyclicCoordinateDescent::printVector(T* vector, int length, ostream &os) {
	os << "(" << vector[0];
	for (int i = 1; i < length; i++) {
		os << ", " << vector[i];
	}
	os << ")";			
}

template <class T> 
T* CyclicCoordinateDescent::readVector(
		const char *fileName,
		int *length) {

	ifstream fin(fileName);
	T d;
	std::vector<T> v;
	
	while (fin >> d) {
		v.push_back(d);
	}
	
	T* ptr = (T*) malloc(sizeof(T) * v.size());
	for (int i = 0; i < v.size(); i++) {
		ptr[i] = v[i];
	}
		
	*length = v.size();
	return ptr;
}

void CyclicCoordinateDescent::testDimension(int givenValue, int trueValue, const char *parameterName) {
	if (givenValue != trueValue) {
		cerr << "Wrong dimension in " << parameterName << " vector." << endl;
		exit(-1);
	}
}	

inline int CyclicCoordinateDescent::sign(double x) {
	if (x == 0) {
		return 0;
	}
	if (x < 0) {
		return -1;
	}
	return 1;
}
