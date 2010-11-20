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

#include "CyclicCoordinateDescent.h"
#include "InputReader.h"

#define PI	3.14159265358979323851280895940618620443274267017841339111328125

using namespace std;


void compareIntVector(int* vec0, int* vec1, int dim, const char* name) {
	for (int i = 0; i < dim; i++) {
		if (vec0[i] != vec1[i]) {
			cerr << "Error at " << name << "[" << i << "]: ";
			cerr << vec0[i] << " != " << vec1[i] << endl;
			exit(0);
		}
	}
}

//CyclicCoordinateDescent::CyclicCoordinateDescent(
//		const char* fileNameX,
//		const char* fileNameEta,
//		const char* fileNameOffs,
//		const char* fileNameNEvents,
//		const char* fileNamePid
//	) {
//
//	hXI = new CompressedIndicatorMatrix(fileNameX);
//
//	K = hXI->getNumberOfRows();
//	J = hXI->getNumberOfColumns();
//
//	conditionId = "NA";
//
//	int lOffs;
//    hOffs = readVector<int>(fileNameOffs, &lOffs);
//
//    int lEta;
//    hEta = readVector<int>(fileNameEta, &lEta);
//
//    int lNEvents;
//    hNEvents = readVector<int>(fileNameNEvents, &lNEvents);
//
//    int lPid;
//    hPid = readVector<int>(fileNamePid, &lPid);
//
//    testDimension(lOffs, K, "hOffs");
//    testDimension(lEta, K, "hEta");
//    testDimension(lPid, K, "hPid");
//
//    N = lNEvents;
//
//    hasLog = false;
//
//    init();
//}

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
	
	free(hXjEta);
	free(offsExpXBeta);
	free(xOffsExpXBeta);
	free(denomPid);
	free(numerPid);
	free(t1);
	
	if (hWeights) {
		free(hWeights);
	}
}

string CyclicCoordinateDescent::getPriorInfo() {
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
}

void CyclicCoordinateDescent::resetBounds() {
	for (int j = 0; j < J; j++) {
		hDelta[j] = 2.0;
	}
}

void CyclicCoordinateDescent::init() {
	
	// Set parameters and statistics space
	hDelta = (real*) malloc(J * sizeof(real));
//	for (int j = 0; j < J; j++) {
//		hDelta[j] = 2.0;
//	}

	hBeta = (real*) calloc(J, sizeof(real)); // Fixed starting state
	hXBeta = (real*) calloc(K, sizeof(real));
	hXBetaSave = (real*) calloc(K, sizeof(real));

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
	offsExpXBeta = (real*) malloc(sizeof(real) * K);
	xOffsExpXBeta = (real*) malloc(sizeof(real) * K);
	denomPid = (real*) malloc(sizeof(real) * N);
	numerPid = (real*) malloc(sizeof(real) * N);
	t1 = (real*) malloc(sizeof(real) * N);
	hNEvents = (int*) malloc(sizeof(int) * N);
	hXjEta = (real*) malloc(sizeof(real) * J);
	hWeights = NULL;

	useCrossValidation = false;
	validWeights = false;
	sufficientStatisticsKnown = false;

#ifdef DEBUG	
	cerr << "Number of patients = " << N << endl;
	cerr << "Number of exposure levels = " << K << endl;
	cerr << "Number of drugs = " << J << endl;	
#endif          
}

void CyclicCoordinateDescent::computeNEvents() {
	zeroVector(hNEvents, N);
	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			hNEvents[hPid[i]] += hEta[i] * int(hWeights[i]); // TODO Consider using only integer weights
		}
	} else {
		for (int i = 0; i < K; i++) {
			hNEvents[hPid[i]] += hEta[i];
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

void CyclicCoordinateDescent::logResults(const char* fileName) {

	ofstream outLog(fileName);
	if (!outLog) {
		cerr << "Unable to open log file: " << fileName << endl;
		exit(-1);
	}

	InputReader* reader = dynamic_cast<InputReader*>(hXI);
	map<int, DrugIdType> drugMap = reader->getDrugNameMap();

	string sep(","); // TODO Make option

	for (int i = 0; i < J; i++) {
		outLog << conditionId << sep <<
//		i << sep <<
		drugMap[i] << sep << hBeta[i] << endl;
	}
	outLog.close();
}

double CyclicCoordinateDescent::getPredictiveLogLikelihood(real* weights) {

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics();
	}

	getDenominators();

	double logLikelihood = 0;

	for (int i = 0; i < K; i++) {
		logLikelihood += hEta[i] * weights[i] * (hXBeta[i] - log(denomPid[hPid[i]]));
	}

	return logLikelihood;
}

real CyclicCoordinateDescent::getBeta(int i) {
	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics();
	}
	return hBeta[i];
}

double CyclicCoordinateDescent::getLogLikelihood(void) {

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics();
	}

	getDenominators();

	double logLikelihood = 0;

	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			logLikelihood += hEta[i] * hXBeta[i] * hWeights[i];
		}
	} else {
		for (int i = 0; i < K; i++) {
			logLikelihood += hEta[i] * hXBeta[i];
		}
	}

	for (int i = 0; i < N; i++) {
		logLikelihood -= hNEvents[i] * log(denomPid[i]);
	}

	return logLikelihood;
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

void CyclicCoordinateDescent::setPriorType(int iPriorType) {
	if (iPriorType != LAPLACE && iPriorType != NORMAL) {
		cerr << "Unknown prior type" << endl;
		exit(-1);
	}
	priorType = iPriorType;
}

void CyclicCoordinateDescent::setWeights(real* iWeights) {

	if (iWeights == NULL) {
		std::cerr << "Turning off weights!" << std::endl;
		// Turn off weights
		useCrossValidation = false;
		validWeights = false;
		sufficientStatisticsKnown = false;
		return;
	}

	if (hWeights == NULL) {
		hWeights = (real*) malloc(sizeof(real) * K);
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
	double criterion = 0;
	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			criterion += hXBeta[i] * hEta[i] * hWeights[i];
		}
	} else {
		for (int i = 0; i < K; i++) {
			criterion += hXBeta[i] * hEta[i];
		}
	}
	return criterion;
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
	memcpy(hXBetaSave, hXBeta, K * sizeof(real));
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

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics();
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
			if (delta != 0) {
				sufficientStatisticsKnown = false;
				updateSufficientStatistics(delta, index);
			}
			
			if ((index+1) % 100 == 0) {
				cout << "Finished variable " << (index+1) << endl;
			}
			
		}
		
		iteration++;
//		bool checkConvergence = (iteration % J == 0 || iteration == maxIterations);
		bool checkConvergence = true; // Check after each complete cycle

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
}

/**
 * Computationally heavy functions
 */

void CyclicCoordinateDescent::computeGradientAndHession(int index, double *ogradient,
		double *ohessian) {
	double gradient = 0;
	double hessian = 0;
	for (int i = 0; i < N; i++) {
		gradient += hNEvents[i] * t1[i];
		hessian += hNEvents[i] * t1[i] * (1.0 - t1[i]);
	}

	gradient -= hXjEta[index];
	*ogradient = gradient;
	*ohessian = hessian;
}

void CyclicCoordinateDescent::computeRatiosForGradientAndHessian(int index) {
	
	zeroVector(numerPid, N);
	
	const int* indicators = hXI->getCompressedColumnVector(index);
	const int n = hXI->getNumberOfEntries(index);
	for (int i = 0; i < n; i++) { // Loop through non-zero entries only
		const int k = indicators[i];
		numerPid[hPid[k]] += offsExpXBeta[k];
	}
	
	for (int i = 0; i < N; i++) {
		t1[i] = numerPid[i] / denomPid[i];
	}
}

double CyclicCoordinateDescent::ccdUpdateBeta(int index) {

	double delta;

	if (!sufficientStatisticsKnown) {
		cerr << "Error in state synchronization." << endl;
		exit(0);		
	}
	
	computeRatiosForGradientAndHessian(index);
	
	double g_d1;
	double g_d2;
					
	computeGradientAndHession(index, &g_d1, &g_d2);
		
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

void CyclicCoordinateDescent::computeXBeta(void) {
	// Separate function for benchmarking
	cerr << "Computing X %*% beta" << endl;
	cerr << "Not yet implemented." << endl;
	exit(-1);
//	hXBeta = hX * hBeta;
}

void CyclicCoordinateDescent::updateXBeta(double delta, int index) {
	// Separate function for benchmarking
	hBeta[index] += delta;

	const int* indicators = hXI->getCompressedColumnVector(index);
	const int n = hXI->getNumberOfEntries(index);
	for (int i = 0; i < n; i++) { // Loop through non-zero entries only
		const int k = indicators[i];
		hXBeta[k] += delta;
	}
}

void CyclicCoordinateDescent::updateSufficientStatistics(double delta, int index) {
	updateXBeta(delta, index);
	computeRemainingStatistics();
}

void CyclicCoordinateDescent::computeRemainingStatistics(void) {
	// Separate function for benchmarking	
	zeroVector(denomPid, N);
	for (int i = 0; i < K; i++) {
		offsExpXBeta[i] = hOffs[i] * exp(hXBeta[i]);
		denomPid[hPid[i]] += offsExpXBeta[i];
	}

	sufficientStatisticsKnown = true;
}

double CyclicCoordinateDescent::oneNorm(real* vector, const int length) {
	double norm = 0;
	for (int i = 0; i < length; i++) {
		norm += abs(vector[i]);
	}
	return norm;
}

double CyclicCoordinateDescent::twoNormSquared(real * vector, const int length) {
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

	hDelta[index] = max(2.0 * abs(delta), 0.5 * hDelta[index]);
	return delta;
}

/**
 * Utility functions
 */

void CyclicCoordinateDescent::computeXjEta(void) {

	for (int drug = 0; drug < J; drug++) {
		hXjEta[drug] = 0;
		const int* indicators = hXI->getCompressedColumnVector(drug);
		const int n = hXI->getNumberOfEntries(drug);

		if (useCrossValidation) {
			for (int i = 0; i < n; i++) { // Loop through non-zero entries only
				const int k = indicators[i];
				hXjEta[drug] += hEta[k] * hWeights[k];
			}
		} else {
			for (int i = 0; i < n; i++) { // Loop through non-zero entries only
				const int k = indicators[i];
				hXjEta[drug] += hEta[k];
			}
		}
	}
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
