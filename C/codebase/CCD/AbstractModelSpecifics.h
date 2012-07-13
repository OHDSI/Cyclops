/*
 * AbstractModelSpecifics.h
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef ABSTRACTMODELSPECIFICS_H_
#define ABSTRACTMODELSPECIFICS_H_

#include "CompressedDataMatrix.h"

class AbstractModelSpecifics {
public:
	AbstractModelSpecifics();
	virtual ~AbstractModelSpecifics();

	void initialize(
			int iN,
			int iK,
			CompressedDataMatrix* iXI,
			real* iNumerPid,
			real* iNumerPid2,
			real* iDenomPid,
			int* iNEvents,
			real* iXjEta,
			std::vector<std::vector<int>* > *iSparseIndices,
			int* iPid,
			real* iOffsExpXBeta,
			real* iXBeta,
			int* iOffs
			) {
		hXI = iXI;
		N = iN;
		K = iK;
		numerPid = iNumerPid;
		numerPid2 = iNumerPid2;
		denomPid = iDenomPid;
		hNEvents = iNEvents;
		hXjEta = iXjEta;
		sparseIndices = iSparseIndices;

		hPid = iPid;
		offsExpXBeta = iOffsExpXBeta;

		hXBeta = iXBeta;
		hOffs = iOffs;
	}

	virtual void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian) = 0; // pure virtual

	virtual void computeNumeratorForGradient(int index) = 0; // pure virtual

	virtual void updateXBeta(real realDelta, int index) = 0; // pure virtual

	virtual void computeRemainingStatistics(void) = 0; // pure virtual

protected:

	template <class T>
	void fillVector(T* vector, const int length, const T& value) {
		for (int i = 0; i < length; i++) {
			vector[i] = value;
		}
	}

	template <class T>
	void zeroVector(T* vector, const int length) {
		fillVector(vector, length, T());
//		for (int i = 0; i < length; i++) {
//			vector[i] = 0;
//		}
	}

	// TODO Currently constructed in CyclicCoordinateDescent, but should be encapsulated here
	CompressedDataMatrix* hXI; // K-by-J-indicator matrix

	int* hOffs;  // K-vector
	int* hEta; // K-vector
	int* hNEvents; // K-vector
	int* hPid; // N-vector
	int** hXColumnRowIndicators; // J-vector

	real* hBeta;
	real* hXBeta;
	real* hXBetaSave;
	real* hDelta;

	int N; // Number of patients
	int K; // Number of exposure levels
	int J; // Number of drugs

	real* hWeights;

	real* expXBeta;
	real* offsExpXBeta;
	real* denomPid;
	real* numerPid;
	real* numerPid2;
	real* xOffsExpXBeta;
	real* hXjEta;

	std::vector<std::vector<int>* > *sparseIndices;
};

#endif /* ABSTRACTMODELSPECIFICS_H_ */
