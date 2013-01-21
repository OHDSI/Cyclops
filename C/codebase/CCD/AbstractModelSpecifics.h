/*
 * AbstractModelSpecifics.h
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef ABSTRACTMODELSPECIFICS_H_
#define ABSTRACTMODELSPECIFICS_H_

#include <vector>
#include <cmath>

class CompressedDataMatrix;  // forward declaration
class ModelData; // forward declaration

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif

//#define DEBUG_COX // Uncomment to get output for Cox model
//#define DEBUG_POISSON

class AbstractModelSpecifics {
public:
//	AbstractModelSpecifics(
//			const std::vector<real>& y,
//			const std::vector<real>& z);

	AbstractModelSpecifics(const ModelData& intput);

	virtual ~AbstractModelSpecifics();

	void initialize(
			int iN,
			int iK,
			int iJ,
			CompressedDataMatrix* iXI,
			real* iNumerPid,
			real* iNumerPid2,
			real* iDenomPid,
			real* iXjY,
			std::vector<std::vector<int>* >* iSparseIndices,
			int* iPid,
			real* iOffsExpXBeta,
			real* iXBeta,
			int* iOffs,
			real* iBeta,
			real* iY);

	virtual void setWeights(real* inWeights, bool useCrossValidation) = 0; // pure virtual

	virtual void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian, bool useWeights) = 0; // pure virtual

	virtual void computeNumeratorForGradient(int index) = 0; // pure virtual

	virtual void updateXBeta(real realDelta, int index, bool useCrossValidation) = 0; // pure virtual

	virtual void computeRemainingStatistics(bool useCrossValidation) = 0; // pure virtual

	virtual void computeFixedTermsInGradientAndHessian(bool useCrossValidation) = 0; // pure virtual

	virtual double getLogLikelihood(bool useCrossValidation) = 0; // pure virtual

	virtual void getRelativeRisks(real* risks) = 0; // pure virtual

	virtual void getCumulativeHazards(real* hazards) = 0; // pure virtual

	virtual double getPredictiveLogLikelihood(real* weights) = 0; // pure virtual

    virtual void getPredictiveEstimates(real* y, real* weights) = 0; // pure virtual

//	virtual void sortPid(bool useCrossValidation) = 0; // pure virtual

protected:

	virtual bool allocateXjY(void) = 0; // pure virtual

	virtual bool allocateXjX(void) = 0; // pure virtual

	template <class T>
	void fillVector(T* vector, const int length, const T& value) {
		for (int i = 0; i < length; i++) {
			vector[i] = value;
		}
	}

	template <class T>
	void zeroVector(T* vector, const int length) {
		fillVector(vector, length, T());
	}

	const std::vector<real>& oY;
	const std::vector<real>& oZ;
	const std::vector<int>& oPid;

	std::vector<real> accDenomPid;
	std::vector<real> accNumerPid;
	std::vector<real> accNumerPid2;

	// TODO Currently constructed in CyclicCoordinateDescent, but should be encapsulated here
	CompressedDataMatrix* hXI; // K-by-J-indicator matrix

	int* hOffs;  // K-vector
	real* hY; // K-vector
	real* hZ; // K-vector

	int* hPid; // K-vector
	int** hXColumnRowIndicators; // J-vector

	real* hBeta;
	real* hXBeta;
	real* hXBetaSave;
	real* hDelta;

	int N; // Number of patients
	int K; // Number of exposure levels
	int J; // Number of drugs

	real* expXBeta;
	real* offsExpXBeta;
	real* denomPid;
	real* numerPid;
	real* numerPid2;
	real* xOffsExpXBeta;
	real* hXjY;
	real* hXjX;

	std::vector<std::vector<int>* > *sparseIndices;
};

#endif /* ABSTRACTMODELSPECIFICS_H_ */
