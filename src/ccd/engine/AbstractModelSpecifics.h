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
#include <map>
#include <cstddef>

#include "Types.h"

namespace bsccs {

class CompressedDataMatrix;  // forward declaration
class CompressedDataColumn; // forward declaration
class ModelData; // forward declaration
enum class ModelType; // forward declaration

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
			real* iOffs,
			real* iBeta,
			real* iY);

	virtual void setWeights(real* inWeights, bool useCrossValidation) = 0; // pure virtual

	virtual void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian, bool useWeights) = 0; // pure virtual

	virtual void computeNumeratorForGradient(int index) = 0; // pure virtual

	virtual void computeFisherInformation(int indexOne, int indexTwo,
			double *oinfo, bool useWeights) = 0; // pure virtual

	virtual void updateXBeta(real realDelta, int index, bool useWeights) = 0; // pure virtual

	virtual void computeRemainingStatistics(bool useWeights) = 0; // pure virtual

	virtual void computeFixedTermsInLogLikelihood(bool useCrossValidation) = 0; // pure virtual

	virtual void computeFixedTermsInGradientAndHessian(bool useCrossValidation) = 0; // pure virtual

	virtual double getLogLikelihood(bool useCrossValidation) = 0; // pure virtual

	virtual double getPredictiveLogLikelihood(real* weights) = 0; // pure virtual

    virtual void getPredictiveEstimates(real* y, real* weights) = 0; // pure virtual

    virtual void makeDirty();

//	virtual void sortPid(bool useCrossValidation) = 0; // pure virtual

	static AbstractModelSpecifics* factory(const ModelType modelType, ModelData* modelData); // TODO return shared_ptr

protected:

	virtual bool allocateXjY(void) = 0; // pure virtual

	virtual bool allocateXjX(void) = 0; // pure virtual
	
	virtual bool initializeAccumulationVectors(void) = 0; // pure virtual

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
	
	std::vector<int> accReset;

	// TODO Currently constructed in CyclicCoordinateDescent, but should be encapsulated here
	CompressedDataMatrix* hXI; // K-by-J-indicator matrix

	real* hOffs;  // K-vector
	real* hY; // K-vector
	real* hZ; // K-vector

	int* hPid; // K-vector
	int** hXColumnRowIndicators; // J-vector

//	real* hBeta;
	real* hXBeta;
	real* hXBetaSave;
	real* hDelta;

	size_t N; // Number of patients
	size_t K; // Number of exposure levels
	size_t J; // Number of drugs

	real* expXBeta;
	real* offsExpXBeta;
	real* denomPid;
	real* numerPid;
	real* numerPid2;
	real* xOffsExpXBeta;
	real* hXjY;
	real* hXjX;
	real logLikelihoodFixedTerm;

	std::vector<std::vector<int>* > *sparseIndices;

	typedef std::map<int, std::vector<real> > HessianMap;
	HessianMap hessianCrossTerms;

	typedef std::map<int, CompressedDataColumn* > HessianSparseMap;
	HessianSparseMap hessianSparseCrossTerms;
	
	typedef std::vector<int> TimeTie;
	std::vector<TimeTie> ties;
	
	std::vector<int> beginTies;
	std::vector<int> endTies;
};

typedef bsccs::shared_ptr<AbstractModelSpecifics> ModelSpecificsPtr;

} // namespace

#endif /* ABSTRACTMODELSPECIFICS_H_ */
