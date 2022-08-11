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
#include "ModelData.h"

namespace bsccs {

typedef std::pair<double, double> GradientHessian;

//class AbstractModelData; // forward declaration
enum class ModelType; // forward declaration

// #ifdef DOUBLE_PRECISION
// 	typedef double real;
// #else
// 	typedef float real;
// #endif

// #define DEBUG_COX // Uncomment to get output for Cox model
// #define DEBUG_COX_MIN
// #define DEBUG_POISSON

class AbstractModelSpecifics {
public:
//	AbstractModelSpecifics(
//			const std::vector<real>& y,
//			const std::vector<real>& z);

	AbstractModelSpecifics(const AbstractModelData& intput);

	virtual ~AbstractModelSpecifics();

	virtual void initialize(
			int iN,
			int iK,
			int iJ,
			const void* iXi,
			// const CompressedDataMatrix<double>* iXI, // TODO Change to const&
			double* iNumerPid,
			double* iNumerPid2,
			double* iDenomPid,
			double* iXjY,
			std::vector<std::vector<int>* >* iSparseIndices,
			const int* iPid,
			double* iOffsExpXBeta,
			double* iXBeta,
			double* iOffs,
			double* iBeta,
			const double* iY) = 0; // pure virtual

	//virtual void setWeights(double* inWeights, bool useCrossValidation) = 0; // pure virtual
	virtual void setWeights(double* inWeights, double* cenWeights, bool useCrossValidation) = 0; // pure virtual

	virtual void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian, bool useWeights) = 0; // pure virtual

	virtual void computeThirdDerivative(int index, double *othird, bool useWeights) = 0; // pure virtual

	virtual void computeMMGradientAndHessian(std::vector<GradientHessian>& gh, const std::vector<bool>& fixBeta, bool useWeights) = 0; // pure virtual

	virtual void computeNumeratorForGradient(int index, bool useWeights) = 0; // pure virtual

	virtual void computeFisherInformation(int indexOne, int indexTwo,
			double *oinfo, bool useWeights) = 0; // pure virtual

	virtual void updateXBeta(double realDelta, int index, bool useWeights) = 0; // pure virtual

	virtual void computeXBeta(double* beta, bool useWeights) = 0; // pure virtual

	virtual void computeRemainingStatistics(bool useWeights) = 0; // pure virtual

	virtual void computeFixedTermsInLogLikelihood(bool useCrossValidation) = 0; // pure virtual

	virtual void computeFixedTermsInGradientAndHessian(bool useCrossValidation) = 0; // pure virtual

	virtual double getLogLikelihood(bool useCrossValidation) = 0; // pure virtual

	virtual double getPredictiveLogLikelihood(double* weights) = 0; // pure virtual

    virtual void getPredictiveEstimates(double* y, double* weights) = 0; // pure virtual

    virtual double getGradientObjective(bool useCrossValidation) = 0; // pure virtual

    virtual void makeDirty();

    virtual void printTiming() = 0; // pure virtual

//	virtual void sortPid(bool useCrossValidation) = 0; // pure virtual

//	static bsccs::shared_ptr<AbstractModelSpecifics> factory(const ModelType modelType, const ModelData& modelData);

	virtual AbstractModelSpecifics* clone() const = 0; // pure virtual

// 	static bsccs::shared_ptr<AbstractModelSpecifics> factory(const ModelType modelType,
//                                                            const ModelData& modelData,
//                                                            const DeviceType deviceType);

	static AbstractModelSpecifics* factory(const ModelType modelType,
                                           const AbstractModelData& modelData,
                                           const DeviceType deviceType,
                                           const std::string& deviceName);

	virtual const std::vector<double> getXBeta() = 0;

	virtual const std::vector<double> getXBetaSave() = 0;

	virtual void saveXBeta() = 0;

	virtual void zeroXBeta() = 0;

	virtual void axpyXBeta(const double beta, const int j) = 0;

protected:

//     template <class Engine>
//     static AbstractModelSpecifics* modelFactory(const ModelType modelType,
//                                            const ModelData& modelData);

    template <class Model, typename RealType>
    static AbstractModelSpecifics* deviceFactory(const ModelData<RealType>& modelData,
                                                 const DeviceType deviceType,
                                                 const std::string& deviceName);

    virtual void deviceInitialization() = 0;

	int getAlignedLength(int N);

	template <typename RealType>
	void setPidForAccumulation(const RealType *weights);

	void setupSparseIndices(const int max);

	virtual bool allocateXjY(void) = 0; // pure virtual

	virtual bool allocateXjX(void) = 0; // pure virtual

	virtual bool initializeAccumulationVectors(void) = 0; // pure virtual

	virtual bool hasResetableAccumulators(void) = 0; // pure virtual

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

protected:
//	const AbstractModelData& modelData;

// 	const std::vector<real>& oY;
// 	const std::vector<real>& oZ;
// 	const std::vector<int>& oPid;

	// TODO Change to const& (is never nullptr)
// 	const CompressedDataMatrix* hXI; // K-by-J-indicator matrix

	// RealVector accDenomPid;
	// RealVector accNumerPid;
	// RealVector accNumerPid2;

	std::vector<int> accReset;

// 	const std::vector<real>& hY;
// 	const std::vector<real>& hOffs;
// // 	const std::vector<int>& hPid;

	const std::vector<int>& hPidOriginal;
	int* hPid;
	size_t hPidSize;
	std::vector<int> hPidInternal;

	// RealVector hXBeta; // TODO Delegate to ModelSpecifics
	// RealVector hXBetaSave; // Delegate


	size_t N; // Number of patients
	size_t K; // Number of exposure levels
	size_t J; // Number of drugs

	// RealVector offsExpXBeta;
	// RealVector denomPid;
	// RealVector numerPid;
	// RealVector numerPid2;
	//
	// RealVector hXjY;
	// RealVector hXjX;
	// RealType logLikelihoodFixedTerm;

	typedef std::vector<int> IndexVector;
	typedef bsccs::shared_ptr<IndexVector> IndexVectorPtr;

	std::vector<IndexVectorPtr> sparseIndices; // TODO in c++11, are pointers necessary?

	typedef std::map<int, std::vector<double> > HessianMap;
	HessianMap hessianCrossTerms;

    // typedef bsccs::shared_ptr<CompressedDataColumn> CDCPtr;
	// typedef std::map<int, CDCPtr> HessianSparseMap;
	// HessianSparseMap hessianSparseCrossTerms;

	typedef std::vector<int> TimeTie;
	std::vector<TimeTie> ties;

	std::vector<int> beginTies;
	std::vector<int> endTies;

	// typedef bsccs::shared_ptr<CompressedDataMatrix> CdmPtr;

	// CdmPtr hXt;
	const MmBoundType boundType;
	std::vector<double> curvature;
};

typedef bsccs::shared_ptr<AbstractModelSpecifics> ModelSpecificsPtr;

} // namespace

#endif /* ABSTRACTMODELSPECIFICS_H_ */
