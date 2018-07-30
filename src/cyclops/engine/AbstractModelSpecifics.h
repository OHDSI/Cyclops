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
#include "priors/CovariatePrior.h"

namespace bsccs {

typedef std::pair<double, double> GradientHessian;

class CompressedDataMatrix;  // forward declaration
class CompressedDataColumn; // forward declaration
class ModelData; // forward declaration
enum class ModelType; // forward declaration

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif

// #define DEBUG_COX // Uncomment to get output for Cox model
// #define DEBUG_COX_MIN
// #define DEBUG_POISSON

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
			const CompressedDataMatrix* iXI, // TODO Change to const&
			real* iNumerPid,
			real* iNumerPid2,
			real* iDenomPid,
			real* iXjY,
			std::vector<std::vector<int>* >* iSparseIndices,
			const int* iPid,
			real* iOffsExpXBeta,
			real* iXBeta,
			real* iOffs,
			real* iBeta,
			const real* iY);

	virtual void setAlgorithmType(AlgorithmType alg);

	virtual void setLogSum(bool logSum);

	virtual void setWeights(double* inWeights, bool useCrossValidation) = 0; // pure virtual

	virtual void computeGradientAndHessian(int index, double *ogradient,
			double *ohessian, bool useWeights) = 0; // pure virtual

	virtual void computeMMGradientAndHessian(std::vector<GradientHessian>& gh, const std::vector<bool>& fixBeta, bool useWeights) = 0; // pure virtual

	virtual void computeNumeratorForGradient(int index) = 0; // pure virtual

	virtual void computeFisherInformation(int indexOne, int indexTwo,
			double *oinfo, bool useWeights) = 0; // pure virtual

	virtual void updateXBeta(real realDelta, int index, bool useWeights) = 0; // pure virtual

	virtual void updateAllXBeta(std::vector<double>& allDelta, bool useWeights) = 0;

	virtual void updateXBeta(std::vector<double>& allDelta, std::vector<std::pair<int,int>>& updateIndices, bool useWeights) = 0; // pure virtual

	virtual void updateXBetaMM(std::vector<double>& allDelta, std::vector<std::pair<int,int>>& updateIndices, bool useWeights) = 0; // pure virtual

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

    virtual bool isGPU() = 0;

//	virtual void sortPid(bool useCrossValidation) = 0; // pure virtual

//	static bsccs::shared_ptr<AbstractModelSpecifics> factory(const ModelType modelType, const ModelData& modelData);

	virtual AbstractModelSpecifics* clone() const = 0; // pure virtual

// 	static bsccs::shared_ptr<AbstractModelSpecifics> factory(const ModelType modelType,
//                                                            const ModelData& modelData,
//                                                            const DeviceType deviceType);

	static AbstractModelSpecifics* factory(const ModelType modelType,
                                           const ModelData& modelData,
                                           const DeviceType deviceType,
                                           const std::string& deviceName);

	virtual const RealVector& getXBeta() = 0;

	virtual const RealVector& getXBetaSave() = 0;

	virtual void saveXBeta() = 0;

	virtual void zeroXBeta() = 0;

	virtual void axpyXBeta(const double beta, const int j) = 0;

	//syncCV
	virtual void turnOnSyncCV(int foldToCompute) = 0;

	virtual void turnOffSyncCV() = 0;

	virtual void axpyXBeta(const double beta, const int j, const int cvIndex) = 0;

	virtual void setWeights(double* inWeights, bool useCrossValidation, int index) = 0; // pure virtual

    virtual std::vector<double> getGradientObjectives() = 0; // pure virtual

    virtual std::vector<double> getLogLikelihoods(bool useCrossValidation) = 0; // pure virtual

	virtual void computeNumeratorForGradient(int index, int cvIndex) = 0; // pure virtual

	virtual void computeGradientAndHessian(int index, double* gradient,
			double* hessian, bool useWeights, int cvIndex) = 0;

	virtual void computeRemainingStatistics(bool useWeights, int cvIndex) = 0; // pure virtual

	virtual void computeRemainingStatistics(bool useWeights, std::vector<bool>& fixBeta) = 0; // pure virtual

	virtual void updateXBeta(real realDelta, int index, bool useWeights, int cvIndex) = 0; // pure virtual

	virtual void updateXBeta(std::vector<double>& realDelta, int index, bool useWeights) = 0; // pure virtual

	virtual void printStuff() = 0;

	virtual void updateAllXBeta(std::vector<double>& allDelta, bool useWeights, int cvIndex) = 0;

	virtual void computeGradientAndHessian(int index, std::vector<priors::GradientHessian>& ghList, std::vector<bool>& fixBetaTemp, bool useWeights) = 0;

	virtual void computeMMGradientAndHessian(std::vector<GradientHessian>& gh, const std::vector<std::pair<int,int>>& updateIndices) = 0; // pure virtual

	virtual void computeGradientAndHessian(std::vector<GradientHessian>& gh, const std::vector<std::pair<int,int>>& updateIndices) = 0; // pure virtual

	virtual void copyXBetaVec() = 0;
	std::vector<RealVector> accDenomPidPool;
	std::vector<RealVector> accNumerPidPool;
	std::vector<RealVector> accNumerPid2Pool;
	std::vector<IntVector> accResetPool;

	std::vector<int*> hPidPool;
	std::vector<std::vector<int>> hPidInternalPool;

	std::vector<RealVector> hXBetaPool;
	std::vector<RealVector> offsExpXBetaPool;

	std::vector<RealVector> denomPidPool;
	std::vector<RealVector> numerPidPool;
	std::vector<RealVector> numerPid2Pool;

	std::vector<RealVector> hXjYPool;
	std::vector<RealVector> hXjXPool;
	std::vector<real> logLikelihoodFixedTermPool;
	/*

	RealVector accDenomPidPool;
	RealVector accNumerPidPool;
	RealVector accNumerPid2Pool;
	IntVector accResetPool;

	std::vector<int> hPidPool;
	std::vector<int> hPidInternalPool;

	RealVector hXBetaPool;
	RealVector offsExpXBetaPool;

	RealVector denomPidPool;
	RealVector numerPidPool;
	RealVector numerPid2Pool;

	RealVector hXjYPool;
	RealVector hXjXPool;
	std::vector<real> logLikelihoodFixedTermPool;
*/



	bool syncCV = false;
	int syncCVFolds;

	virtual void setBounds(double initialBound) {};

	virtual void setPriorTypes(std::vector<int>& typeList) {};

	virtual void setPriorParams(std::vector<double>& paramList) {};

	virtual void updateDoneFolds(std::vector<bool>& donePool) {};

	virtual void runCCDIndex() {};

	virtual void runMM() {};

	virtual void resetBeta() {};

	virtual std::vector<double> getBeta() {
		std::vector<double> blah;
		blah.push_back(0);
		return(blah);
	};

	virtual double getPredictiveLogLikelihood(double* weights, int cvIndex) = 0; // pure virtual

	virtual void computeAllGradientAndHessian(std::vector<GradientHessian>& gh, const std::vector<bool>& fixBeta, bool useWeights) = 0; // pure virtual

protected:

//     template <class Engine>
//     static AbstractModelSpecifics* modelFactory(const ModelType modelType,
//                                            const ModelData& modelData);

    template <class Model, typename RealType>
    static AbstractModelSpecifics* deviceFactory(const ModelData& modelData,
                                                 const DeviceType deviceType,
                                                 const std::string& deviceName);

    template <class Model, typename RealType, class ModelG>
    static AbstractModelSpecifics* deviceFactory(const ModelData& modelData,
                                                 const DeviceType deviceType,
                                                 const std::string& deviceName);

    virtual void deviceInitialization();

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

	//syncCV
	template <typename RealType>
	void setPidForAccumulation(const RealType *weights, int cvIndex);

	void setupSparseIndices(const int max, int cvIndex);

protected:
	const ModelData& modelData;

// 	const std::vector<real>& oY;
// 	const std::vector<real>& oZ;
// 	const std::vector<int>& oPid;

	// TODO Change to const& (is never nullptr)
// 	const CompressedDataMatrix* hXI; // K-by-J-indicator matrix

	RealVector accDenomPid;
	RealVector accNumerPid;
	RealVector accNumerPid2;

	IntVector accReset;

	const std::vector<real>& hY;
	const std::vector<real>& hOffs;
// 	const std::vector<int>& hPid;

// 	real* hY; // K-vector
//	real* hZ; // K-vector
// 	real* hOffs;  // K-vector

	const std::vector<int>& hPidOriginal;
	int* hPid;
	std::vector<int> hPidInternal;

//	int** hXColumnRowIndicators; // J-vector

//	real* hBeta;
// 	real* hXBeta;
// 	real* hXBetaSave;

	AlgorithmType algorithmType;

	bool useLogSum;

	//RealVector hBeta;
	RealVector hXBeta; // TODO Delegate to ModelSpecifics
	RealVector hXBetaSave; // Delegate
	RealVector norm;

	std::vector<RealVector> normPool;

//	real* hDelta;

	size_t N; // Number of patients
	size_t K; // Number of exposure levels
	size_t J; // Number of drugs

//	real* expXBeta;
//	real* offsExpXBeta;
	RealVector offsExpXBeta;

// 	RealVector numerDenomPidCache;
// 	real* denomPid; // all nested with a single cache
// 	real* numerPid;
// 	real* numerPid2;

	RealVector denomPid;
	RealVector numerPid;
	RealVector numerPid2;


//	real* xOffsExpXBeta;
//	real* hXjY;
	RealVector hXjY;
	RealVector hXjX;
	real logLikelihoodFixedTerm;

	typedef std::vector<int> IndexVector;
	typedef bsccs::shared_ptr<IndexVector> IndexVectorPtr;

	std::vector<IndexVectorPtr> sparseIndices; // TODO in c++11, are pointers necessary?

	typedef std::map<int, std::vector<real> > HessianMap;
	HessianMap hessianCrossTerms;

    typedef bsccs::shared_ptr<CompressedDataColumn> CDCPtr;
	typedef std::map<int, CDCPtr> HessianSparseMap;
	HessianSparseMap hessianSparseCrossTerms;

	typedef std::vector<int> TimeTie;
	std::vector<TimeTie> ties;

	std::vector<int> beginTies;
	std::vector<int> endTies;

	typedef bsccs::shared_ptr<CompressedDataMatrix> CdmPtr;

	CdmPtr hXt;
	const MmBoundType boundType;
	std::vector<double> curvature;

	std::vector<std::vector<IndexVectorPtr>> sparseIndicesPool; // TODO in c++11, are pointers necessary?

};

typedef bsccs::shared_ptr<AbstractModelSpecifics> ModelSpecificsPtr;

} // namespace

#endif /* ABSTRACTMODELSPECIFICS_H_ */
