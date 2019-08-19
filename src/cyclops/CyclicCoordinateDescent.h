/*
 * CyclicCoordinateDescent.h
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#ifndef CYCLICCOORDINATEDESCENT_H_
#define CYCLICCOORDINATEDESCENT_H_

#include "CcdInterface.h"
#include "CompressedDataMatrix.h"
#include "ModelData.h"
#include "engine/AbstractModelSpecifics.h"
#include "priors/JointPrior.h"
#include "io/ProgressLogger.h"

#include <Eigen/Dense>
#include <deque>

#include "Types.h"

namespace bsccs {

// TODO Remove 'using' from headers
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::ofstream;
using std::string;

//#define DEBUG

#define TEST_SPARSE // New sparse updates are great
//#define TEST_ROW_INDEX
#define BETTER_LOOPS
#define MERGE_TRANSFORMATION
#define NEW_NUMERATOR
#define SPARSE_PRODUCT

#define USE_ITER
//#define NO_FUSE

class CyclicCoordinateDescent {

public:

	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

	CyclicCoordinateDescent(
			const AbstractModelData& modelData,
			AbstractModelSpecifics& specifics,
			priors::JointPriorPtr prior,
			loggers::ProgressLoggerPtr logger,
			loggers::ErrorHandlerPtr error
		);

	// CyclicCoordinateDescent(
	// 		int inN,
	// 		CompressedDataMatrix* inX,
	// 		int* inEta,
	// 		int* inOffs,
	// 		int* inNEvents,
	// 		int* inPid
	// 	);

	CyclicCoordinateDescent* clone();

	void logResults(const char* fileName, bool withASE);

	virtual ~CyclicCoordinateDescent();

	double getLogLikelihood(void);

	//double getPredictiveLogLikelihood(double* weights);

	double getNewPredictiveLogLikelihood(double* weights);

	void getPredictiveEstimates(double* y, double* weights) const;

	double getLogPrior(void);

	virtual double getObjectiveFunction(int convergenceType);

	double getBeta(int i);

	int getBetaSize(void);

    bool getIsRegularized(int i) const;

	int getPredictionSize(void) const;

	bool getFixedBeta(int i);

	void setFixedBeta(int i, bool value);

	double getHessianDiagonal(int index);

	double getAsymptoticVariance(int i, int j);

	double getAsymptoticPrecision(int i, int j);

//	void setZeroBetaFixed(void);

	void update(const ModeFindingArguments& arguments);

	virtual void resetBeta(void);

	// Setters
	void setPrior(priors::JointPriorPtr newPrior);

	void setHyperprior(double value); // TODO depricate

	void setHyperprior(int index, double value);

	void setClassHyperprior(double value);

	void setPriorType(int priorType);

	void setWeights(double* weights);

	void setCensorWeights(double* weights); // ESK: New function

	std::vector<double> getWeights();

	std::vector<double> getCensorWeights(); // ESK:

	void setLogisticRegression(bool idoLR);

//	template <typename T>
	void setBeta(const std::vector<double>& beta);

	void setBeta(int i, double beta);

//	void double getHessianComponent(int i, int j);

	// Getters

	std::vector<double> getHyperprior(void) const;

	string getPriorInfo() const;

	string getCrossValidationInfo() const;

	void setCrossValidationInfo(string info);

	string getConditionId() const {
		return conditionId;
	}

	int getUpdateCount() const {
		return updateCount;
	}

	int getLikelihoodCount() const {
		return likelihoodCount;
	}

	UpdateReturnFlags getUpdateReturnFlag() const {
		return lastReturnFlag;
	}

	int getIterationCount() const {
		return lastIterationCount;
	}

	void setNoiseLevel(NoiseLevels);

	void makeDirty(void);

	void setInitialBound(double bound);

	Matrix computeFisherInformation(const std::vector<size_t>& indices) const;

	loggers::ProgressLogger& getProgressLogger() const { return *logger; }

	loggers::ErrorHandler& getErrorHandler() const { return *error; }

protected:

	bsccs::unique_ptr<AbstractModelSpecifics> privateModelSpecifics;

	AbstractModelSpecifics& modelSpecifics;
	priors::JointPriorPtr jointPrior;
	const AbstractModelData& hXI;

	CyclicCoordinateDescent(const CyclicCoordinateDescent& copy);

	void init(bool offset);

	void resetBounds(void);

	void computeXBeta(void);

	void saveXBeta(void);

	void computeFixedTermsInLogLikelihood(void);

	void computeFixedTermsInGradientAndHessian(void);

	void findMode(const int maxIterations, const int convergenceType, const double epsilon,
               const AlgorithmType algorithmType, const int qQN);

	template <typename Iterator>
	void findMode(Iterator begin, Iterator end,
		const int maxIterations, const int convergenceType, const double epsilon,
		const AlgorithmType algorithmType, const int qQN);

	template <typename Container>
	void computeKktConditions(Container& set);

	void kktSwindle(const ModeFindingArguments& arguments);

	void computeSufficientStatistics(void);

	void updateSufficientStatistics(double delta, int index);

	void computeNumeratorForGradient(int index);

	void computeAsymptoticPrecisionMatrix(void);

	void computeAsymptoticVarianceMatrix(void);

	template <class IteratorType>
	void incrementNumeratorForGradientImpl(int index);

	virtual void computeNEvents(void);

	virtual void updateXBeta(double delta, int index);

	template <class IteratorType>
	void updateXBetaImpl(double delta, int index);

	virtual void computeRemainingStatistics(bool skip, int index);

	virtual void computeRatiosForGradientAndHessian(int index);

	virtual void computeGradientAndHessian(
			int index,
			double *gradient,
			double *hessian);

	template <class IteratorType>
	void computeGradientAndHessianImpl(
			int index,
			double *gradient,
			double *hessian);

	void computeGradientAndHessianImplHand(
			int index,
						double *gradient,
						double *hessian);

	template <class IteratorType>
	void axpy(double* y, const double alpha, const int index);

	void axpyXBeta(const double beta, const int index);

	virtual void getDenominators(void);

	double computeLogLikelihood(void);

	void checkAllLazyFlags(void);

	double ccdUpdateBeta(int index);


	void mmUpdateAllBeta(std::vector<double>& allDelta,
                         const std::vector<bool>& fixedBeta);


	double applyBounds(
			double inDelta,
			int index);

	bool performCheckConvergence(int convergenceType,
                              double epsilon,
                              int maxIterations,
                              int iteration,
                              double* lastObjFunc);

	double computeConvergenceCriterion(double newObjFxn, double oldObjFxn);

	virtual double computeZhangOlesConvergenceCriterion(void);

	template <class T>
	void fillVector(T* vector, const int length, const T& value) {
		for (int i = 0; i < length; i++) {
			vector[i] = value;
		}
	}

	template <class T>
	void zeroVector(T* vector, const int length) {
		for (int i = 0; i < length; i++) {
			vector[i] = 0;
		}
	}

	int getAlignedLength(int N);

	void testDimension(int givenValue, int trueValue, const char *parameterName);

	template <class T>
	void printVector(T* vector, const int length, ostream &os);

	template <typename Real>
	double oneNorm(Real* vector, const int length);

	template <typename Real>
	double twoNormSquared(Real * vector, const int length);

	int sign(double x);

	template <class T>
	T* readVector(const char *fileName, int *length);

	// Local variables

	ofstream outLog;
	bool hasLog;

 	const double* hY; // K-vector
	const int* hPid;

	int** hXColumnRowIndicators; // J-vector

	typedef std::vector<double> DoubleVector;
	DoubleVector hBeta;

// 	DoubleVector& hXBeta; // TODO Delegate to ModelSpecifics
// 	DoubleVector& hXBetaSave; // Delegate
	DoubleVector hDelta;
	std::vector<bool> fixBeta;

	int N; // Number of patients
	int K; // Number of exposure levels
	int J; // Number of drugs

	string conditionId;

	bool computeMLE;
	int priorType;

	double initialBound;

	bool sufficientStatisticsKnown;
	bool xBetaKnown;
	bool fisherInformationKnown;
	bool varianceKnown;

	bool validWeights;
	bool useCrossValidation;
	bool doLogisticRegression;
	DoubleVector hWeights; // Make DoubleVector and delegate to ModelSpecifics
    DoubleVector cWeights; // ESK
	int updateCount;
	int likelihoodCount;

	NoiseLevels noiseLevel;
	UpdateReturnFlags lastReturnFlag;
	int lastIterationCount;

	Matrix hessianMatrix;
	Matrix varianceMatrix;

	typedef std::map<int, int> IndexMap;
	IndexMap hessianIndexMap;

	typedef std::pair<int, double> SetBetaEntry;
	typedef std::deque<SetBetaEntry> SetBetaContainer;

	SetBetaContainer setBetaList;

	string crossValidationInfo;

	loggers::ProgressLoggerPtr logger;
	loggers::ErrorHandlerPtr error;
};

double convertVarianceToHyperparameter(double variance);

} // namespace

#endif /* CYCLICCOORDINATEDESCENT_H_ */
