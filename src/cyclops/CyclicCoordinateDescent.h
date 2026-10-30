/*
 * CyclicCoordinateDescent.h
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#ifndef CYCLICCOORDINATEDESCENT_H_
#define CYCLICCOORDINATEDESCENT_H_

#include <functional>
#include "CcdInterface.h"
#include "CompressedDataMatrix.h"
#include "ModelData.h"
#include "engine/AbstractModelSpecifics.h"
#include "priors/JointPrior.h"
#include "io/ProgressLogger.h"

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

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

	CyclicCoordinateDescent* clone(ComputeDeviceArguments computeDevice);

	void logResults(const char* fileName, bool withASE);

	virtual ~CyclicCoordinateDescent();

	double getLogLikelihood(void);

	double getLogLikelihoodGradient(int index);

	// The largest variance at which every Laplace-penalized coefficient is 0 (2 / lambda max^2), from a fit of
	// the unpenalized coefficients only; -1 unless all penalized coefficients share one Laplace or elastic net
	// prior variance (an elastic net with laplaceFraction a has lambda max divided by a)
	double getNullModelVariance(const ModeFindingArguments& arguments);

	double getLogPriorGradient(int index);

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

	double getJerkDiagonal(int index);

	double getAsymptoticVariance(int i, int j);

	// double getAsymptoticPrecision(int i, int j);

//	void setZeroBetaFixed(void);

	void update(const ModeFindingArguments& arguments);

	virtual void resetBeta(void);

	// Threads within one fit (the engine may evaluate gradients of several coordinates together)
	void setThreads(int threads);

	// Setters
	void setPrior(priors::JointPriorPtr newPrior);

	void setHyperprior(double value); // TODO depricate

	void setHyperprior(int index, double value);

	void setClassHyperprior(double value);

	void setPriorType(int priorType);

	void setWeights(double* weights);

	// Call after setWeights() in a cross-validation fold fit: rows with weight 0 may keep stale xBeta
	// until getNewPredictiveLogLikelihood(); the next setWeights() clears it
	void setHeldOutRowsUnused(bool unused);

	// Cross-validation: see AbstractModelSpecifics::makeFoldPartition()
	FoldPartitionPtr makeFoldPartition(const std::vector<int>& segmentOfRow, int folds) const;

	// Call after setWeights() and setHeldOutRowsUnused(true): the fold fit iterates the training
	// segments of `partition`; the next setWeights() or clearFoldPartition() drops it
	void setFoldPartition(FoldPartitionPtr partition, int fold);

	void clearFoldPartition();

	void setCensorWeights(double* weights); // ESK: New function

	std::vector<double> getWeights();

	std::vector<double> getCensorWeights(); // ESK:

	void getSchoenfeldResiduals(const IdType index,
                             std::vector<double>* residuals,
                             std::vector<double>* times,
                             std::vector<int>* strata,
                             std::vector<double>* covariate,
                             double* score);

	void setLogisticRegression(bool idoLR);

//	template <typename T>
	void setBeta(const std::vector<double>& beta);

	void setStartingBeta(const std::vector<double>& inStartingBeta);

	void setBeta(int i, double beta);

//	void double getHessianComponent(int i, int j);

	// Getters

	std::vector<double> getHyperprior(void) const;

	string getPriorInfo() const;

	string getCrossValidationInfo() const;

	void setCrossValidationInfo(string info);

	int getCrossValidationFailedFolds() const { return crossValidationFailedFolds; }

	void setCrossValidationFailedFolds(int count) { crossValidationFailedFolds = count; }

	// string getBootStrapInfo() const;
	//
	// void setBootStrapInfo(string info);

	const std::vector<double>& getBootStrapInfo() const;

	void setBootStrapInfo(const std::vector<double>& info);

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

	Matrix computeFisherInformation(const std::vector<IdType>& indices) const;

	loggers::ProgressLogger& getProgressLogger() const { return *logger; }

	loggers::ErrorHandler& getErrorHandler() const { return *error; }

// GPU
	void setWeights(double* iWeights, int syncCVIndex);

	double getPredictiveLogLikelihood(double* weights, int cvIndex);

	std::vector<double> getPredictiveLogLikelihood(std::vector<std::vector<double>>& weightsPool);

	void turnOnSyncCV(int foldToCompute);

	void turnOffSyncCV(void);

	void turnOnStreamCV(int foldToCompute);

	bool GPU() {
		return usingGPU;
	}

	std::vector<double> getObjectiveFunctions(int convergenceType);

	std::vector<double> getLogLikelihoods(void);

	std::vector<double> getLogPriors(void);

	std::vector<double> ccdUpdateBetaVec(int index);

	void ccdUpdateBetaVec(std::vector<double>& deltaVec, std::vector<std::pair<int,int>>& indicesToUpdate);

	void getFold(int fold);

	void getHBeta(void);

	void setHXBeta(void);

protected:

	bsccs::unique_ptr<AbstractModelSpecifics> privateModelSpecifics;

	AbstractModelSpecifics& modelSpecifics;
	priors::JointPriorPtr jointPrior;
	const AbstractModelData& hXI;

	CyclicCoordinateDescent(const CyclicCoordinateDescent& copy, ComputeDeviceArguments computeDevice);

	void init(bool offset);

	void resetBounds(void);

	void computeXBeta(void);

	void saveXBeta(void);

	void computeFixedTermsInLogLikelihood(void);

	void computeFixedTermsInGradientAndHessian(void);

	void findMode(const int maxIterations, const int convergenceType, const double epsilon,
               const AlgorithmType algorithmType, const int qQN, const bool doItAll);

	template <typename Iterator>
	void findMode(Iterator begin, Iterator end,
		const int maxIterations, const int convergenceType, const double epsilon,
		const AlgorithmType algorithmType, const int qQN, const bool doItAll);

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

	double scaledKktResidual(int index, double rawDelta, const priors::GradientHessian& gh) const;
	// Least squares: sqrt(2) sd(y), so that kkt residuals are in standard errors for any units of y, and var(y), the
	// floor of the objective's scale in computeConvergenceCriterion(); 1 otherwise. Set by setOutcomeScale()
	double kktOutcomeScale = 1.0;
	double objectiveFloor = 1.0;
	bool outcomeScaleKnown = false;
	void setOutcomeScale();

	void accumulateKkt(int index, double rawDelta, const priors::GradientHessian& gh);

	double getKktCriterion() const;

	double computeKktResidual();

	void setUpAdaptiveIntercept(AlgorithmType algorithmType, double epsilon);

	void trackInterceptGradient(int index, double delta, double hessian);

	// a Laplace or elastic net prior: a coordinate at zero stays there while |gradient| is within the kink
	bool hasZeroKink(int index) {
		const auto type = jointPrior->getPriorType(index);
		return type == priors::LAPLACE || type == priors::ELASTIC_NET;
	}

	void setUpZeroCertificates(AlgorithmType algorithmType);

	bool certifiedZero(int index) const;

	void recordZeroCertificate(int index, double delta);

	void cycleWithConcurrentGradients(const std::function<void(int)>& log);


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

	bool performCheckConvergence(int convergenceType,
                              double epsilon,
                              int maxIterations,
                              int iteration,
                              std::vector<double>& lastObjFunc);

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
	DoubleVector startingBeta;

// 	DoubleVector& hXBeta; // TODO Delegate to ModelSpecifics
// 	DoubleVector& hXBetaSave; // Delegate
	DoubleVector hDelta;
	bool boundLimitedStep = false; // a step in the current cycle was cut short by its bound
	DoubleVector lastStep;         // previous step of each coefficient, to detect period-2 oscillation
	std::vector<int> flipCount;    // consecutive steps that reversed sign without shrinking

	// "kkt" convergence: size of the minimum-norm subgradient s_j of the objective over the
	// coordinates of a cycle, each evaluated where ccdUpdateBeta() computed its gradient
	struct KktCycleStats {
		double maxScaled = 0.0; // max_j |s_j| / sqrt(h_j), h_j the curvature of the objective
		bool undefined = false; // a NaN gradient
	};
	KktCycleStats kktCycle;
	const int kktConsecutiveNeeded = 2; // cycles in a row below the tolerance
	int kktConsecutive = 0;
	bool trackKkt = false;
	priors::GradientHessian lastGradientHessian; // as passed to getDelta() by ccdUpdateBeta()

	// Active-set cycling (with "kkt" convergence): between full sweeps, cycles visit only coordinates
	// without a Laplace prior, Laplace coordinates away from zero and those that moved when last visited;
	// with a threshold > 0, only those whose last step, in standard-error units, was at least that
	// fraction of the largest step of the last full sweep
	int activeSetEvery = 0; // 0: off; k > 1: a full sweep at least every k cycles
	const int activeSetStart = 2; // first cycles that are full sweeps
	bool nextCycleFull = true;
	bool lastCycleFull = true;
	int cyclesSinceFull = 0;
	std::vector<char> activeSetCandidate; // Laplace coordinates, which can stay at zero
	std::vector<double> lastStepSE; // |delta| sqrt(h) when last visited
	double activeSetThreshold = 0.0;
	double activeSetTau = 0.0; // activeSetThreshold times the largest step in the last full sweep
	double cycleMaxStepSE = 0.0;
	bool inActiveSet(int index) const {
		if (activeSetThreshold > 0.0) {
			return lastStepSE[index] >= activeSetTau;
		}
		return !activeSetCandidate[index] || hBeta[index] != 0.0 || lastStepSE[index] > 0.0;
	}

	// Adaptive extra intercept steps (logistic regression with an unpenalized intercept column, serial
	// cycles, "kkt" convergence): an extra Newton step on the intercept whenever its predicted gain
	// g0^2 / (2 h0) per row of work (2 K) reaches adaptiveInterceptRatio times the gain per row of work of
	// the coordinate steps in the previous cycle; g0, the intercept gradient, is tracked to first order from
	// the coordinate steps. In sparse models those gains are near zero, so the rule also needs g0 / sqrt(h0)
	// above the tolerance; without that floor it took a step after almost every coordinate.
	bool adaptiveIntercept = false;
	static constexpr double adaptiveInterceptRatio = 0.25;
	int interceptIndex = -1; // in the current findMode(); -1 when off
	bool interceptIsFree = false; // least squares with a lazy intercept: a step costs O(1), taken after every move
	double interceptFloor = 0.0; // the tolerance, in standard-error units
	std::vector<double> interceptMomentRatio; // sum_i x_ij / sum_i x_ij^2 by column
	std::vector<double> columnRows; // stored entries by column (K for dense and intercept)
	struct AdaptiveInterceptState {
		double gradient = 0.0; // tracked change of the intercept gradient since its last step
		double hessian = 0.0;  // intercept curvature at its last regular step
		double gain = 0.0;     // predicted gain of this cycle's coordinate steps
		double cost = 0.0;     // their rows of work
		double rate = 0.0;     // gain / cost of the previous cycle
		long extraSteps = 0;
	};
	AdaptiveInterceptState interceptState;

	// Certified zero steps (serial cycles): a Laplace coordinate at zero takes a zero step exactly when
	// |g| <= lambda. Its gradient g, last computed when it stayed at zero, can since have risen and
	// fallen by at most what the model's drift bounds say (AbstractModelSpecifics::getGradientDrift());
	// while g plus the rise stays below lambda and g minus the fall above -lambda, the visit is skipped,
	// with the bookkeeping of a zero step, so results are unchanged.
	struct ZeroCertificate {
		double gradient = 0.0; // g when last computed
		double hessian = 0.0;  // the column's Hessian then
		int split = 0;         // its bucket (see GradientDrift)
		double up = 0.0;       // the model's drift totals then
		double down = 0.0;
		double maxUp = 0.0;
		double maxDown = 0.0;
		double lowUp = 0.0;
		double lowDown = 0.0;
		double highUp = 0.0;
		double highDown = 0.0;
		long updates = 0;      // the model's update count then
		long epoch = -1; // -1: none
	};
	bool useZeroCertificates = false;
	std::vector<ZeroCertificate> zeroCertificates;
	std::vector<double> certifiedRows; // entries of each indicator column; 0 (never skipped) otherwise
	int minIterations = 0; // of the current update(); no convergence (SUCCESS) before this many cycles
	std::vector<bool> fixBeta;

	int N; // Number of patients
	int K; // Number of exposure levels
	int J; // Number of drugs

	string conditionId;

	bool computeMLE;
	int priorType = priors::NONE;

	double initialBound;

	bool sufficientStatisticsKnown;
	bool xBetaKnown;
	bool fisherInformationKnown;
	bool varianceKnown;

	bool validWeights;
	bool useCrossValidation;
	bool heldOutRowsUnused = false;
	FoldPartitionPtr foldPartition;
	int foldPartitionFold = -1;
	bool doLogisticRegression;
	DoubleVector hWeights; // Make DoubleVector and delegate to ModelSpecifics
    DoubleVector cWeights; // ESK
	int updateCount;
	int likelihoodCount;

	NoiseLevels noiseLevel;
	int threads = 1;
	std::vector<bool> stayedAtZero; // coordinate was at zero and did not move in the last cycle
	double moveRate = 0.0; // recent chance that such a coordinate moves
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
	int crossValidationFailedFolds = 0; // fold fits left out of their mean predictive log-likelihood
	string bootStrapInfo;
	std::vector<double> bootStrapInfoVector;

	loggers::ProgressLoggerPtr logger;
	loggers::ErrorHandlerPtr error;

	// GPU items
	std::vector<DoubleVector> hBetaPool;
	std::vector<DoubleVector> hDeltaPool;
	std::vector<std::vector<bool>> fixBetaPool;
	std::vector<bool> donePool;
	std::vector<DoubleVector> hWeightsPool; // Make DoubleVector and delegate to ModelSpecifics

	bool usingGPU;
	bool syncCV = false;
	int syncCVFolds;
	bool usingCUDA;
	bool streamCV = false;
	int streamCVFolds;

};

double convertVarianceToHyperparameter(double variance);

} // namespace

#endif /* CYCLICCOORDINATEDESCENT_H_ */
