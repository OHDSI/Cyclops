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
#include <memory>

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

	AbstractModelSpecifics& getModelSpecifics() const { return modelSpecifics; }

	// Observe-only stopping statistics (Newton plan, Phase 2). With the environment variable CYCLOPS_SHADOW set
	// (1: curvature at log-spaced cycles and the last; 2: every cycle), serial CPU CCD fits on an engine with the
	// second-order API log one row per cycle; the fit itself is unchanged. NaN where a statistic does not apply.
	struct ShadowRow {
		int cycle = 0;
		bool curvature = false;     // the Hessian was formed this cycle
		double f = NAN;             // -(log likelihood + log prior)
		double stepDecrease = NAN;  // sum over the cycle's steps of h_j delta_j^2 / 2 (h_j with the prior's curvature)
		double rhoHat = NAN;        // trigger: rate from the last min(t, 20) step decreases
		double gHat = NAN;          // trigger: stepDecrease rho / (1 - rho)
		double R = NAN;             // sum sigma_j^2 s_j^2 / 2, all free coordinates Normal
		double kktMaxR = NAN;       // max |s_j| / sqrt(H_jj), end of cycle
		double lambda2Half = NAN;   // s_S' H_SS^-1 s_S / 2 (Laplace: S = nonzeros and violating zeros)
		double nu = NAN;            // bound on max_k ||x_k||_{H^-1}
		double gsc = NAN;           // omega(c Lambda) / c^2, c = 2 nu; Inf when c Lambda >= 1
		double RU = NAN;            // Lambda-bar^2 / 2 on U (None) and P (Normal) free sets
		double nuU = NAN;
		double gscU = NAN;          // omega(c Lambda-bar) / c^2
		double bracketLow = NAN;    // Lambda^2 / 2 bounds from 5 conjugate-gradient iterations, all Normal
		double bracketHigh = NAN;
		double consistency = NAN;   // Lambda^2 / max_j s_j^2 / H_jj (>= 1)
		int rankDeficit = 0;        // directions dropped from H_SS
	};

	const std::vector<ShadowRow>& getShadowLog() const { return shadowLog; }

	// What a "gap" fit did (Newton plan 2.10 reporting): the path, the stop's certificate and work counts
	struct ConvergenceReport {
		bool present = false;
		std::string algorithm;          // "ccd", or "newton" once a Newton step was taken
		std::string certificate = "none"; // "R", "gsc" or "none" (an estimate)
		bool certified = false;
		double tauEff = NAN;            // tolerance in log-posterior units: tau times the mean positive weight
		double gapEstimate = NAN;
		double R = NAN;
		double gsc = NAN;
		double lambda2Half = NAN;
		double costRatio = NAN;         // r_B: one Newton iteration in CCD cycles
		int cycles = 0;
		int newtonIterations = 0;
		int hessians = 0;
		int fallbacks = 0;              // Newton phases abandoned for CCD
		int newtonStart = -1;           // cycle at which Newton took over
		int droppedDirections = 0;      // aliased directions left out of the last Newton solve
		std::string solver;             // "dense" or "cg" (Newton-CG) once Newton is considered
		int hessianVectorProducts = 0;
		int certificationProducts = 0;  // Hv spent in certification solves
		std::string ineligible;         // why a requested Newton path ran as CCD
	};

	const ConvergenceReport& getConvergenceReport() const { return convergenceReport; }

	// Unpenalized coefficients whose likelihood is monotone (separation, Newton plan 2.7 and D5), as indices
	const std::vector<int>& getDivergentCoefficients() const { return divergentCoefficients; }

	// Groups of exact-duplicate columns merged in the last fit (Newton plan 2.8), representative first
	const std::vector<std::vector<int>>& getAliasedColumns() const { return aliasGroups; }

	// Profile and confint refits (Newton plan 2.9): the likelihood Hessian at the current coefficients, from which
	// refits under "auto" / "newton" + "gap" take chord Newton steps with a certificate that allows for the lag.
	// Inactive when the model has no second-order kernels, a free coefficient has a Laplace prior, or H is too large
	void beginProfileReference();
	void endProfileReference();
	void shareProfileReference(const CyclicCoordinateDescent& other);

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

	void accumulateKkt(int index, double rawDelta, const priors::GradientHessian& gh);

	double getKktCriterion() const;

	double computeKktResidual();

	void setUpAdaptiveIntercept(AlgorithmType algorithmType, double epsilon);

	void trackInterceptGradient(int index, double delta, double hessian);

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

	ConvergenceReport convergenceReport;
	std::vector<int> divergentDirection;   // per coefficient: -1 / +1 when its MLE is at -Inf / +Inf, else 0
	std::vector<int> divergentCoefficients;
	void findDivergentCoefficients();
	std::vector<std::vector<int>> aliasGroups;
	std::vector<uint64_t> columnHashes; // per column, cached: the data do not change
	void findAliasGroups();
	double divergentLimit(int j) const;
	std::vector<signed char> separationCache; // per column: -1, 0, +1, or 2 not yet computed (cleared with the weights)
	bool trackStepDecrease = false; // shadow log or "gap": accumulate shadowStepDecrease in the serial cycle
	std::string secondOrderIneligibility() const;
	struct NewtonContext;
	enum class NewtonOutcome { ACCEPTED, REJECTED, CERTIFIED, UNCERTIFIED, FAILED };
	void findModeGap(const std::function<void()>& cycle, int& iteration, int maxIterations, double tau,
	                 AlgorithmType algorithmType, int convergenceType);
	NewtonOutcome newtonIteration(NewtonContext& context);
	NewtonOutcome proxNewtonIteration(NewtonContext& context);
	bool newtonDirectionDense(NewtonContext& context, Eigen::VectorXd& s, Eigen::VectorXd& d, bool* stop,
	                          bool* certified, bool* damped);
	bool newtonDirectionCG(NewtonContext& context, Eigen::VectorXd& s, Eigen::VectorXd& d, bool* stop,
	                       bool* certified);
	bool certifyGap(NewtonContext& context, double* gscValue, double* lambda2Half);
	struct ProfileReference;
	struct ProfileFactor;
	std::vector<std::shared_ptr<const ProfileReference>> profilePool; // the mode first; kept between profile calls
	std::shared_ptr<const ProfileReference> profileReference;          // in use
	bool profileActive = false;
	std::shared_ptr<const ProfileReference> makeProfileReference(const ProfileReference* like);
	std::shared_ptr<ProfileFactor> profileFactor; // cached per set of free columns
	NewtonOutcome chordIteration(NewtonContext& context);

	int shadowMode = 0;
	double shadowStepDecrease = 0.0;
	std::vector<double> shadowDecreases;
	std::vector<ShadowRow> shadowLog;
	void shadowObserve(int cycle, bool last);
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
