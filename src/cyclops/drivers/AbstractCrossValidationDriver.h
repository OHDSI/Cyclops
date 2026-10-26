/*
 * CrossValidationDriver.h
 *
 *  Created on: Sep 10, 2010
 *      Author: msuchard
 */

#ifndef ABSTRACTCROSSVALIDATIONDRIVER_H_
#define ABSTRACTCROSSVALIDATIONDRIVER_H_

#include "AbstractDriver.h"

namespace bsccs {

 // forward references
class CyclicCoordinateDescent;
class AbstractSelector;
struct CCDArguments;

struct MaxPoint {
    std::vector<double> point;
    double value;
};

class AbstractCrossValidationDriver : public AbstractDriver {
public:
	AbstractCrossValidationDriver(
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
			std::vector<double>* wtsExclude = nullptr
	);

	virtual ~AbstractCrossValidationDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

	virtual void resetForOptimal(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

	virtual void logResults(const CCDArguments& arguments) = 0; // pure virtual

protected:

    // Derived classes use different optimization loops
	virtual MaxPoint doCrossValidationLoop(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool) = 0; // pure virtual

	double doCrossValidationStep(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments,
			int step,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool,
			std::vector<double> & predLogLikelihood);

	double computePointEstimate(const std::vector<double>& value);

	double computeStDev(const std::vector<double>& value, double mean);

	// Mean and standard deviation of the fold scores of each hyperparameter tried, over the folds computed
	// at every one of them (a fold's score sums over its rows, so means over different folds are not
	// comparable); over each one's own computed folds when no fold was computed at all of them. NaN
	// when a hyperparameter has no computed fold.
	void computeCommonFoldEstimates(const std::vector<std::vector<double>>& scores,
	                                std::vector<double>& means, std::vector<double>& stdDevs);

	void restoreWeights(CyclicCoordinateDescent& ccd, const CCDArguments& arguments);

	MaxPoint maxPoint;
	std::vector<double>* weightsExclude;
	std::vector<double> savedWeights; // before cross-validation; empty when unweighted
	int failedFolds = 0; // fold fits left out of their mean predictive log-likelihood
	struct FoldSolution {
	    std::vector<double> hyperprior;
	    std::vector<std::pair<int, double>> beta; // non-zero coefficients
	};
	std::vector<std::vector<FoldSolution>> foldSolutions; // of each fold, at the hyperparameters tried
};

} // namespace

#endif /* ABSTRACTCROSSVALIDATIONDRIVER_H_ */
