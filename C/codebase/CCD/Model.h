/*
 * Model.h
 *
 *  Created on: Jul 24, 2013
 *      Author: tshaddox
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "Parameter.h"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "CyclicCoordinateDescent.h"


namespace bsccs {
class Model {
public:
	Model();
	virtual ~Model();

	void generateCholesky();
	void initialize(CyclicCoordinateDescent& ccdIn, long int seed);
	void initializeHessian();

	void clearHessian();

	void Beta_HatRestorableSet(bool restorable);
	void BetaRestorableSet(bool restorable);
	void SigmaSquaredRestorableSet(bool restorable);
	void Beta_HatStore();
	void BetaStore();
	void SigmaSquaredStore();

	bool getNewLogPriorAndLikelihood();
	void setNewLogPriorAndLikelihood(bool newOrNot);

	void acceptChanges();

	void restore();

	void logState();

	void writeVariances();

	void resetWithNewSigma();

	bool getUseHastingsRatio();
	void setUseHastingsRatio(bool useHastingsRatio);

	double getLogLikelihood();
	double getStoredLogLikelihood();
	void setLogLikelihood(double newLikelihood);

	double getLogPrior();
	double getStoredLogPrior();
	void setLogPrior(double newPrior);

	Parameter& getBeta();
	Parameter& getBeta_Hat();
	Parameter& getSigmaSquared();
	Eigen::LLT<Eigen::MatrixXf> getCholeskyLLT();
	Eigen::MatrixXf& getHessian();

	double getTuningParameter();
	void setTuningParameter(double nextTuningParameter);

	boost::mt19937 & getRng();


	CyclicCoordinateDescent& getCCD();

private:
	Eigen::LLT<Eigen::MatrixXf> CholDecom;

	Eigen::MatrixXf HessianMatrix;
	Eigen::MatrixXf HessianMatrixInverse;


	vector<vector<bsccs::real> > hessian;

	CyclicCoordinateDescent* ccd;

	Parameter Beta_Hat;
	Parameter Beta;
	Parameter SigmaSquared;

	double logLikelihood;
	double logPrior;

	double storedLogLikelihood;
	double storedLogPrior;

	bool useHastingsRatio;

	bool newLogPriorAndLikelihood;

	int J;

	double tuningParameter;

	boost::mt19937 rng;


};
}





#endif /* MODEL_H_ */
