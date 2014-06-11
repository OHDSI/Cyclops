/*
 * Model.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: tshaddox
 */



#include "MCMCModel.h"

namespace bsccs {

 MCMCModel::MCMCModel(){}

 void MCMCModel::initialize(CyclicCoordinateDescent& ccdIn, long int seed){
	 cout << "MCMCModel Initialize" << endl;


	 ccd = &ccdIn;

	 J = ccd->getBetaSize();
	 initializeHessian();

	 HessianMatrix = (ccd->getHessianMatrix()).cast<float>();

	 generateCholesky();

	 // Beta_Hat = modes from ccd
	 Beta_Hat.initialize(*ccd, J);
	 cout << "prestore?!" << endl;
	 Beta_Hat.store();
	 cout << "poststore?!" << endl;
	 // Set up Beta
	 Beta.initialize(*ccd, J);
	 cout << "here?!" << endl;
	 //Beta.setProbabilityUpdate(betaAmount);
	 Beta.store();

	 // Set up Sigma squared
	 bsccs::real sigma2Start;
	 sigma2Start = (bsccs::real) ccd->getHyperprior();
	 SigmaSquared.initialize(*ccd,1);
	 SigmaSquared.logParameter();

	logLikelihood = ccd->getLogLikelihood();
	logPrior = ccd->getLogPrior();

	useHastingsRatio = true;
 }

 MCMCModel::~MCMCModel(){}

 void MCMCModel::initializeHessian() {

 	for (int i = 0; i < J; i ++){
 		vector<bsccs::real> columnInHessian(J,0);
 		hessian.push_back(columnInHessian);
 	}
 	clearHessian();
 }

 void MCMCModel::resetWithNewSigma(){
		// TODO Need Wrapper for this....
	 	// cout << "\n\n \t\t MCMCModel::resetWithNewSigma " << endl;
	 	// cout << "old likelihood = " << ccd->getLogLikelihood() << endl;
		ccd->resetBeta();
		ccd->setHyperprior(SigmaSquared.get(0));
		int ZHANG_OLES = 1;
		int ccdIterations = 100;
		double tolerance = 5E-4;

		ccd->update(ccdIterations, ZHANG_OLES, tolerance);
		//ccd->setUpHessianComponents(true);
		clearHessian();
		//ccd->getHessian(&hessian);
		generateCholesky();
		//Beta_Hat.set(ccd->hBeta);
		//Beta_Hat.setRestorable(true);
		//Beta.set(ccd->hBeta);
		Beta.setRestorable(true);

		setLogLikelihood(ccd->getLogLikelihood());
		setLogPrior(ccd->getLogPrior());
		// cout << "new likelihood = " << logLikelihood << endl;
		newLogPriorAndLikelihood = true;
 }

 void MCMCModel::clearHessian() {

 	for (int i = 0; i < J; i ++){
 		for (int j = 0; j < J; j++) {
 			hessian[j][i] = 0;
 		}
 	}
 }

 void MCMCModel::generateCholesky() {
 	HessianMatrix.resize(J, J);

 	//Convert to Eigen for Cholesky decomposition
 	for (int i = 0; i < J; i++) {
 		for (int j = 0; j < J; j++) {
 			HessianMatrix(i, j) = -hessian[i][j];
 		}
 	}

 	//  Debugging code
 	HessianMatrixInverse.resize(J,J);
 	HessianMatrixInverse = HessianMatrix.inverse();
 	// cout << "Hessian Inverse" << endl;
 	// cout << HessianMatrixInverse(0,0) << endl;

 	//Perform Cholesky Decomposition
 	CholDecom.compute(HessianMatrix);

 #ifdef Debug_TRS
 		cout << "Printing Hessian in generateCholesky" << endl;

 		for (int i = 0; i < J; i ++) {
 			cout << "[";
 				for (int j = 0; j < J; j++) {
 					cout << HessianMatrix(i,j) << ", ";
 				}
 			cout << "]" << endl;
 			}
 #endif

 }

void MCMCModel::restore(){

	if (Beta_Hat.getRestorable()){
		Beta_Hat.restore();
	}
	if (Beta.getRestorable()) {
		Beta.restore();
	}
	if (SigmaSquared.getRestorable()){
		SigmaSquared.restore();
	}

	setLogLikelihood(storedLogLikelihood);
	setLogPrior(storedLogPrior);
}

void MCMCModel::acceptChanges(){
	// cout << "MCMCModel::acceptChanges" << endl;

	Beta_Hat.setRestorable(false);
	Beta.setRestorable(false);
	SigmaSquared.setRestorable(false);
}

void MCMCModel::setNewLogPriorAndLikelihood(bool newOrNot){
	newLogPriorAndLikelihood = newOrNot;
}

bool MCMCModel::getNewLogPriorAndLikelihood(){
	return(newLogPriorAndLikelihood);
}


void MCMCModel::Beta_HatStore(){
	Beta_Hat.store();
}
void MCMCModel::BetaStore(){
	// cout <<"MCMCModel:BetaStore" << endl;

	Beta.store();
	//cout << "Beta current" << endl;
	//Beta.logParameter();
	//cout << "Beta storred" << endl;
	//Beta.logStored();
	//cout << "MCMCModel:BetaStore End" << endl;

}void MCMCModel::SigmaSquaredStore(){
	SigmaSquared.store();
}


void MCMCModel::logState(){
	// cout << "MCMCModel::logState" << endl;
	Beta.logParameter();
	Beta.logStored();
	Beta_Hat.logParameter();
	Beta_Hat.logStored();
	SigmaSquared.logParameter();
	SigmaSquared.logStored();
}

void MCMCModel::writeVariances(){
	for(int i = 0; i<J; i ++){
		Beta.set(i, HessianMatrixInverse(i,i));
		//cout<< "Beta[" <<i <<"] = " << Beta.get(i) << endl;
	}

	Beta.setRestorable(true);

	setLogLikelihood(0.1);

	setLogPrior(0.2);

}

bool MCMCModel::getUseHastingsRatio(){
	return(useHastingsRatio);
}

void MCMCModel::setUseHastingsRatio(bool newUseHastingsRatio){
	useHastingsRatio = newUseHastingsRatio;
}


double MCMCModel::getLogLikelihood(){
	return(logLikelihood);
}

double MCMCModel::getStoredLogLikelihood(){
	return(storedLogLikelihood);
}

void MCMCModel::setLogLikelihood(double newLoglikelihood){
	storedLogLikelihood = logLikelihood;
	logLikelihood = newLoglikelihood;
}

double MCMCModel::getLogPrior(){
	return(logPrior);
}

double MCMCModel::getStoredLogPrior(){
	return(storedLogPrior);
}

void MCMCModel::setLogPrior(double newLogPrior){
	storedLogPrior = logPrior;
	logPrior = newLogPrior;
}

 BetaParameter& MCMCModel::getBeta(){
	 return(Beta);
 }

 BetaParameter& MCMCModel::getBeta_Hat(){
	 return(Beta_Hat);
 }

 HyperpriorParameter& MCMCModel::getSigmaSquared(){
	 return(SigmaSquared);
 }

 Eigen::LLT<Eigen::MatrixXf> MCMCModel::getCholeskyLLT(){
	 return(CholDecom);
 }

 Eigen::MatrixXf& MCMCModel::getHessian(){
 	 return(HessianMatrix);
  }

 double MCMCModel::getTuningParameter(){
	 return(tuningParameter);
 }

 void MCMCModel::setTuningParameter(double nextTuningParameter){
	 tuningParameter = nextTuningParameter;
 }

 CyclicCoordinateDescent& MCMCModel::getCCD(){
	 return(*ccd);
 }




}
