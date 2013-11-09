/*
 * Model.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: tshaddox
 */



#include "Model.h"

namespace bsccs {

 Model::Model(){}

 void Model::initialize(CyclicCoordinateDescent& ccdIn, long int seed){
	 // cout << "Model Initialize" << endl;

	 boost::mt19937 rng(seed);
	 ccd = &ccdIn;
	 J = ccd->getBetaSize();
	 ccd->setUpHessianComponents(false);
	 initializeHessian();
	 ccd->computeXBeta_GPU_TRS_initialize();

	 clearHessian();
	 ccd->getHessian(&hessian);
	 generateCholesky();
	 // Beta_Hat = modes from ccd

	 Beta_Hat.initialize(ccd->hBeta, J);
	 //Beta_Hat.store();
	 // Set up Beta
	 Beta.initialize(ccd->hBeta, J);
	 //Beta.setProbabilityUpdate(betaAmount);
	 Beta.store();

	 // Set up Sigma squared
	 bsccs::real sigma2Start;
	 sigma2Start = (bsccs::real) ccd->sigma2Beta;
	 SigmaSquared.initialize(&sigma2Start, 1);
	 SigmaSquared.logParameter();

	logLikelihood = ccd->getLogLikelihood();
	logPrior = ccd->getLogPrior();

	useHastingsRatio = true;
 }

 Model::~Model(){}

 void Model::initializeHessian() {

 	for (int i = 0; i < J; i ++){
 		vector<bsccs::real> columnInHessian(J,0);
 		hessian.push_back(columnInHessian);
 	}
 }

 void Model::resetWithNewSigma(){
		// TODO Need Wrapper for this....
	 	// cout << "\n\n \t\t Model::resetWithNewSigma " << endl;
	 	// cout << "old likelihood = " << ccd->getLogLikelihood() << endl;
		ccd->resetBeta();
		ccd->setHyperprior(SigmaSquared.get(0));
		int ZHANG_OLES = 1;
		int ccdIterations = 100;
		double tolerance = 5E-4;

		ccd->update(ccdIterations, ZHANG_OLES, tolerance);
		ccd->setUpHessianComponents(true);
		clearHessian();
		ccd->getHessian(&hessian);
		generateCholesky();
		Beta_Hat.set(ccd->hBeta);
		Beta_Hat.setRestorable(true);
		Beta.set(ccd->hBeta);
		Beta.setRestorable(true);

		setLogLikelihood(ccd->getLogLikelihood());
		setLogPrior(ccd->getLogPrior());
		// cout << "new likelihood = " << logLikelihood << endl;
		newLogPriorAndLikelihood = true;
 }

 void Model::clearHessian() {

 	for (int i = 0; i < J; i ++){
 		for (int j = 0; j < J; j++) {
 			hessian[j][i] = 0;
 		}
 	}
 }

 void Model::generateCholesky() {
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

void Model::restore(){
	// cout << "Model::restore" << endl;
	if (Beta_Hat.getRestorable()){
		Beta_Hat.restore();
	}
	if (Beta.getRestorable()) {
		Beta.restore();
	}
	if (SigmaSquared.getRestorable()){
		SigmaSquared.restore();
	}
	//cout << "in model::restore, storedLogLikelihood = " << storedLogLikelihood << endl;
	setLogLikelihood(storedLogLikelihood);
	setLogPrior(storedLogPrior);
}

void Model::acceptChanges(){
	// cout << "Model::acceptChanges" << endl;

	Beta_Hat.setRestorable(false);
	Beta.setRestorable(false);
	SigmaSquared.setRestorable(false);
}

void Model::setNewLogPriorAndLikelihood(bool newOrNot){
	newLogPriorAndLikelihood = newOrNot;
}

bool Model::getNewLogPriorAndLikelihood(){
	return(newLogPriorAndLikelihood);
}


void Model::Beta_HatStore(){
	Beta_Hat.store();
}
void Model::BetaStore(){
	// cout <<"Model:BetaStore" << endl;

	Beta.store();
	//cout << "Beta current" << endl;
	//Beta.logParameter();
	//cout << "Beta storred" << endl;
	//Beta.logStored();
	//cout << "Model:BetaStore End" << endl;

}void Model::SigmaSquaredStore(){
	SigmaSquared.store();
}


void Model::logState(){
	// cout << "Model::logState" << endl;
	Beta.logParameter();
	Beta.logStored();
	Beta_Hat.logParameter();
	Beta_Hat.logStored();
	SigmaSquared.logParameter();
	SigmaSquared.logStored();
}

void Model::writeVariances(){
	for(int i = 0; i<J; i ++){
		Beta.set(i, HessianMatrixInverse(i,i));
		//cout<< "Beta[" <<i <<"] = " << Beta.get(i) << endl;
	}

	Beta.setRestorable(true);

	setLogLikelihood(0.1);

	setLogPrior(0.2);

}

bool Model::getUseHastingsRatio(){
	return(useHastingsRatio);
}

void Model::setUseHastingsRatio(bool newUseHastingsRatio){
	useHastingsRatio = newUseHastingsRatio;
}


double Model::getLogLikelihood(){
	return(logLikelihood);
}

double Model::getStoredLogLikelihood(){
	return(storedLogLikelihood);
}

void Model::setLogLikelihood(double newLoglikelihood){
	storedLogLikelihood = logLikelihood;
	logLikelihood = newLoglikelihood;
}

double Model::getLogPrior(){
	return(logPrior);
}

double Model::getStoredLogPrior(){
	return(storedLogPrior);
}

void Model::setLogPrior(double newLogPrior){
	storedLogPrior = logPrior;
	logPrior = newLogPrior;
}

 Parameter& Model::getBeta(){
	 return(Beta);
 }

 Parameter& Model::getBeta_Hat(){
	 return(Beta_Hat);
 }

 Parameter& Model::getSigmaSquared(){
	 return(SigmaSquared);
 }

 Eigen::LLT<Eigen::MatrixXf> Model::getCholeskyLLT(){
	 return(CholDecom);
 }

 Eigen::MatrixXf& Model::getHessian(){
 	 return(HessianMatrix);
  }

 double Model::getTuningParameter(){
	 return(tuningParameter);
 }

 void Model::setTuningParameter(double nextTuningParameter){
	 tuningParameter = nextTuningParameter;
 }

 CyclicCoordinateDescent& Model::getCCD(){
	 return(*ccd);
 }

 boost::mt19937 & Model::getRng(){
	 return(rng);
 }



}
