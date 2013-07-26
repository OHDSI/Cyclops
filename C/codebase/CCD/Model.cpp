/*
 * Model.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: tshaddox
 */



#include "Model.h"

namespace bsccs {

 Model::Model(){}

 void Model::initialize(CyclicCoordinateDescent& ccd){
	 cout << "Model Initialize" << endl;
	 J = ccd.getBetaSize();
	 ccd.setUpHessianComponents();
	 initializeHessian();
	 ccd.computeXBeta_GPU_TRS_initialize();

	 clearHessian();
	 ccd.getHessian(&hessian);
	 generateCholesky();
	 // Beta_Hat = modes from ccd

	 Beta_Hat.initialize(ccd.hBeta, J);

	 // Set up Beta
	 Beta.initialize(ccd.hBeta, J);
	 //Beta.setProbabilityUpdate(betaAmount);
	 Beta.store();
	 // Set up Sigma squared
	 bsccs::real sigma2Start;
	 sigma2Start = (bsccs::real) ccd.sigma2Beta;
	 SigmaSquared.initialize(&sigma2Start, 1);
	 SigmaSquared.logParameter();
 }

 Model::~Model(){}

 void Model::initializeHessian() {

 	for (int i = 0; i < J; i ++){
 		vector<bsccs::real> columnInHessian(J,0);
 		hessian.push_back(columnInHessian);
 	}
 }

 void Model::resetWithNewSigma(CyclicCoordinateDescent& ccd){
		// TODO Need Wrapper for this....
		ccd.resetBeta();
		ccd.setHyperprior(SigmaSquared.get(0));
		int ZHANG_OLES = 1;
		int ccdIterations = 100;
		double tolerance = 5E-4;

		ccd.update(ccdIterations, ZHANG_OLES, tolerance);
		clearHessian();
		ccd.getHessian(&hessian);
		generateCholesky();
		Beta_Hat.set(ccd.hBeta);
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
 			// TODO Debugging here
 //			if (i == j) {
 //				HessianMatrix(i,j) = 1.0;
 //			} else {
 //				HessianMatrix(i,j) = 0.0;
 //			}

 		}
 	}

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
	cout << "Model::restore" << endl;
	if (Beta_HatRestorable){
		Beta_Hat.restore();
		Beta_HatRestorable = false;
	}
	if (BetaRestorable) {
		Beta.restore();
		BetaRestorable = false;
	}
	if (SigmaSquaredRestorable){
		SigmaSquared.restore();
		SigmaSquaredRestorable = false;
	}
}

void Model::Beta_HatRestorableSet(bool restorable){
	Beta_HatRestorable = restorable;
}
void Model::BetaRestorableSet(bool restorable){
	BetaRestorable = restorable;
}
void Model::SigmaSquaredRestorableSet(bool restorable){
	SigmaSquaredRestorable = restorable;
}

void Model::logState(){
	cout << "Model::logState" << endl;
	Beta.logParameter();
	Beta.logStored();
	Beta_Hat.logParameter();
	Beta_Hat.logStored();
	SigmaSquared.logParameter();
	SigmaSquared.logStored();
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


}
