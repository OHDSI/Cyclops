/*
 * Model.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: tshaddox
 */



#include "MCMCModel.h"

#define Debug_TRS;

namespace bsccs {

 MCMCModel::MCMCModel(){}

 void MCMCModel::initialize(CyclicCoordinateDescent& ccdIn, long int seed){

	 ccd = &ccdIn;

	 J = ccd->getBetaSize();

	 set<int> nullFixed;

 	 setFixedIndices(nullFixed);

	 initializeHessian();

	 HessianMatrix = (ccd->getHessianMatrix()).cast<float>();

	 generateCholesky();

	 // Beta_Hat = modes from ccd
	 Beta_Hat.initialize(*ccd, J);
	 Beta_Hat.store();

	 // Set up Beta
	 Beta.initialize(*ccd, J);
	 //Beta.setProbabilityUpdate(betaAmount);
	 Beta.store();

	 // Set up Sigma squared

	 SigmaSquared.initialize(*ccd,1);
	 SigmaSquared.store();
	// SigmaSquared.logParameter();

	 logLikelihood = ccd->getLogLikelihood();
	 logPrior = ccd->getLogPrior();
	 storedLogLikelihood = ccd->getLogLikelihood();
	 storedLogPrior = ccd->getLogPrior();

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
		ccd->resetBeta();
		ccd->setHyperprior(SigmaSquared.get(0));
		int ZHANG_OLES = 1;
		int ccdIterations = 100;
		double tolerance = 5E-4;

		for (int j = 0; j < J; j++){
			ccd->setFixedBeta(j, 0);
		}

		set<int>::iterator iter;
		for(iter=fixedBetaIndices.begin(); iter!=fixedBetaIndices.end();++iter) {
			ccd->setBeta((*iter), 0);
			ccd->setFixedBeta((*iter),1);
		}

		ccd->update(ccdIterations, ZHANG_OLES, tolerance);
		HessianMatrix = (ccd->getHessianMatrix()).cast<float>();
		//ccd->setUpHessianComponents(true);
		//clearHessian();
		//ccd->getHessian(&hessian);
		generateCholesky();
		for (int i = 0; i < J; i++){
			Beta_Hat.set(i,ccd->getBeta(i));
			Beta.set(i,ccd->getBeta(i));
		}

		Beta_Hat.setRestorable(true);
		Beta.setRestorable(true);

		setLogLikelihood(ccd->getLogLikelihood());
		setLogPrior(ccd->getLogPrior());
		newLogPriorAndLikelihood = true;
 }

 void MCMCModel::syncCCDwithModel(set<int>& lastFixedIndices){

	ccd->resetBeta();
	ccd->setHyperprior(SigmaSquared.get(0));
	int ZHANG_OLES = 1;
	int ccdIterations = 100;
	double tolerance = 5E-4;


	for(int i = 0; i < J; i++) {
		ccd->setFixedBeta(i,0);
	}

	set<int>::iterator iter;
	for(iter=fixedBetaIndices.begin(); iter!=fixedBetaIndices.end();++iter) {
		ccd->setBeta((*iter), 0);
		ccd->setFixedBeta((*iter),1);
	}

	ccd->update(ccdIterations, ZHANG_OLES, tolerance);
	HessianMatrix = (ccd->getHessianMatrix()).cast<float>();
	generateCholesky();
	Beta.setRestorable(true);

	setLogLikelihood(ccd->getLogLikelihood());
	setLogPrior(ccd->getLogPrior());
	newLogPriorAndLikelihood = true;
 }

 void MCMCModel::setFixedIndices(set<int>& fixedIndices){
	 modelKey = "";
	 ostringstream convert;
	 set<int> newSet;
	 fixedBetaIndices = newSet;
	 set<int> newSet2;
	 variableBetaIndices = newSet;
	 set<int>::iterator iter;
	 for(iter=fixedIndices.begin(); iter!=fixedIndices.end();++iter) {
		 fixedBetaIndices.insert((*iter));
		 convert << (*iter)<< "|";
	 }
	 modelKey = convert.str();
	 setVariableIndices();
 }

 void MCMCModel::setVariableIndices(){
 	 //variableBetaIndices(J - fixedBetaIndices.size());
 	 set<int> allBetaIndices;

 	 for (int i = 0;i < J; i++){
 		 allBetaIndices.insert(i);
 	 }

 	 std::set_difference(allBetaIndices.begin(), allBetaIndices.end(),fixedBetaIndices.begin(), fixedBetaIndices.end(),
 	    std::inserter(variableBetaIndices, variableBetaIndices.end()));
 }

 set<int> MCMCModel::getVariableIndices(){
	 return(variableBetaIndices);
 }

 set<int> MCMCModel::getFixedIndices(){
 	 return(fixedBetaIndices);
  }

 int MCMCModel::getFixedSize(){
	 return(fixedBetaIndices.size());
 }

 void MCMCModel::setModelProbability(double modelProbabilityIn){
	 modelProbability = modelProbabilityIn;
  }

 double MCMCModel::getModelProbability(){
 	 return(modelProbability);
  }

 string MCMCModel::getModelKey(){
 	 return(modelKey);
  }

 void MCMCModel::printFixedIndices(){
	 cout << "fixed indices are <";
	 set<int>::iterator iter;
	 for(iter=fixedBetaIndices.begin(); iter!=fixedBetaIndices.end();++iter) {
		 cout<<(*iter) << ", ";
	 }
	 cout << ">" << endl;

	 cout << "variable indices are <";
	 set<int>::iterator iter2;
	 for(iter2=variableBetaIndices.begin(); iter2!=variableBetaIndices.end();++iter2) {
		 cout<<(*iter2) << " ";
	 }
	 cout << ">" << endl;
 }

 int MCMCModel::getBetaSize(){
 	 return(Beta.getSize() - fixedBetaIndices.size());
  }

 void MCMCModel::setBeta(Eigen::VectorXf& newBeta){
	 int counter = 0;

	 set<int>::iterator iter;
	 for(iter=fixedBetaIndices.begin(); iter!=fixedBetaIndices.end();++iter) {
		 Beta.set((*iter), 0);
		 ccd->setBeta((*iter), 0);
		 ccd->setFixedBeta((*iter),1);
	 }
	 set<int>::iterator iter2;
	 for(iter2=variableBetaIndices.begin(); iter2!=variableBetaIndices.end();++iter2) {
		 Beta.set((*iter2), newBeta[counter]);
		 ccd->setBeta((*iter2), newBeta[counter]);
		 counter ++;
	 }

   }

 void MCMCModel::clearHessian() {

 	for (int i = 0; i < J; i ++){
 		for (int j = 0; j < J; j++) {
 			hessian[j][i] = 0;
 		}
 	}
 }

 void MCMCModel::generateCholesky() {

 	//  Debugging code
 	HessianMatrixInverse.resize(J,J);
 	HessianMatrixInverse = HessianMatrix.inverse();


 	//Perform Cholesky Decomposition
 	CholDecom.compute(HessianMatrix);

 	Eigen::MatrixXf L = CholDecom.matrixL();


 }

 void MCMCModel::store(){


 	Beta_Hat.store();
 	Beta_Hat.setRestorable(true);
  	Beta.store();
  	Beta.setRestorable(true);

 	SigmaSquared.store();
 	SigmaSquared.setRestorable(true);

 	storedLogLikelihood = logLikelihood;
 	storedLogPrior = logPrior;
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
	Beta.store();
}

void MCMCModel::SigmaSquaredStore(){
	SigmaSquared.store();
}


void MCMCModel::logState(){
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
