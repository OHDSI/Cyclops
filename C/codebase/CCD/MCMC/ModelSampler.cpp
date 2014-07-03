/*
 * ModelSampler.cpp
 *
 *  Created on: Jun 17, 2014
 *      Author: trevorshaddox2
 */



#include "ModelSampler.h"

#include <iterator>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#define PI	3.14159265358979323851280895940618620443274267017841339111328125





namespace bsccs {

ModelSampler::ModelSampler(MCMCModel& model, ModelPrior& prior, long int seed){
	nSamples = 1000;
	srand(seed);
	betaLength = model.getBeta().getSize();
	std::set<int> initialFixedIndices;
	model.setFixedIndices(initialFixedIndices);
	model.syncCCDwithModel(initialFixedIndices);
	string firstKey = model.getModelKey();
	modelProbabilities[firstKey] = getApproximateMarginalLikelihood(model);
	modelVisitCounts[firstKey] = 1;
	keys.insert(firstKey);
	lastKey = firstKey;
	models[firstKey] = initialFixedIndices;
}

ModelSampler::~ModelSampler(){}

vector<SampledModel> ModelSampler::sample(MCMCModel& model, ModelPrior& prior){
	for (int i = 0; i < nSamples; i++) {
		cout << "\t\t\t GIBBS STEP " << i  << endl;
		GibbsStep(model, prior);
	}
	vector<SampledModel> sampledModels = sortResults();
	cout << "sampledModels.size() = " << sampledModels.size() << endl;
	for (int j = 0; j < sampledModels.size(); j++){
		cout << "sampledModels[j].key() = " << sampledModels[j].key << endl;
		cout << "sampledModels[j].visits = " << sampledModels[j].sampledProbability << endl;
	}
//	printHistory();
	//exit(-1);
	return(sampledModels);
}

vector<SampledModel> ModelSampler::sortResults(){
	vector<SampledModel> sampledModels;
	set<string>::iterator iterOut;
	for(iterOut=keys.begin(); iterOut!=keys.end();++iterOut) {
		double test = (modelVisitCounts[(*iterOut)])/(nSamples+0.01);
		SampledModel model((*iterOut), test, modelProbabilities[(*iterOut)]);
		sampledModels.push_back(model);
	}
	std::sort(sampledModels.begin(), sampledModels.end(), compareModels);
	cout << "sampledModels.size() = " << sampledModels.size() << endl;
	return(sampledModels);
}


bool ModelSampler::compareModels(SampledModel model1, SampledModel model2){
	return(model1.sampledProbability > model2.sampledProbability);
}



void ModelSampler::printHistory() {
	cout << "Printing current history" << endl;
	set<string>::iterator iterOut;
	for(iterOut=keys.begin(); iterOut!=keys.end();++iterOut) {
		cout << "Model " << (*iterOut) << endl;
		set<int>::iterator iter;
		for(iter=models[(*iterOut)].begin(); iter!=models[(*iterOut)].end();++iter) {
			cout<<(*iter) << ' ';
		}
		std::cout << endl;
		cout << "Visited " << modelVisitCounts[(*iterOut)] << " times" << endl;
		cout << "Model Probability is " << modelProbabilities[(*iterOut)] << endl;
	}
}

void ModelSampler::GibbsStep (MCMCModel& model, ModelPrior& prior){
	cout << "ModelSelection::GibbsStep" << endl;


	set<int> currentFixedIndices =  models[lastKey];
	model.setFixedIndices(currentFixedIndices);
	double oldPrior = prior.getLogPrior(model);
	double oldProbability = modelProbabilities[lastKey];

	std::set<int> nextFixedIndices = randomWalk(currentFixedIndices);
	model.setFixedIndices(nextFixedIndices);
	model.syncCCDwithModel(currentFixedIndices);
	string newKey = model.getModelKey();
	cout << "lastKey = " << lastKey << endl;
	cout << "newKey = " << newKey << endl;
	const bool is_in = keys.find(newKey) != keys.end();
	if (is_in){
		//model.getModelProbability();
	} else {
		model.syncCCDwithModel(currentFixedIndices);
		keys.insert(newKey);
		modelProbabilities[newKey] = getApproximateMarginalLikelihood(model);
		models[newKey] = nextFixedIndices;
	}

	double newPrior = prior.getLogPrior(model);
	double newProbability = modelProbabilities[newKey];

	double uniformRandom = rand() / ((double) RAND_MAX);
	double epsilon = 0.000000001;
	double ratio = -log(1 + exp(oldPrior + oldProbability - newPrior - newProbability));
	cout << "Gibbs Step Ratio = " << ratio << endl;
	cout << "old = " << oldProbability << endl;
	cout << "uniformRandom = " << log(uniformRandom) << endl;
	cout << "new = " << newProbability << endl;

	if (ratio > log(uniformRandom)) {
		modelComplementIndices.push_back(nextFixedIndices);
		modelVisitCounts[newKey] ++;
		lastKey = newKey;
	} else{
		modelComplementIndices.push_back(currentFixedIndices);
		modelVisitCounts[lastKey] ++;
	}

}

set<int> ModelSampler::getFixedIndices(string key){
	return(models[key]);
}

set<int> ModelSampler::randomWalk(set<int>& currentComplementIndices){

	int randomval = rand() % 2;
	randomval = min(randomval, currentComplementIndices.size());
	if (currentComplementIndices.size() == betaLength){
		randomval = 1;
	}
	if (randomval > 0){
		return(removeComplementIndex(currentComplementIndices));
	} else {
		return(addComplementIndex(currentComplementIndices));
	}
	return(currentComplementIndices);
}

int ModelSampler::min(int int1, int int2){
	if (int1 < int2){
		return(int1);
	} else{
		return(int2);
	}
}

set<int> ModelSampler::addComplementIndex(set<int>& currentComplementIndices){
	set<int> newIndices = currentComplementIndices;
	bool stop = false;
	while(!stop){
		int randomval = rand() % betaLength;
		const bool is_in = currentComplementIndices.find(randomval) != currentComplementIndices.end();
		if(is_in){
			stop = false;
		} else{
			stop = true;
			newIndices.insert(randomval);
		}
	}
	return(newIndices);
}
set<int> ModelSampler::removeComplementIndex(set<int>& currentComplementIndices){
	std::set<int> newIndices = currentComplementIndices;
	bool stop = false;
	std::set<int>::iterator it;
	int randomval = rand() % newIndices.size();
	it = newIndices.begin();
	for (int i = 0; i< randomval; i++){
		++it;
	}
	newIndices.erase(it);
	return(newIndices);
}

double ModelSampler::getApproximateMarginalLikelihood(MCMCModel& model){
	return(getApproximateMarginalLikelihoodLaplace(model));
}

double ModelSampler::getApproximateMarginalLikelihoodLaplace(MCMCModel& model){

	/*
	int betaLength = (model.getBeta()).getSize();
	int numberFixed = model.getFixedSize();

	Eigen::VectorXf betaCurrent(betaLength-numberFixed);
	Eigen::VectorXf precisionProduct_current(betaLength-numberFixed);

	set<int> variableIndices = model.getVariableIndices();
	set<int>::iterator iter;

	int counter = 0;
	for(iter=variableIndices.begin(); iter!=variableIndices.end();++iter) {
		cout << "getApproxML (*iter) = " << (*iter) << endl;
		betaCurrent(counter) = (model.getBeta()).getStored((*iter));
		counter ++;
	}


	cout << "model.getHessian() = " << endl;
	cout << model.getHessian() << endl;
	model.printFixedIndices();
	precisionProduct_current =  model.getHessian() * betaCurrent;

	 */
	double numerator = model.getLogLikelihood();
	double determinant = model.getHessian().determinant();

	// Return approximate log likelihood under the Laplace Approximation
	return(numerator - log(sqrt(determinant/(2*PI))));

	// TODO Check these numbers!
}

}
