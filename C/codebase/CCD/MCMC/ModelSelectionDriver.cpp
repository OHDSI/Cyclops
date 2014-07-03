/*
 * ModelSelectionDriver.cpp
 *
 *  Created on: Jun 26, 2014
 *      Author: trevorshaddox2
 */


#include "ModelSelectionDriver.h"

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#define PI	3.14159265358979323851280895940618620443274267017841339111328125

namespace bsccs {

ModelSelectionDriver::ModelSelectionDriver(){


}

ModelSelectionDriver::~ModelSelectionDriver(){}


void ModelSelectionDriver::drive(CyclicCoordinateDescent& ccd, long int seed, string MCMCFilename, double betaAmount){


	MCMCModel model;
	ModelPrior modelPrior;

	model.initialize(ccd, seed);

	ModelSampler sampler(model, modelPrior, seed);
	MCMCDriver driver(MCMCFilename);
	driver.initialize(betaAmount, ccd);

	vector<SampledModel> sampledModels = sampler.sample(model, modelPrior);

	double epsilon = 0.01;
	int nSampledModels = sampledModels.size();
	bool continueLoop = true;
	int counter = 0;
	while(continueLoop){
		if (sampledModels[counter].sampledProbability < epsilon){
			break;
		}
		string fixedIndicesKey = sampledModels[counter].key;
		set<int> theseFixedIndices = sampler.getFixedIndices(fixedIndicesKey);
		model.setFixedIndices(theseFixedIndices);
		model.syncCCDwithModel(theseFixedIndices);
		driver.drive(model, ccd, betaAmount, seed, sampledModels[counter].sampledProbability);
		counter ++;
		cout << "counter = " << counter << endl;
		cout << "key = " << fixedIndicesKey << endl;
	}




	cout << "End ModelSelectionDriver::drive" << endl;
}

}

