/*
 * ModelSampler.h
 *
 *  Created on: Jun 17, 2014
 *      Author: trevorshaddox2
 */

#ifndef MODELSAMPLER_H_
#define MODELSAMPLER_H_

#include <unordered_map>
#include <iostream>
#include <numeric>
#include <vector>
#include <set>
#include <math.h>
#include <cstdlib>
#include <time.h>

#include "MCMCModel.h"
#include "ModelPrior.h"


namespace bsccs {

struct SampledModel
{
    std::string key;
    double sampledProbability;
    double probability;

    SampledModel(const std::string& keyIn, double sampledProbCalc, double prob) : key(keyIn), sampledProbability(sampledProbCalc), probability(prob) {}
};

class ModelSampler {
public:


	ModelSampler(MCMCModel& model, ModelPrior& prior, long int seed);

	virtual ~ModelSampler();

	vector<SampledModel> sample(MCMCModel& model, ModelPrior& prior);

	vector<SampledModel> sortResults();

	void GibbsStep(MCMCModel& model, ModelPrior& prior);

	double getApproximateMarginalLikelihood(MCMCModel& model);

	set<int> getFixedIndices(string key);


private:

	static bool compareModels(SampledModel model1, SampledModel model2);

	//bool compareKeys(string key1, string key2);

	set<int> randomWalk(set<int>& currentComplementIndices);

	set<int> addComplementIndex(set<int>& currentComplementIndices);
	set<int> removeComplementIndex(set<int>& currentComplementIndices);

	double getApproximateMarginalLikelihoodLaplace(MCMCModel& model);
	void printHistory();


	int min(int int1, int int2);

	int betaLength;
	int nSamples;
	std::vector< std::set<int> > modelComplementIndices;
	string lastKey;
	set<string> keys;
	unordered_map <string, set<int>> models;
	unordered_map <string, double> modelProbabilities;
	unordered_map <string, int> modelVisitCounts;




};
}



#endif /* MODELSAMPLER_H_ */
