/*
 * BootstrapSelector.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <cstdlib>
#include <iostream>

#include "BootstrapSelector.h"

BootstrapSelector::BootstrapSelector(
		int replicates,
		const std::vector<int>& inIds,
		SelectorType inType,
		long inSeed,
		std::vector<real>* wtsExclude) : AbstractSelector(inIds, inType, inSeed) {

	std::cout << "Performing bootstrap estimation with " << replicates
		<< " replicates [seed = " << seed << "]" << std::endl;

	if(wtsExclude){
		for(int i = 0; i < wtsExclude->size(); i++){
			if(wtsExclude->at(i) == 0){
				indicesIncluded.push_back(i);
			}
		}
	}
	else{
		for(int i = 0; i < N; i++){
			indicesIncluded.push_back(i);
		}
	}

	permute();

//	exit(0);
}

BootstrapSelector::~BootstrapSelector() {
	// Nothing to do
}

void BootstrapSelector::permute() {
	selectedSet.clear();

	// Get non-excluded indices
	int N_new = indicesIncluded.size();
	if (type == SUBJECT) {
		for (int i = 0; i < N_new; i++) {
			int ind = rand() / (RAND_MAX / N_new + 1);
			int draw = indicesIncluded[ind];
			selectedSet.insert(draw);
		}
	} else {
		std::cerr << "BootstrapSelector::permute is not yet implemented." << std::endl;
		exit(-1);
	}

//	int total = 0;
//	for (int i = 0; i < N; i++) {
//		int count = selectedSet.count(i);
//		std::cout << i << " : " << count << std::endl;
//		total += count;
//	}
//	std::cout << "Total = " << total << std::endl;
//	exit(0);
}

void BootstrapSelector::getWeights(int batch, std::vector<real>& weights) {
	if (weights.size() != K) {
		weights.resize(K);
	}

	std::fill(weights.begin(), weights.end(), 0.0);
	if (batch == -1) {
		return;
	}

	if (type == SUBJECT) {
		for (int k = 0; k < K; k++) {
			int count = selectedSet.count(ids.at(k));
			weights[k] = static_cast<real>(count);
		}
	} else {
		std::cerr << "BootstrapSelector::getWeights is not yet implemented." << std::endl;
		exit(-1);
	}
}

void BootstrapSelector::getComplement(std::vector<real>& weights) {
	std::cerr << "BootstrapSelector::getComplement is not yet implemented." << std::endl;
	exit(-1);
}
