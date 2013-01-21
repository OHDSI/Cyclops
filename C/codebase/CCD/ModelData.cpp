/*
 * ModelData.cpp
 *
 *  Created on: August, 2012
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "ModelData.h"


ModelData::ModelData() : hasOffsetCovariate(false) {
	// Do nothing
}

int ModelData::getNumberOfPatients() {
	return nPatients;
}

string ModelData::getConditionId() {
	return conditionId;
}

ModelData::~ModelData() {
	// Do nothing
}

int* ModelData::getPidVector() { // TODO deprecated	
	return makeDeepCopy(&pid[0], pid.size());
}

std::vector<int>* ModelData::getPidVectorSTL() { // TODO deprecated
	return new std::vector<int>(pid);
}

std::map<int, int> ModelData::getNumericalLabelsMap() {
	std::map<int, int> labelMap;
	for(int i = 0; i < nCols; i++) {
		int label = getColumn(i).getNumericalLabel();
		labelMap.insert(std::make_pair(label, i));
	}
	return labelMap;
}

real* ModelData::getYVector() { // TODO deprecated
	return makeDeepCopy(&y[0], y.size());
}

void ModelData::setYVector(vector<real> y_){
	y = y_;
}

int* ModelData::getNEventVector() { // TODO deprecated
	return makeDeepCopy(&nevents[0], nevents.size());
}

int* ModelData::getOffsetVector() { // TODO deprecated
	return makeDeepCopy(&offs[0], offs.size());
}

void ModelData::sortDataColumns(vector<int> sortedInds){
	reindexVector(allColumns,sortedInds);
}
