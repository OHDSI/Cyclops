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


ModelData::ModelData() {
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

real* ModelData::getYVector() { // TODO deprecated
	return makeDeepCopy(&y[0], y.size());
}

int* ModelData::getNEventVector() { // TODO deprecated
	return makeDeepCopy(&nevents[0], nevents.size());
}

int* ModelData::getOffsetVector() { // TODO deprecated
	return makeDeepCopy(&offs[0], offs.size());
}

map<int, DrugIdType> ModelData::getDrugNameMap() {
	return indexToDrugIdMap;
}
