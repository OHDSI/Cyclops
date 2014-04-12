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
#include <numeric>

#include "ModelData.h"

namespace bsccs {

ModelData::ModelData() : hasOffsetCovariate(false), hasInterceptCovariate(false), nPatients(0) {
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
//	return makeDeepCopy(&pid[0], pid.size());
	return &pid[0];
}

std::vector<int>* ModelData::getPidVectorSTL() { // TODO deprecated
	return new std::vector<int>(pid);
}

real* ModelData::getYVector() { // TODO deprecated
//	return makeDeepCopy(&y[0], y.size());
	return &y[0];
}

void ModelData::setYVector(vector<real> y_){
	y = y_;
}

//int* ModelData::getNEventVector() { // TODO deprecated
////	return makeDeepCopy(&nevents[0], nevents.size());
//	return &nevents[0];
//}

real* ModelData::getOffsetVector() { // TODO deprecated
//	return makeDeepCopy(&offs[0], offs.size());
	return &offs[0];
}

void ModelData::sortDataColumns(vector<int> sortedInds){
	reindexVector(allColumns,sortedInds);
}

double ModelData::getSquaredNorm() const {

	int startIndex = 0;
	if (hasInterceptCovariate) ++startIndex;
	if (hasOffsetCovariate) ++startIndex;

	std::vector<double> squaredNorm;

	for (int index = startIndex; index < getNumberOfColumns(); ++index) {
		squaredNorm.push_back(getColumn(index).squaredSumColumn());
	}

	return std::accumulate(squaredNorm.begin(), squaredNorm.end(), 0.0);
}

double ModelData::getNormalBasedDefaultVar() const {
	return getNumberOfVariableColumns() * getNumberOfRows() / getSquaredNorm();
}

int ModelData::getNumberOfVariableColumns() const {
	int dim = getNumberOfColumns();
	if (hasInterceptCovariate) --dim;
	if (hasOffsetCovariate) --dim;
	return dim;
}

const string ModelData::missing = "NA";

} // namespace
