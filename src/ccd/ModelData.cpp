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
#include <list>

#include "ModelData.h"

namespace bsccs {

using std::string;
using std::vector;

ModelData::ModelData() : nPatients(0), nStrata(0), hasOffsetCovariate(false), hasInterceptCovariate(false) {
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

	for (size_t index = startIndex; index < getNumberOfColumns(); ++index) {
		squaredNorm.push_back(getColumn(index).squaredSumColumn());
	}

	return std::accumulate(squaredNorm.begin(), squaredNorm.end(), 0.0);
}

size_t ModelData::getNumberOfStrata() const {
    typedef std::list<int>  List;
    if (nStrata == 0) {
        List cPid(pid.begin(), pid.end());
        List::iterator pos = std::unique(cPid.begin(), cPid.end());
        nStrata = std::distance(pos, cPid.begin());        
    }
    return nStrata;
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
