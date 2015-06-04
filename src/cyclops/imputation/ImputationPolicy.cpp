/*
 * ImputationHelper.cpp
 *
 *  Created on: Jul 28, 2012
 *      Author: Sushil Mittal
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <time.h>

#include "ImputationPolicy.h"

#define MAX_ENTRIES		1000000000

#define FORMAT_MATCH_1	"CONDITION_CONCEPT_ID"
#define MATCH_LENGTH_1	20
#define FORMAT_MATCH_2	"PID"
#define MATCH_LENGTH_2	3
#define MISSING_STRING	"NA"
#define MISSING_LENGTH	-1
#define DELIMITER		","

namespace bsccs {

using namespace std;

ImputationHelper::ImputationHelper(){
	nMissingY = 0; 
	nCols_Orig = 0;
}

ImputationHelper::~ImputationHelper() { }

/**
 * Reads in a dense CSV data file with format:
 * Stratum,Outcome,X1 ...
 *
**/

template <class InputIterator1, class InputIterator2>
int set_intersection(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2)
{
	int result = 0;
	while (first1!=last1 && first2!=last2)
	{
		if(*first1 < *first2) 
			++first1;
		else if(*first2 < *first1) 
			++first2;
		else{ 
			result++;
			first1++; 
			first2++; 
		}
	}
	return result;
}

template <class InputIterator>
int set_difference(int* columns, int length, InputIterator first2, InputIterator last2)
{
	int result = length;
	int i = 0;
	while(i < length && first2 != last2)
	{
		if(columns[i] < *first2)
			i++;
		else if(*first2 < columns[i]) 
			++first2;
		else{ 
			result--;
			i++;
			first2++; 
		}
	}
	return result;
}
void ImputationHelper::sortColumns(){
	colIndices.clear();
	srand(time(NULL));
	vector<int> rands;
	for(int i = 0; i < nCols_Orig; i++){
		colIndices.push_back(i);
		rands.push_back(rand());
	}
	sort(colIndices.begin(),colIndices.end(),Compare(nMissingPerColumn,rands));

	reverseColIndices.resize(nCols_Orig,0);
	for(int i = 0; i < nCols_Orig; i++)
		reverseColIndices[colIndices[i]] = i;

	reindexVector(nMissingPerColumn,colIndices);

	vector<int_vector*> missingEntries_ = missingEntries;
	for(int i = 0; i < nCols_Orig; i++){
		missingEntries[i] = missingEntries_[colIndices[i]];
	}
}

void ImputationHelper::resortColumns(){
	reindexVector(nMissingPerColumn,reverseColIndices);

	vector<int_vector*> missingEntries_ = missingEntries;
	for(int i = 0; i < nCols_Orig; i++){
		missingEntries[i] = missingEntries_[reverseColIndices[i]];
	}
	
	colIndices.clear();
	
	for(int i = 0; i < nCols_Orig; i++)
		colIndices.push_back(i);

	reverseColIndices = colIndices;
}

void ImputationHelper::push_back(int_vector* vecMissing, int nMissing){
	missingEntries.push_back(vecMissing);
	nMissingPerColumn.push_back(nMissing);
	nCols_Orig++;
}

void ImputationHelper::push_back(int col, int indMissing){
	missingEntries[col]->push_back(indMissing);
	nMissingPerColumn[col]++;
}

void ImputationHelper::push_backY(int indMissing){
	missingEntriesY.push_back(indMissing);
	nMissingY++;
}

void ImputationHelper::includeYVector(){
	push_back(&missingEntriesY,missingEntriesY.size());
}

void ImputationHelper::pop_back(){
	missingEntries.pop_back();
	nMissingPerColumn.pop_back();
	nCols_Orig--;
}

const vector<int>& ImputationHelper::getnMissingPerColumn() const{
	return nMissingPerColumn;
}

const vector<int>& ImputationHelper::getSortedColIndices() const{
	return colIndices;
}

const vector<int>& ImputationHelper::getReverseColIndices() const{
	return reverseColIndices;
}

void ImputationHelper::setWeightsForImputation(int col, vector<real>& weights, int nRows){
	weights.clear();
	weights.resize(nRows,1.0);
	for(int i = 0; i < (int)missingEntries[col]->size(); i++)
		weights[(missingEntries[col])->at(i)] = 0.0;
}

int ImputationHelper::getOrigNumberOfColumns(){
	return nCols_Orig;
}

vector<real> ImputationHelper::getOrigYVector(){
	return y_Orig;
}

void ImputationHelper::saveOrigYVector(real* y, int nRows){
	y_Orig.resize(nRows,0.0);
	for(int i = 0; i < nRows; i++)
		y_Orig[i] = y[i];
}

void ImputationHelper::saveOrigNumberOfColumns(int nCols){
	nCols_Orig = nCols;
}

void ImputationHelper::getMissingEntries(int col, vector<int>& missing){
	missing = *missingEntries[col];
}

void ImputationHelper::getSampleMeanVariance(int col, real& Xmean, real& Xvar, real* dataVec, int* columnVec, FormatType formatType, int nRows, int nEntries){
	real sumx2 = 0.0;
	real sumx = 0.0;
	int n = nRows - (int)missingEntries[col]->size();
	if(formatType == DENSE) {
		int ind = 0;
		for(int i = 0; i < nRows; i++){
			if(ind < (int)missingEntries[col]->size()){
				if(i == missingEntries[col]->at(ind)){
					ind++;
				}
				else{
					real xi = dataVec[i];
					sumx2 += xi * xi;
					sumx += xi;
				}
			}
			else{
				real xi = dataVec[i];
				sumx2 += xi * xi;
				sumx += xi;
			}
		}
	}
	else{
		sumx = set_difference(columnVec,nEntries, missingEntries[col]->begin(), missingEntries[col]->end());
		sumx2 = sumx;
	}
	Xmean = sumx/n;
	Xvar =  (sumx2 - Xmean*Xmean*n)/(n-1);
}

} // namespace
