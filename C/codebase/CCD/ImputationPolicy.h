/*
 * ImputateInputReader.h
 *
 *  Created on: Jul 28, 2012
 *      Author: Sushil Mittal
 */

#ifndef ImputationHelper_H_
#define ImputationHelper_H_

#include "io/InputReader.h"

class Compare{
	vector<int>& _vec;
public:
	Compare(vector<int>& vec) : _vec(vec) {}
	bool operator()(size_t i, size_t j){
		return _vec[i] < _vec[j];
	}
};

class ImputationHelper{
public:
	ImputationHelper();
	virtual ~ImputationHelper();

	void sortColumns();
	void resortColumns();
	void push_back(int_vector* vecAbsent,int valMissing);
	void push_back(int col, int indAbsent);
	const vector<int>& getnMissingPerColumn() const;
	const vector<int>& getSortedColIndices() const;
	const vector<int>& getReverseColIndices() const;
	void setWeightsForImputation(int col, vector<real>& weights, int nRows);
	void saveOrigYVector(real* y, int nRows);
	void saveOrigNumberOfColumns(int nCols);
	int getOrigNumberOfColumns();
	vector<real> getOrigYVector();
	void getMissingEntries(int col, vector<int>& missing);
	void getSampleMeanVariance(int col, real& Xmean, real& Xvar, real* dataVec, int* columnVec, FormatType formatType, int nRows, int nEntries);
protected:
	vector<int> nMissingPerColumn;
	vector<int> colIndices;
	vector<int> reverseColIndices;
	vector<real> y_Orig;
	vector<int_vector*> missingEntries;

	int nCols_Orig;
};


class NoImputation{
public:
	NoImputation() {}
	virtual ~NoImputation() {}

	void sortColumns() {}
	void resortColumns() {}
	void push_back(int_vector* vecAbsent,int valMissing) {}
	void push_back(int col, int indAbsent) {}
	const vector<int>& getnMissingPerColumn() const {}
	const vector<int>& getSortedColIndices() const {}
	const vector<int>& getReverseColIndices() const {}
	void setWeightsForImputation(int col, vector<real>& weights, int nRows) {}
	void saveOrigYVector(real* y, int nRows) {}
	void saveOrigNumberOfColumns(int nCols) {}
	int getOrigNumberOfColumns() {}
	vector<real> getOrigYVector() {}
	void getMissingEntries(int col, vector<int>& missing) {}
	void getSampleMeanVariance(int col, real& Xmean, real& Xvar, real* dataVec, int* columnVec, FormatType formatType, int nRows, int nEntries) {}
};

#endif /* ImputationHelper_H_ */
