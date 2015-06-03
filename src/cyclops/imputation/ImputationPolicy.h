/*
 * ImputateInputReader.h
 *
 *  Created on: Jul 28, 2012
 *      Author: Sushil Mittal
 */

#ifndef ImputationHelper_H_
#define ImputationHelper_H_

#include "io/InputReader.h"

namespace bsccs {

class Compare{
	vector<int>& _vec;
	vector<int>& _rands;
public:
	Compare(vector<int>& vec, vector<int>& rands) : _vec(vec), _rands(rands) {}
	bool operator()(size_t i, size_t j){
		if(_vec[i] != _vec[j])
			return _vec[i] < _vec[j];
		else
			return _rands[i] < _rands[j];
	}
};

class ImputationHelper{
public:
	ImputationHelper();
	virtual ~ImputationHelper();

	void sortColumns();
	void resortColumns();
	void push_back(IntVector* vecMissing,int nMissing);
	void push_back(int col, int indMissing);
	void push_backY(int indMissing);
	void includeYVector();
	void pop_back();
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
	vector<int> missingEntriesY;
	int nMissingY;
	vector<int> nMissingPerColumn;
	vector<int> colIndices;
	vector<int> reverseColIndices;
	vector<real> y_Orig;
	vector<IntVector*> missingEntries;

	int nCols_Orig;
};

class NoImputation{
public:
	NoImputation() {}
	virtual ~NoImputation() {}

	void sortColumns() {}
	void resortColumns() {}
	void push_back(IntVector* vecMissing,int nMissing) {}
	void push_back(int col, int indMissing) {}
	void push_backY(int indMissing) {}
	void includeY() {}
	void pop_back() {}
//	const vector<int>& getnMissingPerColumn() const {}
//	const vector<int>& getSortedColIndices() const {}
//	const vector<int>& getReverseColIndices() const {}
	void setWeightsForImputation(int col, vector<real>& weights, int nRows) {}
	void saveOrigYVector(real* y, int nRows) {}
	void saveOrigNumberOfColumns(int nCols) {}
	int getOrigNumberOfColumns() { return 0; }
//	vector<real> getOrigYVector() {}
	void getMissingEntries(int col, vector<int>& missing) {}
	void getSampleMeanVariance(int col, real& Xmean, real& Xvar, real* dataVec, int* columnVec, FormatType formatType, int nRows, int nEntries) {}
};

} // namespace

#endif /* ImputationHelper_H_ */
