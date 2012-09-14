/*
 * SparseRowVector.h
 *
 *  Created on: Jul 14, 2012
 *      Author: trevorshaddox
 */

#ifndef SPARSEROWVECTOR_H_
#define SPARSEROWVECTOR_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>

#include "InputReader.h"

using std::cout;
using std::cerr;
using std::endl;
//using std::in;
using std::ifstream;

namespace bsccs {

typedef std::vector<int> int_vector;
typedef std::vector<bsccs::real> real_vector;


class SparseRowVector {

public:

	SparseRowVector();

	void fillSparseRowVector(CompressedDataMatrix* columnData);

	virtual ~SparseRowVector();

	void printSparseMatrix();

	int getNumberOfRows();

	int getNumberOfColumns();

	int getNumberOfEntries(int column);

	int * getNumberOfEntriesList() const;

	FormatType getFormatType();

	void printRow(int row);

	int * getCompressedRowVector(int row) const;

	bsccs::real* getDataVector(int row) const;

	void setChangedStatus(bool changeSetting);

	bool getChangedStatus();

	void setUseThisStatus(bool useTransposeSetting);

	bool getUseThisStatus();

	template <class T>
	void printVector(T values, const int size) {
		cout << "[" << values[0];
		for (int i = 1; i < size; ++i) {
			cout << " " << values[i];
		}
		cout << "]" << endl;
	}

protected:

	int nTransposeRows;
	int nTransposeCols;
	int nEntries;

	int_vector* listCompressedVectorLengths;

	std::vector<int_vector*> matrixTransposeIndicator;

	std::vector<real_vector*> matrixTransposeDense;

	void transposeIndicator(CompressedDataMatrix* columnData);

	void transposeDense(CompressedDataMatrix* columnData);

	FormatType formatType;

	bool hasChanged;

	bool useTransposeMatrix;

};

}
#endif /* SPARSEROWVECTOR_H_ */
