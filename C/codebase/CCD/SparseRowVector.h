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

class SparseRowVector {

public:

	SparseRowVector();

	void fillSparseRowVector(CompressedDataMatrix* columnData);

	virtual ~SparseRowVector();

	int getNumberOfRows();

	int getNumberOfColumns();

	int getNumberOfEntries(int column);

	void printRow(int row);

	int * getCompressedRowVector(int row) const;

	template <class T>
	void printVector(T values, const int size) {
		cout << "[" << values[0];
		for (int i = 1; i < size; ++i) {
			cout << " " << values[i];
		}
		cout << "]" << endl;
	}

protected:

//private:
	int nTransposeRows;
	int nTransposeCols;
	int nEntries;

	std::vector<int_vector*> matrixTranspose;

//	std::vector<int> rows;  // standard CSC representation
//	std::vector<int> ptrStart;

};

}
#endif /* SPARSEROWVECTOR_H_ */
