/*
 * CompressedDataMatrix.h
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 *      
 * This class provides a sparse matrix composed of 0/1 entries only.  Internal representation
 * is Compressed Sparse Column (CSC) format without a 'value' array, since these all equal 1.         
 *     
 */

#ifndef COMPRESSEDINDICATORMATRIX_H_
#define COMPRESSEDINDICATORMATRIX_H_

#include <vector>
#include <iostream>

using namespace std;

#define DEBUG

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif 

typedef std::vector<int> int_vector;
typedef std::vector<real> real_vector;

enum FormatType {
	DENSE, SPARSE, INDICATOR
};

class CompressedDataMatrix {

public:

	CompressedDataMatrix();

	CompressedDataMatrix(const char* fileName);

	virtual ~CompressedDataMatrix();
	
	int getNumberOfRows(void) const;
	
	int getNumberOfColumns(void);

	int getNumberOfEntries(int column) const;

	int* getCompressedColumnVector(int column) const;

	real* getDataVector(int column) const;

	FormatType getFormatType(int column) const;

	void convertColumnToDense(int column);

	void convertColumnToSparse(int column);

	void printColumn(int column);

	template <class T>
	void printVector(T values, const int size) {
		cout << "[" << values[0];
		for (int i = 1; i < size; ++i) {
			cout << " " << values[i];
		}
		cout << "]" << endl;
	}

protected:
	void allocateMemory(int nCols);

	void push_back(int_vector* colIndices, real_vector* colData, FormatType colFormat) {
		columns.push_back(colIndices);
		data.push_back(colData);
		formatType.push_back(colFormat);
	}

	void erase(int column) {
		if (columns[column]) {
			delete columns[column];
		}
		columns.erase(columns.begin() + column);
		if (data[column]) {
			delete data[column];
		}
		data.erase(data.begin() + column);
		formatType.erase(formatType.begin() + column);
	}

//private:
	int nRows;
	int nCols;
	int nEntries;

	std::vector<int_vector*> columns;
	std::vector<real_vector*> data;
	std::vector<FormatType> formatType;

//	std::vector<int> rows;  // standard CSC representation
//	std::vector<int> ptrStart;

};

#endif /* COMPRESSEDINDICATORMATRIX_H_ */
