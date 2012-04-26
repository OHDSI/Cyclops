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

#include <cstdlib>
#include <vector>
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
//using std::in;
using std::ifstream;

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

	real sumColumn(int column);

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

	void push_back(FormatType colFormat) {
		if (colFormat == DENSE) {
			real_vector* r = new real_vector();
			push_back(NULL, r, DENSE);
		} else if (colFormat == SPARSE) {
			real_vector* r = new real_vector();
			int_vector* i = new int_vector();
			push_back(i, r, SPARSE);
		} else if (colFormat == INDICATOR) {
			int_vector* i = new int_vector();
			push_back(i, NULL, INDICATOR);
		} else {
			cerr << "Error" << endl;
			exit(-1);
 		}
	}

	void add_data(int column, int row, real value) {
		FormatType colFormat = getFormatType(column);
		if (colFormat == DENSE) {
			data[column]->push_back(value);
		} else if (colFormat == SPARSE) {
			if (value != static_cast<real>(0)) {
				data[column]->push_back(value);
				columns[column]->push_back(row);
			}
		} else if (colFormat == INDICATOR) {
			if (value != static_cast<real>(0)) {
				columns[column]->push_back(row);
			}
		} else {
			cerr << "Error" << endl;
			exit(-1);
		}
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
