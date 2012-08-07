/*
 * CompressedDataMatrix.cpp
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include "CompressedDataMatrix.h"

namespace bsccs {

CompressedDataMatrix::CompressedDataMatrix() {
	// Do nothing
}

CompressedDataMatrix::CompressedDataMatrix(const char* fileName) {
	ifstream in(fileName);	
	if (!in) {
		cerr << "Unable to open " << fileName << endl;
		exit(-1);
	}
	
	// Read header line
	char buffer[256];
	in.getline(buffer, 256);
	
	// Read matrix dimensions
	in >> nRows >> nCols >> nEntries;
	
//	// Allocate some memory
//	columns = std::vector<int_vector>(nCols);
//	for (int j = 0; j < nCols; j++) {
//		columns[j] = int_vector(); // Create empty list
//	}
	allocateMemory(nCols);
	
	// Read each matrix entry
	for (int k = 0; k < nEntries; k++) {
		int i, j;
		double x;
		in >> i >> j >> x;
		i--; // C uses 0-indices, MatrixMarket uses 1-indices
		j--;	
		if (x != 1) {
			cerr << "Non-zero/one element in matrix." << endl;
			exit(-1);
		}
		columns[j]->push_back(i);
	}
	
	// Sort all columns, just in case MatrixMarket file is corrupted
	for (int j = 0; j < nCols; j++) {
		std::sort(columns[j]->begin(), columns[j]->end());
	}
		
#ifdef DEBUG
	cerr << "Read in sparse indicator matrix from " << fileName << endl;
	cerr << "Spare matrix dimensions = " << nRows << " x " << nCols << endl;
	cerr << "Number of non-zero elements = " << nEntries << endl;	
#endif
	
}

CompressedDataMatrix::~CompressedDataMatrix() {
	typedef std::vector<real_vector*>::iterator RIterator;
	for (RIterator it = data.begin(); it != data.end(); ++it) {
		if (*it) {
			delete *it;
		}
	}

	typedef std::vector<int_vector*>::iterator IIterator;
	for (IIterator it = columns.begin(); it != columns.end(); ++it) {
		if (*it) {
			delete *it;
		}
	}
}

bsccs::real CompressedDataMatrix::sumColumn(int column) {
	bsccs::real sum = 0.0;
	if (getFormatType(column) == DENSE) {
		cerr << "Not yet implemented (DENSE)." << endl;
		exit(-1);
	} else if (getFormatType(column) == SPARSE) {
		cerr << "Not yet implemented (SPARSE)." << endl;
		exit(-1);
	} else { // is indiciator
		sum = columns[column]->size();
	}
	return sum;
}

void CompressedDataMatrix::printColumn(int column) {
	real_vector values;
	if (getFormatType(column) == DENSE) {
		values.assign(data[column]->begin(), data[column]->end());
	} else {
		bool isSparse = getFormatType(column) == SPARSE;
		values.assign(nRows, 0.0);
		int* indicators = getCompressedColumnVector(column);
		int n = getNumberOfEntries(column);
		for (int i = 0; i < n; ++i) {
			const int k = indicators[i];
			if (isSparse) {
				values[k] = data[column]->at(i);
			} else {
				values[k] = 1.0;
			}
		}
	}
	printVector(values.data(), values.size());
}

//template <class T>
//void CompressedDataMatrix::printVector(T values, const int size) {
//	cout << "[" << values[0];
//	for (int i = 1; i < size; ++i) {
//		cout << " " << values[i];
//	}
//	cout << "]" << endl;
//}



void CompressedDataMatrix::convertColumnToSparse(int column) {
	if (getFormatType(column) == SPARSE) {
		return;
	}
	if (getFormatType(column) == DENSE) {
		fprintf(stderr, "Format not yet support.\n");
		exit(-1);
	}

	while (data.size() <= column) {
		data.push_back(NULL);
	}
	if (data[column] == NULL) {
		data[column] = new real_vector();
	}

#if 1
	const bsccs::real value = 1.0;
#else
	const bsccs::real value = 2.0;
#endif

	data[column]->assign(nRows, value);
	formatType[column] = SPARSE;
}

void CompressedDataMatrix::convertColumnToDense(int column) {
	if (getFormatType(column) == DENSE) {
		return;
	}
	if (getFormatType(column) == SPARSE) {
		fprintf(stderr, "Format not yet support.\n");
		exit(-1);
	}

	while (data.size() <= column) {
		data.push_back(NULL);
	}
	if (data[column] == NULL) {
		data[column] = new real_vector();
	}
	data[column]->resize(nRows, static_cast<bsccs::real>(0));

	int* indicators = getCompressedColumnVector(column);
	int n = getNumberOfEntries(column);
//	int nonzero = 0;
	for (int i = 0; i < n; ++i) {
		const int k = indicators[i];
//		cerr << " " << k;
//		nonzero++;

#if 0
		const bsccs::real value = 1.0;
#else
		const bsccs::real value = 2.0;
#endif
		data[column]->at(k) = value;
	}
//	cerr << endl;
//	cerr << "Non-zero count: " << nonzero << endl;
//	exit(0);
	formatType[column] = DENSE;
	delete columns[column]; columns[column] = NULL;
}

int CompressedDataMatrix::getNumberOfRows(void) const {
	return nRows;
}

int CompressedDataMatrix::getNumberOfColumns(void) {
	return nCols;
}

int CompressedDataMatrix::getNumberOfEntries(int column) const {
	return columns[column]->size();
}

int* CompressedDataMatrix::getCompressedColumnVector(int column) const {
	return const_cast<int*>(&(columns[column]->at(0)));
}

bsccs::real* CompressedDataMatrix::getDataVector(int column) const {
	return const_cast<bsccs::real*>(data[column]->data());
}

void CompressedDataMatrix::allocateMemory(int nCols) {
	// Allocate some memory
//	columns = std::vector<int_vector*>(nCols);
	columns.resize(nCols);
	for (int j = 0; j < nCols; j++) {
//		columns[j] = int_vector(); // Create empty list
		columns[j] = new int_vector();
	}
}

FormatType CompressedDataMatrix::getFormatType(int column) const {
	return formatType[column];
}
}
