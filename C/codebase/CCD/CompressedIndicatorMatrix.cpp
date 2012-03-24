/*
 * CompressedIndicatorMatrix.cpp
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include "CompressedIndicatorMatrix.h"

CompressedIndicatorMatrix::CompressedIndicatorMatrix() {
	// Do nothing
}

CompressedIndicatorMatrix::CompressedIndicatorMatrix(const char* fileName) {
	
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

CompressedIndicatorMatrix::~CompressedIndicatorMatrix() {
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

void CompressedIndicatorMatrix::convertColumnToSparse(int column) {
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
	const real value = 1.0;
#else
	const real value = 2.0;
#endif

	data[column]->assign(nRows, value);
	formatType[column] = SPARSE;
}

void CompressedIndicatorMatrix::convertColumnToDense(int column) {
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
	data[column]->resize(nRows, static_cast<real>(0));

	int* indicators = getCompressedColumnVector(column);
	int n = getNumberOfEntries(column);
//	int nonzero = 0;
	for (int i = 0; i < n; ++i) {
		const int k = indicators[i];
//		cerr << " " << k;
//		nonzero++;

#if 0
		const real value = 1.0;
#else
		const real value = 2.0;
#endif
		data[column]->at(k) = value;
	}
//	cerr << endl;
//	cerr << "Non-zero count: " << nonzero << endl;
//	exit(0);
	formatType[column] = DENSE;
	delete columns[column]; columns[column] = NULL;
}

int CompressedIndicatorMatrix::getNumberOfRows(void) const {
	return nRows;
}

int CompressedIndicatorMatrix::getNumberOfColumns(void) {
	return nCols;
}

int CompressedIndicatorMatrix::getNumberOfEntries(int column) const {
	return columns[column]->size();
}

int* CompressedIndicatorMatrix::getCompressedColumnVector(int column) const {
	return const_cast<int*>(&(columns[column]->at(0)));
}

real* CompressedIndicatorMatrix::getDataVector(int column) const {
	return const_cast<real*>(data[column]->data());
}

void CompressedIndicatorMatrix::allocateMemory(int nCols) {
	// Allocate some memory
//	columns = std::vector<int_vector*>(nCols);
	columns.resize(nCols);
	for (int j = 0; j < nCols; j++) {
//		columns[j] = int_vector(); // Create empty list
		columns[j] = new int_vector();
	}
}

FormatType CompressedIndicatorMatrix::getFormatType(int column) const {
	return formatType[column];
}
