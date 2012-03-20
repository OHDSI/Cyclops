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
		columns[j].push_back(i);
	}
	
	// Sort all columns, just in case MatrixMarket file is corrupted
	for (int j = 0; j < nCols; j++) {
		std::sort(columns[j].begin(), columns[j].end());
	}
		
#ifdef DEBUG
	cerr << "Read in sparse indicator matrix from " << fileName << endl;
	cerr << "Spare matrix dimensions = " << nRows << " x " << nCols << endl;
	cerr << "Number of non-zero elements = " << nEntries << endl;	
#endif
	
}

CompressedIndicatorMatrix::~CompressedIndicatorMatrix() {
	// Do nothing
}

int CompressedIndicatorMatrix::getNumberOfRows(void) const {
	return nRows;
}

int CompressedIndicatorMatrix::getNumberOfColumns(void) {
	return nCols;
}

int CompressedIndicatorMatrix::getNumberOfEntries(int column) const {
	return columns[column].size();
}

int* CompressedIndicatorMatrix::getCompressedColumnVector(int column) const {
	return const_cast<int*>(&(columns[column])[0]);
}

void CompressedIndicatorMatrix::allocateMemory(int nCols) {
	// Allocate some memory
	columns = std::vector<int_vector>(nCols);
	for (int j = 0; j < nCols; j++) {
		columns[j] = int_vector(); // Create empty list
	}
}
