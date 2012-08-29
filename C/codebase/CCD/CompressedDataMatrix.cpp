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

CompressedDataMatrix::CompressedDataMatrix() {
	// Do nothing
}

CompressedDataMatrix::~CompressedDataMatrix() {
	typedef std::vector<CompressedDataColumn*>::iterator CIterator;
	for (CIterator it = allColumns.begin(); it != allColumns.end(); ++it) {
		delete *it;
	}
}

real CompressedDataMatrix::sumColumn(int column) {
	real sum = 0.0;
	if (getFormatType(column) == DENSE) {
		cerr << "Not yet implemented (DENSE)." << endl;
		exit(-1);
	} else if (getFormatType(column) == SPARSE) {
		cerr << "Not yet implemented (SPARSE)." << endl;
		exit(-1);
	} else { // is indiciator
		sum = allColumns[column]->getNumberOfEntries();
	}
	return sum;
}

void CompressedDataMatrix::printColumn(int column) {
#if 1
	cerr << "Not yet implemented.\n";
	exit(-1);
#else
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
#endif
}

void CompressedDataMatrix::convertColumnToSparse(int column) {
	allColumns[column]->convertColumnToSparse();
}

void CompressedDataMatrix::convertColumnToDense(int column) {
	allColumns[column]->convertColumnToDense(nRows);
}

int CompressedDataMatrix::getNumberOfRows(void) const {
	return nRows;
}

int CompressedDataMatrix::getNumberOfColumns(void) const {
	return nCols;
}

int CompressedDataMatrix::getNumberOfEntries(int column) const {
	return allColumns[column]->getNumberOfEntries();
}

int* CompressedDataMatrix::getCompressedColumnVector(int column) const {
	return allColumns[column]->getColumns();
}

real* CompressedDataMatrix::getDataVector(int column) const {
	return allColumns[column]->getData();
}

FormatType CompressedDataMatrix::getFormatType(int column) const {
	return allColumns[column]->getFormatType();
}

void CompressedDataColumn::printColumn(int nRows) {
	real_vector values;
	if (formatType == DENSE) {
		values.assign(data->begin(), data->end());
	} else {
		bool isSparse = formatType == SPARSE;
		values.assign(nRows, 0.0);
		int* indicators = getColumns();
		int n = getNumberOfEntries();
		for (int i = 0; i < n; ++i) {
			const int k = indicators[i];
			if (isSparse) {
				values[k] = data->at(i);
			} else {
				values[k] = 1.0;
			}
		}
	}
	printVector(values.data(), values.size());
}

void CompressedDataColumn::convertColumnToSparse(void) {
	if (formatType == SPARSE) {
		return;
	}
	if (formatType == DENSE) {
		fprintf(stderr, "Format not yet support.\n");
		exit(-1);
	}

	if (data == NULL) {
		data = new real_vector();
	}

	const real value = 1.0;

	data->assign(getNumberOfEntries(), value);
	formatType = SPARSE;
}

void CompressedDataColumn::convertColumnToDense(int nRows) {
	if (formatType == DENSE) {
		return;
	}
	if (formatType == SPARSE) {
		fprintf(stderr, "Format not yet support.\n");
		exit(-1);
	}

	if (data == NULL) {
		data = new real_vector();
	}
	data->resize(nRows, static_cast<real>(0));

	int* indicators = getColumns();
	int n = getNumberOfEntries();
//	int nonzero = 0;
	for (int i = 0; i < n; ++i) {
		const int k = indicators[i];
//		cerr << " " << k;
//		nonzero++;

		const real value = 1.0;

		data->at(k) = value;
	}
//	cerr << endl;
//	cerr << "Non-zero count: " << nonzero << endl;
//	exit(0);
	formatType = DENSE;
	delete columns; columns = NULL;
}
