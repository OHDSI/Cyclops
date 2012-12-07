/*
 * CompressedDataMatrix.cpp
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <vector>

#include "CompressedDataMatrix.h"

CompressedDataMatrix::CompressedDataMatrix() : nCols(0), nRows(0), nEntries(0) {
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

void CompressedDataColumn::fill(real_vector& values, int nRows) {
	values.resize(nRows);
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
}

CompressedDataMatrix* CompressedDataMatrix::transpose() {
	CompressedDataMatrix* matTranspose = new CompressedDataMatrix();

	matTranspose->nRows = this->getNumberOfColumns();
	int numCols = this->getNumberOfRows();

	bool flagDense = false;
	bool flagIndicator = false;
	bool flagSparse = false;
	for (int i = 0; i < nCols; i++) {
		FormatType thisFormatType = this->allColumns[i]->getFormatType();
		if (thisFormatType == DENSE)
			flagDense = true;
		if (thisFormatType == INDICATOR)
			flagIndicator = true;
	}

	if (flagIndicator && flagDense) {
		flagSparse = true;
		flagIndicator = flagDense = false;
	}
	for (int k = 0; k < numCols; k++) {
		if (flagIndicator) {
			matTranspose->push_back(INDICATOR);
		} else if (flagDense) {
			matTranspose->push_back(DENSE);
		} else {
			matTranspose->push_back(SPARSE);
		}
	}

	for (int i = 0; i < matTranspose->nRows; i++) {
		FormatType thisFormatType = this->allColumns[i]->getFormatType();
		if (thisFormatType == INDICATOR || thisFormatType == SPARSE) {
			int rows = this->getNumberOfEntries(i);
			for (int j = 0; j < rows; j++) {
				if (thisFormatType == SPARSE)
					matTranspose->allColumns[this->getCompressedColumnVector(i)[j]]->add_data(
							i, this->getDataVector(i)[j]);
				else
					matTranspose->allColumns[this->getCompressedColumnVector(i)[j]]->add_data(
							i, 1.0);
			}
		} else {
			for (int j = 0; j < nRows; j++) {
				matTranspose->getColumn(j).add_data(i,
						this->getDataVector(i)[j]);
			}
		}
	}

	return matTranspose;
}

// TODO Fix massive copying
void CompressedDataMatrix::addToColumnVector(int column, int_vector addEntries) const{
	allColumns[column]->addToColumnVector(addEntries);
}

void CompressedDataMatrix::removeFromColumnVector(int column, int_vector removeEntries) const{
	allColumns[column]->removeFromColumnVector(removeEntries);
}


void CompressedDataMatrix::getDataRow(int row, real* x) const {
	for(int j = 0; j < nCols; j++)
	{
		if(this->allColumns[j]->getFormatType() == DENSE)
			x[j] = this->getDataVector(j)[row];
		else{
			x[j] = 0.0;
			int* col = this->getCompressedColumnVector(j);
			for(int i = 0; i < this->allColumns[j]->getNumberOfEntries(); i++){
				if(col[i] == row){
					x[j] = 1.0;
					break;
				}
				else if(col[i] > row)
					break;
			}
		}
	}
}

void CompressedDataMatrix::setNumberOfColumns(int nColumns) {
	nCols = nColumns;
}
// End TODO

void CompressedDataColumn::printColumn(int nRows) {
	real_vector values;
	fill(values, nRows);
	printVector(values.data(), values.size());
}

real CompressedDataColumn::sumColumn(int nRows) {
	real_vector values;
	fill(values, nRows);
	return std::accumulate(values.begin(), values.end(), 0);
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
//	if (formatType == SPARSE) {
//		fprintf(stderr, "Format not yet support.\n");
//		exit(-1);
//	}

	real_vector* oldData = data;	
	data = new real_vector();
	
	data->resize(nRows, static_cast<real>(0));

	int* indicators = getColumns();
	int n = getNumberOfEntries();
//	int nonzero = 0;
	for (int i = 0; i < n; ++i) {
		const int k = indicators[i];
//		cerr << " " << k;
//		nonzero++;

		real value = (formatType == SPARSE) ? oldData->at(i) : 1.0;	

		data->at(k) = value;
	}
//	cerr << endl;
//	cerr << "Non-zero count: " << nonzero << endl;
//	exit(0);
	formatType = DENSE;
	delete columns; columns = NULL;
	if (oldData) {
		delete oldData;
	}
}

// TODO Fix massive copying
void CompressedDataColumn::addToColumnVector(int_vector addEntries){
	int lastit = 0;

	for(int i = 0; i < (int)addEntries.size(); i++)
	{
		int_vector::iterator it = columns->begin() + lastit;
		if(columns->size() > 0){
			while(*it < addEntries[i]){
				it++;
				lastit++;
			}
		}
		columns->insert(it,addEntries[i]);
	}
}

void CompressedDataColumn::removeFromColumnVector(int_vector removeEntries){
	int lastit = 0;
	int_vector::iterator it1 = removeEntries.begin();
	int_vector::iterator it2 = columns->begin();
	while(it1 < removeEntries.end() && it2 < columns->end()){
		if(*it1 < *it2)
			it1++;
		else if(*it2 < *it1){
			it2++;
		}
		else{
			columns->erase(it2);
			it2 = columns->begin() + lastit;
		}
	}
}
