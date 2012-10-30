/*
 * SparseRowVector.cpp
 *
 *  Created on: Jul 14, 2012
 *      Author: trevorshaddox
 */

/*
 * SparseRowVector.cpp
 *
 *  Created on: July, 2012
 *      Author: tshaddox
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>

#include "SparseRowVector.h"
#include "CompressedDataMatrix.h"
#include "InputReader.h"


namespace bsccs {

SparseRowVector::SparseRowVector() {
	// Do nothing
}

void SparseRowVector::fillSparseRowVector(CompressedDataMatrix * columnData) {

	nTransposeRows = columnData->getNumberOfColumns();
	cout << "nCols Original start = " << nTransposeRows << endl;
	nTransposeCols = columnData->getNumberOfRows();
	cout << "nRows Original start = " << nTransposeCols << endl;

	formatType = columnData->getFormatType(0);

	switch (formatType) {
	case INDICATOR:
		listCompressedVectorLengths = new int_vector;
		transposeIndicator(columnData);
		break;
	case DENSE:
		transposeDense(columnData);
		break;
	case SPARSE:
		//TODO do this too
		break;
	}

}



SparseRowVector::~SparseRowVector() {

	switch(formatType) {
		case(INDICATOR): {
			typedef std::vector<int_vector*>::iterator IIterator;
			for (IIterator it = matrixTransposeIndicator.begin(); it != matrixTransposeIndicator.end(); ++it) {
				if (*it) {
					delete *it;
				}
			}
		}
		break;
		case(DENSE): {
			typedef std::vector<int_vector*>::iterator IIterator;
			for (IIterator it = matrixTransposeIndicator.begin(); it != matrixTransposeIndicator.end(); ++it) {
				if (*it) {
					delete *it;
				}
			}
		}
		break;
		case(SPARSE): {
			//TODO Write this
		}
		break;
	}
}

int* SparseRowVector::getCompressedRowVector(int row) const {
	return const_cast<int*>(&(matrixTransposeIndicator[row]->at(0)));
}

bsccs::real* SparseRowVector::getDataVector(int row) const {
	return const_cast<bsccs::real*>(&(matrixTransposeDense[row]->at(0)));
}

void SparseRowVector::printRow(int row) {
	real_vector values;
	values.assign(nTransposeRows, 0.0);
	int* indicators = getCompressedRowVector(row);
	int n = getNumberOfEntries(row);
	for (int i = 0; i < n; ++i) {
		const int k = indicators[i];
		if (false) {
			//values[k] = data[column]->at(i);
		} else {
			values[k] = 1.0;
		}
	}
	printVector(values.data(), values.size());

}

void SparseRowVector::printSparseMatrix() {

	switch(formatType) {
		case(INDICATOR): {
			cout << "Printing Indicator Matrix Transpose" << endl;
			for (int i = 0; i < this->nTransposeCols; i++) {
				int numRows = this->getNumberOfEntries(i);
				cout << "Column i = " << i << " [";
				for (int j = 0; j< numRows; j++) {
					cout << this->getCompressedRowVector(i)[j] << ",";
				}
				cout << "]" << endl;
			}
		}
		break;
		case(DENSE): {
			cout << "Printing Dense Matrix Transpose" << endl;
			for (int i = 0; i < this->nTransposeCols; i++){
				cout << "Column i = " << i << " [";
				for (int j = 0; j< nTransposeRows; j++) {
					cout << this->getDataVector(i)[j] << ",";
				}
				cout << "]" << endl;
			}
		}
		break;
		case(SPARSE): {
			cout << "Printing Sparse Matrix Transpose" << endl;
			//TODO - write this
		}
		break;
	}
}

FormatType SparseRowVector::getFormatType() {
	return formatType;
}

int SparseRowVector::getNumberOfRows() {
	return nTransposeRows;
}

int SparseRowVector::getNumberOfColumns(){
	return nTransposeCols;
}

int SparseRowVector::getNumberOfEntries(int column){
	return matrixTransposeIndicator[column]->size();
}

int * SparseRowVector::getNumberOfEntriesList() const {
	//return &(listCompressedVectorLengths);
	return const_cast<int*>(&(listCompressedVectorLengths->at(0)));
}

void SparseRowVector::transposeIndicator(CompressedDataMatrix* columnData) {

	cout <<"TransposeIndicator Used" << endl;
	matrixTransposeIndicator.resize(nTransposeCols);

	for (int k = 0; k < nTransposeCols; k++) {
		matrixTransposeIndicator[k] = new int_vector();
	}

	for (int i = 0; i < nTransposeRows; i++) {
		int rows = columnData->getNumberOfEntries(i);
		for (int j = 0; j < rows; j++) {
			matrixTransposeIndicator[columnData->getCompressedColumnVector(i)[j]]->push_back(i);
		}
		//listCompressedVectorLengths->push_back(rows);
	}

	for (int p = 0; p < nTransposeCols; p++) {
		listCompressedVectorLengths->push_back(matrixTransposeIndicator[p]->size());
	}

}

void SparseRowVector::transposeDense(CompressedDataMatrix* columnData) {

	cout <<"TransposeDense Used" << endl;

	matrixTransposeDense.resize(nTransposeCols);

	cout << "Number of Columns = " << nTransposeCols << endl;

	for (int k = 0; k < nTransposeCols; k++) {
		matrixTransposeDense[k] = new real_vector();
	}

	for (int i = 0; i < nTransposeRows; i++) {
		for (int j = 0; j < nTransposeCols; j++) {
			matrixTransposeDense[j]->push_back(columnData->getDataVector(i)[j]);
		}
	}
}

void SparseRowVector::setChangedStatus(bool changeSetting) {
	hasChanged = changeSetting;
}

bool SparseRowVector::getChangedStatus() {
	return hasChanged;
}

void SparseRowVector::setUseThisStatus(bool useTransposeSetting) {
	useTransposeMatrix = useTransposeSetting;
}

bool SparseRowVector::getUseThisStatus() {
	return useTransposeMatrix;
}

}
