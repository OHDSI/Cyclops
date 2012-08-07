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

	nTransposeRows = columnData->getNumberOfColumns();;
	nTransposeCols = columnData->getNumberOfRows();

	matrixTranspose.resize(nTransposeCols);

	for (int k = 0; k < nTransposeCols; k++) {
		matrixTranspose[k] = new int_vector();
	}

	for (int i = 0; i < nTransposeRows; i++) {
		int rows = columnData->getNumberOfEntries(i);
		for (int j = 0; j < rows; j++) {
			matrixTranspose[columnData->getCompressedColumnVector(i)[j]]->push_back(i);
		}
	}

}



SparseRowVector::~SparseRowVector() {

	typedef std::vector<int_vector*>::iterator IIterator;
		for (IIterator it = matrixTranspose.begin(); it != matrixTranspose.end(); ++it) {
			if (*it) {
				delete *it;
			}
		}
}

int* SparseRowVector::getCompressedRowVector(int row) const {
	return const_cast<int*>(&(matrixTranspose[row]->at(0)));
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

int SparseRowVector::getNumberOfRows() {
	return nTransposeRows;
}

int SparseRowVector::getNumberOfColumns(){
	return nTransposeCols;
}

int SparseRowVector::getNumberOfEntries(int column){
	return matrixTranspose[column]->size();
}


}
