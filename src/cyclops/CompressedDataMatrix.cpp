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
#include <stdexcept>

#include "CompressedDataMatrix.h"

namespace bsccs {

CompressedDataMatrix::CompressedDataMatrix() : nRows(0), nCols(0), nEntries(0) {
	// Do nothing
}

CompressedDataMatrix::~CompressedDataMatrix() {
//	typedef std::vector<CompressedDataColumn*>::iterator CIterator;
//	for (CIterator it = allColumns.begin(); it != allColumns.end(); ++it) {
//		delete *it;
//	}
}

real CompressedDataMatrix::sumColumn(int column) {
	real sum = 0.0;
	if (getFormatType(column) == DENSE) {
	    throw new std::invalid_argument("DENSE");
// 		cerr << "Not yet implemented (DENSE)." << endl;
// 		exit(-1);
	} else if (getFormatType(column) == SPARSE) {
	    throw new std::invalid_argument("DENSE");
// 		cerr << "Not yet implemented (SPARSE)." << endl;
// 		exit(-1);
	} else { // is indiciator
		sum = allColumns[column]->getNumberOfEntries();
	}
	return sum;
}

struct DrugIDComparator {
	IdType id;
	DrugIDComparator(IdType i) : id(i) { }
	bool operator()(const CompressedDataColumn::Ptr& column) {
		return column->getNumericalLabel() == id;
	}
};

int CompressedDataMatrix::getColumnIndexByName(IdType name) const {

	DrugIDComparator cmp(name);
	auto found = std::find_if(
			allColumns.begin(), allColumns.end(), cmp);
	if (found != allColumns.end()) {
		return std::distance(allColumns.begin(), found);
	} else {
		return -1;
	}
}

// void CompressedDataMatrix::printColumn(int column) {
// #if 1
// 	cerr << "Not yet implemented.\n";
// 	exit(-1);
// #else
// 	real_vector values;
// 	if (getFormatType(column) == DENSE) {
// 		values.assign(data[column]->begin(), data[column]->end());
// 	} else {
// 		bool isSparse = getFormatType(column) == SPARSE;
// 		values.assign(nRows, 0.0);
// 		int* indicators = getCompressedColumnVector(column);
// 		size_t n = getNumberOfEntries(column);
// 		for (size_t i = 0; i < n; ++i) {
// 			const int k = indicators[i];
// 			if (isSparse) {
// 				values[k] = data[column]->at(i);
// 			} else {
// 				values[k] = 1.0;
// 			}
// 		}
// 	}
// 	printVector(values.data(), values.size());
// #endif
// }

void CompressedDataMatrix::convertColumnToSparse(int column) {
	allColumns[column]->convertColumnToSparse();
}

void CompressedDataMatrix::convertColumnToDense(int column) {
	allColumns[column]->convertColumnToDense(nRows);
}

size_t CompressedDataMatrix::getNumberOfRows(void) const {
	return nRows;
}

size_t CompressedDataMatrix::getNumberOfColumns(void) const {
	return nCols;
}

size_t CompressedDataMatrix::getNumberOfEntries(int column) const {
	return allColumns[column]->getNumberOfEntries();
}

size_t CompressedDataMatrix::getNumberOfNonZeroEntries(int column) const {
    const auto type = getFormatType(column);
    if (type == INTERCEPT || type == DENSE) {
        return getNumberOfRows();
    } else {
        return getNumberOfEntries(column);
    }
}

int* CompressedDataMatrix::getCompressedColumnVector(int column) const {
	return allColumns[column]->getColumns();
}

std::vector<int>& CompressedDataMatrix::getCompressedColumnVectorSTL(int column) const {
	return allColumns[column]->getColumnsVector();
}

real* CompressedDataMatrix::getDataVector(int column) const {
	return allColumns[column]->getData();
}

std::vector<real>& CompressedDataMatrix::getDataVectorSTL(int column) const {
	return allColumns[column]->getDataVector();
}


FormatType CompressedDataMatrix::getFormatType(int column) const {
	return allColumns[column]->getFormatType();
}

void CompressedDataColumn::fill(RealVector& values, int nRows) {
	values.resize(nRows);
	if (formatType == DENSE) {
			values.assign(data->begin(), data->end());
		} else {
			bool isSparse = formatType == SPARSE;
			values.assign(nRows, 0.0);
			int* indicators = getColumns();
			size_t n = getNumberOfEntries();
			for (size_t i = 0; i < n; ++i) {
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
//	bool flagSparse = false;
	for (size_t i = 0; i < nCols; i++) {
		FormatType thisFormatType = this->allColumns[i]->getFormatType();
		if (thisFormatType == DENSE)
			flagDense = true;
		if (thisFormatType == INDICATOR)
			flagIndicator = true;
	}

	if (flagIndicator && flagDense) {
//		flagSparse = true;
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

	for (size_t i = 0; i < matTranspose->nRows; i++) {
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
			for (size_t j = 0; j < nRows; j++) {
				matTranspose->getColumn(j).add_data(i,
						this->getDataVector(i)[j]);
			}
		}
	}

	return matTranspose;
}

// TODO Fix massive copying
void CompressedDataMatrix::addToColumnVector(int column, IntVector addEntries) const{
	allColumns[column]->addToColumnVector(addEntries);
}

void CompressedDataMatrix::removeFromColumnVector(int column, IntVector removeEntries) const{
	allColumns[column]->removeFromColumnVector(removeEntries);
}


void CompressedDataMatrix::getDataRow(int row, real* x) const {
	for(size_t j = 0; j < nCols; j++)
	{
		if(this->allColumns[j]->getFormatType() == DENSE)
			x[j] = this->getDataVector(j)[row];
		else{
			x[j] = 0.0;
			int* col = this->getCompressedColumnVector(j);
			for(size_t i = 0; i < this->allColumns[j]->getNumberOfEntries(); i++){
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

//void CompressedDataColumn::printColumn(int nRows) {
//	real_vector values;
//	fill(values, nRows);
//	printVector(values.data(), values.size());
//}

real CompressedDataColumn::sumColumn(int nRows) {
	RealVector values;
	fill(values, nRows);
	return std::accumulate(values.begin(), values.end(), static_cast<real>(0.0));
}

real CompressedDataColumn::squaredSumColumn(size_t n) const {
	if (formatType == INDICATOR) {
		return getNumberOfEntries();
	} else if (formatType == INTERCEPT) {
	    return static_cast<real>(n);
	} else {
		return std::inner_product( data->begin(), data->end(), data->begin(), static_cast<real>(0.0));
	}
}

void CompressedDataColumn::convertColumnToSparse(void) {
	if (formatType == SPARSE) {
		return;
	}
	if (formatType == DENSE) {
// 		fprintf(stderr, "Format not yet support.\n");
// 		exit(-1);
        throw new std::invalid_argument("DENSE");
	}

	if (data == NULL) {
//		data = new real_vector();
        data = make_shared<RealVector>();
	}

	const real value = 1.0;
	data->assign(getNumberOfEntries(), value);
	formatType = SPARSE;
}

void CompressedDataColumn::convertColumnToDense(int nRows) {
	if (formatType == DENSE) {
		return;
	}

//	real_vector* oldData = data;
    RealVectorPtr oldData = data;
//	data = new real_vector();
    data = make_shared<RealVector>();

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
	formatType = DENSE;
//	delete columns;
    columns = NULL;
//	if (oldData) {
//		delete oldData;
//	}
}

// TODO Fix massive copying
void CompressedDataColumn::addToColumnVector(IntVector addEntries){
	int lastit = 0;

	for(int i = 0; i < (int)addEntries.size(); i++)
	{
		IntVector::iterator it = columns->begin() + lastit;
        for(; it != columns->end(); it++){
    		if(*it > addEntries[i]){
                break;
            }
		    lastit++;
		}
		columns->insert(it,addEntries[i]);
	}
}

void CompressedDataColumn::removeFromColumnVector(IntVector removeEntries){
	int lastit = 0;
	IntVector::iterator it1 = removeEntries.begin();
	IntVector::iterator it2 = columns->begin();
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

void CompressedDataColumn::printMatrixMarketFormat(std::ostream& stream, const int rows, const int columnNumber) const {

    if (formatType == DENSE || formatType == INTERCEPT) {
        for (int row = 0; row < rows; ++row) {
            double value = (formatType == DENSE) ? getDataVector()[row] : 1.0;
            stream << (row + 1) << " " << (columnNumber + 1) << " " << value << "\n";
        }
    } else if (formatType == SPARSE || formatType == INDICATOR) {
        const auto columns = getColumnsVector();

        for (int i = 0; i < columns.size(); ++i) {
            double value = (formatType == SPARSE) ? getDataVector()[i] : 1.0;
            stream << (columns[i] + 1) << " " << (columnNumber + 1) <<  " " << value << "\n";
        }
    } else {
        throw new std::invalid_argument("Unknon type");
    }
}

void CompressedDataMatrix::printMatrixMarketFormat(std::ostream& stream) const {

    size_t nnz = 0;
    for (int i = 0; i < getNumberOfColumns(); ++i) {
        nnz += getNumberOfNonZeroEntries(i);
    }

    stream << "%%MatrixMarket matrix coordinate real general\n";
    stream << "%\n";
    stream << getNumberOfRows() << " " << getNumberOfColumns() << " " << nnz << "\n";

    for (int col = 0; col < getNumberOfColumns(); ++col) {
        getColumn(col).printMatrixMarketFormat(stream, getNumberOfRows(), col);
    }
}

} // namespace
