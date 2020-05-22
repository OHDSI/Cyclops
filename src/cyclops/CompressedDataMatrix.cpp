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

template <typename RealType>
CompressedDataMatrix<RealType>::CompressedDataMatrix() : nRows(0), nCols(0), nEntries(0) {
	// Do nothing
}

template <typename RealType>
CompressedDataMatrix<RealType>::~CompressedDataMatrix() {
//	typedef std::vector<CompressedDataColumn*>::iterator CIterator;
//	for (CIterator it = allColumns.begin(); it != allColumns.end(); ++it) {
//		delete *it;
//	}
}

template <typename RealType>
RealType CompressedDataMatrix<RealType>::sumColumn(int column) {
	RealType sum = 0.0;
	if (getFormatType(column) == DENSE) {
	    throw new std::invalid_argument("DENSE");
	} else if (getFormatType(column) == SPARSE) {
	    throw new std::invalid_argument("DENSE");
	} else { // is indiciator
		sum = allColumns[column]->getNumberOfEntries();
	}
	return sum;
}

template <typename RealType>
struct DrugIDComparator {
	IdType id;
	DrugIDComparator(IdType i) : id(i) { }
	bool operator()(const typename CompressedDataColumn<RealType>::Ptr& column) {
		return column->getNumericalLabel() == id;
	}
};

template <typename RealType>
int CompressedDataMatrix<RealType>::getColumnIndexByName(IdType name) const {

	DrugIDComparator<RealType> cmp(name);
	auto found = std::find_if(
			allColumns.begin(), allColumns.end(), cmp);
	if (found != allColumns.end()) {
		return std::distance(allColumns.begin(), found);
	} else {
		return -1;
	}
}

template <typename RealType>
void CompressedDataMatrix<RealType>::convertColumnToSparse(int column) {
	allColumns[column]->convertColumnToSparse();
}

template <typename RealType>
void CompressedDataMatrix<RealType>::convertColumnToDense(int column) {
	allColumns[column]->convertColumnToDense(nRows);
}

template <typename RealType>
size_t CompressedDataMatrix<RealType>::getNumberOfRows(void) const {
	return nRows;
}

template <typename RealType>
size_t CompressedDataMatrix<RealType>::getNumberOfColumns(void) const {
	return nCols;
}

template <typename RealType>
size_t CompressedDataMatrix<RealType>::getNumberOfEntries(int column) const {
	return allColumns[column]->getNumberOfEntries();
}

template <typename RealType>
size_t CompressedDataMatrix<RealType>::getNumberOfNonZeroEntries(int column) const {
    const auto type = getFormatType(column);
    if (type == INTERCEPT || type == DENSE) {
        return getNumberOfRows();
    } else {
        return getNumberOfEntries(column);
    }
}

template <typename RealType>
int* CompressedDataMatrix<RealType>::getCompressedColumnVector(int column) const {
	return allColumns[column]->getColumns();
}

template <typename RealType>
std::vector<int>& CompressedDataMatrix<RealType>::getCompressedColumnVectorSTL(int column) const {
	return allColumns[column]->getColumnsVector();
}

template <typename RealType>
RealType* CompressedDataMatrix<RealType>::getDataVector(int column) const {
	return allColumns[column]->getData();
}

template <typename RealType>
typename CompressedDataMatrix<RealType>::RealVector&
CompressedDataMatrix<RealType>::getDataVectorSTL(int column) const {
	return allColumns[column]->getDataVector();
}

template <typename RealType>
FormatType CompressedDataMatrix<RealType>::getFormatType(int column) const {
	return allColumns[column]->getFormatType();
}

template <typename RealType>
void CompressedDataColumn<RealType>::fill(RealVector& values, int nRows) const {
	values.resize(nRows);
	if (formatType == DENSE) {
			values.assign(data->begin(), data->end());
		} else {
			bool isSparse = formatType == SPARSE;
			values.assign(nRows, static_cast<RealType>(0));
			int* indicators = getColumns();
			size_t n = getNumberOfEntries();
			for (size_t i = 0; i < n; ++i) {
				const int k = indicators[i];
				if (isSparse) {
					values[k] = data->at(i);
				} else {
					values[k] = static_cast<RealType>(1.0);
				}
			}
		}
}

template <typename RealType>
bsccs::shared_ptr<CompressedDataMatrix<RealType>>
CompressedDataMatrix<RealType>::transpose() const {
	auto matTranspose = bsccs::make_shared<CompressedDataMatrix<RealType>>();
// 	CompressedDataMatrix* matTranspose = new CompressedDataMatrix();

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
		} else if (thisFormatType == INTERCEPT) {
			int rows = getNumberOfRows();
			for (int j = 0; j < rows; ++j) {
				matTranspose->getColumn(j).add_data(i, 1.0);
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
template <typename RealType>
void CompressedDataMatrix<RealType>::addToColumnVector(int column, IntVector addEntries) const{
	allColumns[column]->addToColumnVector(addEntries);
}

template <typename RealType>
void CompressedDataMatrix<RealType>::removeFromColumnVector(int column, IntVector removeEntries) const{
	allColumns[column]->removeFromColumnVector(removeEntries);
}

template <typename RealType>
void CompressedDataMatrix<RealType>::getDataRow(int row, RealType* x) const {
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

template <typename RealType>
void CompressedDataMatrix<RealType>::setNumberOfColumns(int nColumns) { // TODO Depricate?
	nCols = nColumns;
}
// End TODO

//void CompressedDataColumn::printColumn(int nRows) {
//	real_vector values;
//	fill(values, nRows);
//	printVector(values.data(), values.size());
//}

template <typename RealType>
RealType CompressedDataColumn<RealType>::sumColumn(int nRows) { // TODO Depricate
	RealVector values;
	fill(values, nRows);
	return std::accumulate(values.begin(), values.end(), static_cast<RealType>(0.0));
}


template <typename RealType>
RealType CompressedDataColumn<RealType>::squaredSumColumn(size_t n) const {
	if (formatType == INDICATOR) {
		return getNumberOfEntries();
	} else if (formatType == INTERCEPT) {
	    return static_cast<RealType>(n);
	} else {
		return std::inner_product( data->begin(), data->end(), data->begin(), static_cast<RealType>(0));
	}
}

template <typename RealType>
void CompressedDataColumn<RealType>::convertColumnToSparse(void) {
	if (formatType == SPARSE) {
		return;
	}
	if (formatType == DENSE) {
        throw new std::invalid_argument("DENSE");
	}

	if (data == NULL) {
        data = make_shared<RealVector>();
	}

	const RealType value = static_cast<RealType>(1);
	data->assign(getNumberOfEntries(), value);
	formatType = SPARSE;
}

template <typename RealType>
void CompressedDataColumn<RealType>::convertColumnToDense(int nRows) {
	if (formatType == DENSE) {
		return;
	}

    RealVectorPtr oldData = data;
    data = make_shared<RealVector>();

	data->resize(nRows, static_cast<RealType>(0));

	int* indicators = getColumns();
	int n = getNumberOfEntries();
//	int nonzero = 0;
	for (int i = 0; i < n; ++i) {
		const int k = indicators[i];
//		cerr << " " << k;
//		nonzero++;

		RealType value = (formatType == SPARSE) ? oldData->at(i) : static_cast<RealType>(1);

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
template <typename RealType>
void CompressedDataColumn<RealType>::addToColumnVector(IntVector addEntries){
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

template <typename RealType>
void CompressedDataColumn<RealType>::removeFromColumnVector(IntVector removeEntries){
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

template <typename RealType>
void CompressedDataColumn<RealType>::printMatrixMarketFormat(std::ostream& stream, const int rows, const int columnNumber) const {

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

template <typename RealType>
void CompressedDataMatrix<RealType>::printMatrixMarketFormat(std::ostream& stream) const {

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

// Instantiate classes
template class CompressedDataColumn<double>;
template class CompressedDataColumn<float>;
template class CompressedDataMatrix<double>;
template class CompressedDataMatrix<float>;

} // namespace
