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
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

//#define DATA_AOS

#include "Types.h"

namespace bsccs {

// typedef std::vector<int> int_vector;
// typedef std::vector<real> real_vector;
// typedef std::vector<int> IntVector;
// typedef std::vector<real> RealVector;
// typedef bsccs::shared_ptr<IntVector> IntVectorPtr;
// typedef bsccs::shared_ptr<RealVector> RealVectorPtr;

enum FormatType {
	DENSE, SPARSE, INDICATOR, INTERCEPT
};

template <typename RealType>
class CompressedDataColumn {
public:

//     typedef bsccs::shared_ptr<CompressedDataColumn> Ptr;
    typedef bsccs::unique_ptr<CompressedDataColumn> Ptr;

	CompressedDataColumn(VectorPtr<int> colIndices, VectorPtr<RealType> colData, FormatType colFormat,
			std::string colName = "", IdType nName = 0, bool sPtrs = false) :
		 columns(colIndices), data(colData), formatType(colFormat), stringName(colName),
		 numericalName(nName), sharedPtrs(sPtrs) {
		// Do nothing
	}

	virtual ~CompressedDataColumn() {
	    // Only release data if not shared
//		if (columns && !sharedPtrs) {
//			delete columns;
//		}
//		if (data && !sharedPtrs) {
//			delete data;
//		}
	}

	int* getColumns() const {
		return static_cast<int*>(columns->data());
	}

	RealType* getData() const {
		return static_cast<RealType*>(data->data());
	}

	const Vector<int>& getColumnsVector() const {
		return *columns;
	}

	const Vector<RealType>& getDataVector() const {
		return *data;
	}

	Vector<int>& getColumnsVector() {
		return *columns;
	}

	Vector<RealType>& getDataVector() {
		return *data;
	}

	Vector<RealType> copyData() {
// 		std::vector copy(std::begin(data), std::end(data));
// 		return std::move(copy);
		return Vector<RealType>(*data);
	}

	template <typename Function>
	void transform(Function f) {
	    std::transform(data->begin(), data->end(), data->begin(), f);
	}

	template <typename Function, typename ValueType>
	ValueType accumulate(Function f, ValueType x) {
	    return std::accumulate(data->begin(), data->end(), x, f);
	}

	FormatType getFormatType() const {
		return formatType;
	}

	const std::string& getLabel() const {
		if (stringName == "") {
			std::stringstream ss;
			ss << numericalName;
			stringName = ss.str();
		}
		return stringName;
	}

	const IdType& getNumericalLabel() const {
		return numericalName;
	}

	const std::string getTypeString() const {
		std::string str;
		if (formatType == DENSE) {
			str = "dense";
		} else if (formatType == SPARSE) {
			str = "sparse";
		} else if (formatType == INDICATOR) {
			str = "indicator";
		} else if (formatType == INTERCEPT) {
			str = "intercept";
		} else {
			str = "unknown";
		}
		return str;
	}

	size_t getNumberOfEntries() const {
		return columns->size();
	}

	size_t getDataVectorLength() const {
		return data->size();
	}

	void add_label(std::string label) {
		stringName = label;
	}

	void add_label(IdType label) {
		numericalName = label;
	}

	template <typename T> // *** TODO FP remove template?
	bool add_data(int row, T value) {
		if (formatType == DENSE) {
			//Making sure that we are at the correct row
			for(int i = data->size(); i < row; i++) {
				data->push_back(static_cast<RealType>(0));
			}
			data->push_back(value);
		} else if (formatType == SPARSE) {
			if (value != static_cast<RealType>(0)) {
				// Check for previous entry
				if (columns->size() > 0 && columns->back() == row) {
					return false;
				}
				data->push_back(value);
				columns->push_back(row);
			}
		} else if (formatType == INDICATOR) {
			if (value != static_cast<RealType> (0)) {
				// Check for previous entry
				if (columns->size() > 0 && columns->back() == row) {
					return false;
				}
				columns->push_back(row);
			}
		} else if (formatType == INTERCEPT) {
			// Do nothing
		} else {
            throw new std::invalid_argument("Unknown type");
		}
		return true;
	}

	void convertColumnToDense(int nRows);

	void convertColumnToSparse(void);

	void fill(Vector<RealType>& values, int nRows) const;

	void printColumn(int nRows);

	RealType sumColumn(int nRows);

	RealType squaredSumColumn(size_t n) const;

// 	template <class T>
// 	void printVector(T values, const int size) {
// 	    using std::cout;
// 	    using std::endl;
// 		cout << "[" << values[0];
// 		for (int i = 1; i < size; ++i) {
// 			cout << " " << values[i];
// 		}
// 		cout << "]" << endl;
// 	}

// 	static bool sortNumerically(const CompressedDataColumn* i, const CompressedDataColumn* j) {
// 		return i->getNumericalLabel() < j->getNumericalLabel();
// 	}

	static bool sortNumerically(const Ptr& i, const Ptr& j) {
		return i->getNumericalLabel() < j->getNumericalLabel();
	}

	void addToColumnVector(Vector<int> addEntries);
	void removeFromColumnVector(Vector<int> removeEntries);
private:
	// Disable copy-constructors and assignment constructors
	CompressedDataColumn();
	CompressedDataColumn(const CompressedDataColumn&);
	CompressedDataColumn& operator = (const CompressedDataColumn&);

	VectorPtr<int> columns;
	VectorPtr<RealType> data;

	FormatType formatType;
	mutable std::string stringName;
	IdType numericalName;
	bool sharedPtrs; // TODO Actually use shared pointers
};

template <typename RealType>
class CompressedDataMatrix {

public:

	CompressedDataMatrix();

	CompressedDataMatrix(const char* fileName);

	virtual ~CompressedDataMatrix();

	size_t getNumberOfRows(void) const;

	size_t getNumberOfColumns(void) const;

	void setNumberOfColumns(int nColumns);

	size_t getNumberOfEntries(int column) const;

	size_t getNumberOfNonZeroEntries(int column) const;

	int* getCompressedColumnVector(int column) const; // TODO depreciate
	std::vector<int>& getCompressedColumnVectorSTL(int column) const;

	void removeFromColumnVector(int column, Vector<int> removeEntries) const;
	void addToColumnVector(int column, Vector<int> addEntries) const;

 	RealType* getDataVector(int column) const;  // TODO depreciate

	Vector<RealType>& getDataVectorSTL(int column) const;

	void getDataRow(int row, RealType* x) const;
	bsccs::shared_ptr<CompressedDataMatrix> transpose() const;

	FormatType getFormatType(int column) const;

	void convertColumnToDense(int column);

	void convertColumnToSparse(int column);

	void printColumn(int column);

	RealType sumColumn(int column);

	/**
	 * To sort by any arbitrary measure:
	 * 1. Construct a std::map<CompressedDataColumn*, measure>
	 * 2. Build a comparator that examines:
	 * 			map[CompressedDataColumn* i] < map[CompressedDataColumn* j]
	 */

	template <typename Comparator>
	void sortColumns(Comparator cmp) {
		std::sort(allColumns.begin(), allColumns.end(),
				cmp);
	}

	const CompressedDataColumn<RealType>& getColumn(size_t column) const {
		return *(allColumns[column]);
	}

	CompressedDataColumn<RealType>& getColumn(size_t column) {
		return *(allColumns[column]);
	}

	int getColumnIndexByName(IdType name) const;

	// Make deep copy
	template <typename IntVectorItr, typename RealVectorItr>
	void push_back(
			const IntVectorItr int_begin, const IntVectorItr int_end,
			const RealVectorItr real_begin, const RealVectorItr real_end,
			FormatType colFormat) {
		if (colFormat == DENSE) {
// 			real_vector* r = new real_vector(real_begin, real_end);
            VectorPtr<RealType> r = make_shared<Vector<RealType>>(real_begin, real_end);
			push_back(NULL, r, DENSE);
		} else if (colFormat == SPARSE) {
// 			real_vector* r = new real_vector(real_begin, real_end);
            VectorPtr<RealType> r = make_shared<Vector<RealType>>(real_begin, real_end);
// 			int_vector* i = new int_vector(int_begin, int_end);
            VectorPtr<int> i = make_shared<Vector<int>>(int_begin, int_end);
			push_back(i, r, SPARSE);
		} else if (colFormat == INDICATOR) {
// 			int_vector* i = new int_vector(int_begin, int_end);
            VectorPtr<int> i = make_shared<Vector<int>>(int_begin, int_end);
			push_back(i, NULL, INDICATOR);
		} else if (colFormat == INTERCEPT) {
			push_back(NULL, NULL, INTERCEPT);
		} else {
            throw new std::invalid_argument("Unknown type");
 		}
	}

	template <typename IntVectorItr, typename RealVectorItr>
	void replace(int index,
			const IntVectorItr int_begin, const IntVectorItr int_end,
			const RealVectorItr real_begin, const RealVectorItr real_end,
			FormatType colFormat) {
		if (colFormat == DENSE) {
            VectorPtr<RealType> r = make_shared<Vector<RealType>>(real_begin, real_end);
			replace(index, NULL, r, DENSE);
		} else if (colFormat == SPARSE) {
            VectorPtr<RealType> r = make_shared<Vector<RealType>>(real_begin, real_end);
            VectorPtr<int> i = make_shared<Vector<int>>(int_begin, int_end);
			replace(index, i, r, SPARSE);
		} else if (colFormat == INDICATOR) {
            VectorPtr<int> i = make_shared<Vector<int>>(int_begin, int_end);
			replace(index, i, NULL, INDICATOR);
		} else if (colFormat == INTERCEPT) {
			replace(index, NULL, NULL, INTERCEPT);
		} else {
            throw new std::invalid_argument("Unknown type");
 		}
	}

	void push_back(FormatType colFormat) {
		if (colFormat == DENSE) {
// 			real_vector* r = new real_vector();
            VectorPtr<RealType> r = make_shared<Vector<RealType>>();
			push_back(NULL, r, DENSE);
		} else if (colFormat == SPARSE) {
// 			real_vector* r = new real_vector();
			VectorPtr<RealType> r = make_shared<Vector<RealType>>();
// 			int_vector* i = new int_vector();
			VectorPtr<int> i = make_shared<Vector<int>>();
			push_back(i, r, SPARSE);
		} else if (colFormat == INDICATOR) {
// 			int_vector* i = new int_vector();
            VectorPtr<int> i = make_shared<Vector<int>>();
			push_back(i, NULL, INDICATOR);
		} else if (colFormat == INTERCEPT) {
			push_back(NULL, NULL, INTERCEPT);
		} else {
            throw new std::invalid_argument("Unknown type");
 		}
	}

	void insert(size_t position, FormatType colFormat) {
	    if (colFormat == DENSE) {
//	        real_vector* r = new real_vector();
            VectorPtr<RealType> r = make_shared<Vector<RealType>>();
	        insert(allColumns.begin() + position, NULL, r, DENSE);
	    } else if (colFormat == INTERCEPT) {
	        insert(allColumns.begin() + position, NULL, NULL, INTERCEPT);
	    } else {
            throw new std::invalid_argument("Unknown type");
	    }
	}

	void moveToFront(size_t column) {
        if (column > 0 && column < allColumns.size())  {
            size_t reversePosition = allColumns.size() - column - 1;
            std::rotate(
                allColumns.rbegin() + reversePosition,
                allColumns.rbegin() + reversePosition + 1, // rotate one element
                allColumns.rend());
    	}
    }

	void erase(size_t column) {
		//if (allColumns[column]) {
		//	delete allColumns[column];
		//}
		allColumns.erase(allColumns.begin() + column);
		nCols--;
	}

	void push_back(VectorPtr<int> colIndices, VectorPtr<RealType> colData, FormatType colFormat) {
	    allColumns.push_back(
	        //		new CompressedDataColumn
	        // 		make_shared<CompressedDataColumn>
	        make_unique<CompressedDataColumn<RealType>>
	            (colIndices, colData, colFormat)
	    );
	    nCols++;
	}

	size_t nRows;
	size_t nCols;
	size_t nEntries;

protected:

    typedef typename CompressedDataColumn<RealType>::Ptr CompressedDataColumnPtr;
    typedef Vector<CompressedDataColumnPtr>  DataColumnVector;

	void replace(int position, VectorPtr<int> colIndices, VectorPtr<RealType> colData, FormatType colFormat) {
		auto newColumn = make_unique<CompressedDataColumn<RealType>>(colIndices, colData, colFormat);
	    allColumns[position] = std::move(newColumn);
	}

	void insert(typename DataColumnVector::iterator position, VectorPtr<int> colIndices, VectorPtr<RealType> colData, FormatType colFormat) {
	    allColumns.insert(position,
// 	    new CompressedDataColumn
// 	    make_shared<CompressedDataColumn>
        make_unique<CompressedDataColumn<RealType>>
	    (colIndices, colData, colFormat)
	    );
	    nCols++;
	}

	DataColumnVector allColumns;

private:
	// Disable copy-constructors and copy-assignment
	CompressedDataMatrix(const CompressedDataMatrix&);
	CompressedDataMatrix& operator = (const CompressedDataMatrix&);
};

} // namespace

#endif /* COMPRESSEDINDICATORMATRIX_H_ */
