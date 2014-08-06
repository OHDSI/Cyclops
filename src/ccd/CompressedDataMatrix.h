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
#include <stdexcept>

//#define DATA_AOS

#include "Types.h"

namespace bsccs {

typedef std::vector<int> int_vector;
typedef std::vector<real> real_vector;

enum FormatType {
	DENSE, SPARSE, INDICATOR, INTERCEPT
};

class CompressedDataColumn {
public:
    typedef bsccs::shared_ptr<CompressedDataColumn> Ptr;
    
	CompressedDataColumn(int_vector* colIndices, real_vector* colData, FormatType colFormat,
			std::string colName = "", IdType nName = 0, bool sPtrs = false) :
		 columns(colIndices), data(colData), formatType(colFormat), stringName(colName), 
		 numericalName(nName), sharedPtrs(sPtrs) {
		// Do nothing
	}
	
	virtual ~CompressedDataColumn() {		
	    // Only release data if not shared
		if (columns && !sharedPtrs) {
			delete columns;
		}
		if (data && !sharedPtrs) {
			delete data;
		}
	}

	int* getColumns() const {
		return static_cast<int*>(columns->data());
	}
	
	real* getData() const {
		return static_cast<real*>(data->data());
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

	bool add_data(int row, real value) {
		if (formatType == DENSE) {
			//Making sure that we are at the correct row
			for(int i = data->size(); i < row; i++) {
				data->push_back(0.0);
			}
			data->push_back(value);
		} else if (formatType == SPARSE) {
			if (value != static_cast<real> (0)) {
				// Check for previous entry
				if (columns->size() > 0 && columns->back() == row) {
					return false;
				}
				data->push_back(value);
				columns->push_back(row);
			}
		} else if (formatType == INDICATOR) {
			if (value != static_cast<real> (0)) {
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
// 			std::cerr << "Error" << std::endl;
// 			exit(-1);
		}
		return true;
	}

	void convertColumnToDense(int nRows);
	
	void convertColumnToSparse(void);

	void fill(real_vector& values, int nRows);

	void printColumn(int nRows);

	real sumColumn(int nRows);

	real squaredSumColumn() const;

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
		
	void addToColumnVector(int_vector addEntries);
	void removeFromColumnVector(int_vector removeEntries);
private:
	// Disable copy-constructors and assignment constructors
	CompressedDataColumn();
	CompressedDataColumn(const CompressedDataColumn&);
	CompressedDataColumn& operator = (const CompressedDataColumn&);
	
	int_vector* columns;
	real_vector* data;
	FormatType formatType;
	mutable std::string stringName;
	IdType numericalName;
	bool sharedPtrs; // TODO Actually use shared pointers
};

class CompressedDataMatrix {

public:

	CompressedDataMatrix();

	CompressedDataMatrix(const char* fileName);

	virtual ~CompressedDataMatrix();
	
	size_t getNumberOfRows(void) const;
	
	size_t getNumberOfColumns(void) const;

	void setNumberOfColumns(int nColumns);

	size_t getNumberOfEntries(int column) const;

	int* getCompressedColumnVector(int column) const;

	void removeFromColumnVector(int column, int_vector removeEntries) const;
	void addToColumnVector(int column, int_vector addEntries) const;

	real* getDataVector(int column) const;

	void getDataRow(int row, real* x) const;
	CompressedDataMatrix* transpose();

	FormatType getFormatType(int column) const;

	void convertColumnToDense(int column);

	void convertColumnToSparse(int column);

	void printColumn(int column);

	real sumColumn(int column);

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

	const CompressedDataColumn& getColumn(size_t column) const {
		return *(allColumns[column]);
	}

	CompressedDataColumn& getColumn(size_t column) {
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
			real_vector* r = new real_vector(real_begin, real_end);
			push_back(NULL, r, DENSE);
		} else if (colFormat == SPARSE) {
			real_vector* r = new real_vector(real_begin, real_end);
			int_vector* i = new int_vector(int_begin, int_end);
			push_back(i, r, SPARSE);
		} else if (colFormat == INDICATOR) {
			int_vector* i = new int_vector(int_begin, int_end);
			push_back(i, NULL, INDICATOR);
		} else if (colFormat == INTERCEPT) {
			push_back(NULL, NULL, INTERCEPT);
		} else {
// 			std::cerr << "Error" << std::endl;
// 			exit(-1);
            throw new std::invalid_argument("Unknown type");
 		}
	}

	void push_back(FormatType colFormat) {
		if (colFormat == DENSE) {
			real_vector* r = new real_vector();
			push_back(NULL, r, DENSE);
		} else if (colFormat == SPARSE) {
			real_vector* r = new real_vector();
			int_vector* i = new int_vector();
			push_back(i, r, SPARSE);
		} else if (colFormat == INDICATOR) {
			int_vector* i = new int_vector();
			push_back(i, NULL, INDICATOR);
		} else if (colFormat == INTERCEPT) {
			push_back(NULL, NULL, INTERCEPT);
		} else {
// 			std::cerr << "Error" << std::endl;
// 			exit(-1);
            throw new std::invalid_argument("Unknown type");			
 		}
	}
	
	void insert(size_t position, FormatType colFormat) {
	    if (colFormat == DENSE) {
	        real_vector* r = new real_vector();
	        insert(allColumns.begin() + position, NULL, r, DENSE);	    
	    } else {
            throw new std::invalid_argument("Unknown type");	    
// 	        std::cerr << "Error" << std::endl;
// 	        exit(-1); // TODO Use error handler
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

protected:
   
    typedef CompressedDataColumn::Ptr CompressedDataColumnPtr;
    typedef std::vector<CompressedDataColumnPtr>  DataColumnVector;
   
	void push_back(int_vector* colIndices, real_vector* colData, FormatType colFormat) {
		allColumns.push_back(
//		new CompressedDataColumn
		make_shared<CompressedDataColumn>(colIndices, colData, colFormat)
		);	
		nCols++;
	}
	
	void insert(DataColumnVector::iterator position, int_vector* colIndices, real_vector* colData, FormatType colFormat) {
	    allColumns.insert(position, 
// 	    new CompressedDataColumn
	    make_shared<CompressedDataColumn>
	    (colIndices, colData, colFormat)
	    );
	    nCols++;
	}
	
	size_t nRows;
	size_t nCols;
	size_t nEntries;
	DataColumnVector allColumns;

private:
	// Disable copy-constructors and copy-assignment
	CompressedDataMatrix(const CompressedDataMatrix&);
	CompressedDataMatrix& operator = (const CompressedDataMatrix&);
};

} // namespace

#endif /* COMPRESSEDINDICATORMATRIX_H_ */
