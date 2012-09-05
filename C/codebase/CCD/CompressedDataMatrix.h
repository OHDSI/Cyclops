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

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;

#define DATA_AOS

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif 

typedef std::vector<int> int_vector;
typedef std::vector<real> real_vector;

enum FormatType {
	DENSE, SPARSE, INDICATOR
};

typedef int DrugIdType;

class CompressedDataColumn {
public:
	CompressedDataColumn(int_vector* colIndices, real_vector* colData, FormatType colFormat,
			std::string colName = "", DrugIdType nName = 0) :
		 columns(colIndices), data(colData), formatType(colFormat), stringName(colName), numericalName(nName) {
		// Do nothing
	}
	
	virtual ~CompressedDataColumn() {		
		if (columns) {
			delete columns;
		}
		if (data) {
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
	
	const std::string& getLabel() {
		if (stringName == "") {
			std::stringstream ss;
			ss << numericalName;
			stringName = ss.str();
		}
		return stringName;
	}

	const DrugIdType& getNumericalLabel() const {
		return numericalName;
	}
	
	int getNumberOfEntries() const {
		return columns->size();
	}
	
	void add_label(std::string label) {
		stringName = label;
	}
	
	void add_label(DrugIdType label) {
		numericalName = label;
	}	

	void add_data(int row, real value) {
		if (formatType == DENSE) {
			data->push_back(value);
			// TODO Make sure we are at the correct row
		} else if (formatType == SPARSE) {
			if (value != static_cast<real> (0)) {
				data->push_back(value);
				columns->push_back(row);
			}
		} else if (formatType == INDICATOR) {
			if (value != static_cast<real> (0)) {
				columns->push_back(row);
			}
		} else {
			cerr << "Error" << endl;
			exit(-1);
		}
	}

	void convertColumnToDense(int nRows);
	
	void convertColumnToSparse(void);

	void printColumn(int nRows);

	template <class T>
	void printVector(T values, const int size) {
		cout << "[" << values[0];
		for (int i = 1; i < size; ++i) {
			cout << " " << values[i];
		}
		cout << "]" << endl;
	}

	static bool sortNumerically(const CompressedDataColumn* i, const CompressedDataColumn* j) {
		return i->getNumericalLabel() < j->getNumericalLabel();
	}
		
	void addToColumnVector(int_vector addEntries);
	void removeFromColumnVector(int_vector removeEntries);
private:
	// Disable copy-constructors
	CompressedDataColumn();
	CompressedDataColumn(const CompressedDataColumn&);
	
	int_vector* columns;
	real_vector* data;
	FormatType formatType;
	std::string stringName;
	DrugIdType numericalName;
};

class CompressedDataMatrix {

public:

	CompressedDataMatrix();

	CompressedDataMatrix(const char* fileName);

	virtual ~CompressedDataMatrix();
	
	int getNumberOfRows(void) const;
	
	int getNumberOfColumns(void) const;

	void setNumberOfColumns(int nColumns);

	int getNumberOfEntries(int column) const;

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

	CompressedDataColumn& getColumn(int column) {
		return *(allColumns[column]);
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
		} else {
			cerr << "Error" << endl;
			exit(-1);
 		}
	}	

protected:

	void push_back(int_vector* colIndices, real_vector* colData, FormatType colFormat) {
		allColumns.push_back(new CompressedDataColumn(colIndices, colData, colFormat));	
		nCols++;
	}
	
	void erase(int column) {
		if (allColumns[column]) {
			delete allColumns[column];
		}
		allColumns.erase(allColumns.begin() + column);
	}

//private:
	int nRows;
	int nCols;
	int nEntries;

	std::vector<CompressedDataColumn*> allColumns;

//	std::vector<int> rows;  // standard CSC representation
//	std::vector<int> ptrStart;

};

#endif /* COMPRESSEDINDICATORMATRIX_H_ */
