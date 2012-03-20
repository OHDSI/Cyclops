/*
 * CompressedIndicatorMatrix.h
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

#include <vector>

using namespace std;

#define DEBUG

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif 

typedef std::vector<int> int_vector;
typedef std::vector<real> real_vector;

class CompressedIndicatorMatrix {

public:
	enum FormatType {
		DENSE, SPARSE, INDICATOR
	};

	CompressedIndicatorMatrix();

	CompressedIndicatorMatrix(const char* fileName);

	virtual ~CompressedIndicatorMatrix();
	
	int getNumberOfRows(void) const;
	
	int getNumberOfColumns(void);

	int getNumberOfEntries(int column) const;

	int* getCompressedColumnVector(int column) const;

	FormatType getFormatType(int column) const;

protected:
	void allocateMemory(int nCols);

//private:
	int nRows;
	int nCols;
	int nEntries;
	std::vector<int_vector> columns;	
	std::vector<real_vector> data;
//	std::vector<int> rows;  // standard CSC representation
//	std::vector<int> ptrStart;

};

#endif /* COMPRESSEDINDICATORMATRIX_H_ */
