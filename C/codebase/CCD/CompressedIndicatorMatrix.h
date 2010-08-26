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

class CompressedIndicatorMatrix {

public:
	CompressedIndicatorMatrix();

	CompressedIndicatorMatrix(const char* fileName);

	virtual ~CompressedIndicatorMatrix();
	
	int getNumberOfRows(void);
	
	int getNumberOfColumns(void);

	int getNumberOfEntries(int column);

	int* getCompressedColumnVector(int column);

protected:
	void allocateMemory(int nCols);

//private:
	int nRows;
	int nCols;
	int nEntries;
	std::vector<int_vector> columns;	
//	std::vector<int> rows;  // standard CSC representation
//	std::vector<int> ptrStart;

};

#endif /* COMPRESSEDINDICATORMATRIX_H_ */
