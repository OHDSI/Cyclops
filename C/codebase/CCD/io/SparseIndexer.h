/*
 * SparseIndexer.h
 *
 *  Created on: Aug 28, 2012
 *      Author: msuchard
 */

#ifndef SPARSEINDEXER_H_
#define SPARSEINDEXER_H_

#include <map>

#include "../CompressedDataMatrix.h"

class SparseIndexer {
public:
	SparseIndexer(CompressedDataMatrix& matrix) : dataMatrix(matrix), nCovariates(0) {}
	virtual ~SparseIndexer() {}
	

	CompressedDataColumn& getColumn(const DrugIdType& covariate) {
		return dataMatrix.getColumn(sparseMap[covariate]);
	}
	
	void addColumn(const DrugIdType& covariate, FormatType type) {
		const int index = dataMatrix.getNumberOfColumns();
		sparseMap.insert(std::make_pair(covariate, index));
		
		dataMatrix.push_back(type);
		
		// Add numerical labels
		dataMatrix.getColumn(index).add_label(covariate);
	}
	
	bool hasColumn(DrugIdType covariate) const {
		return sparseMap.count(covariate) != 0;
	}	
		
private:
	CompressedDataMatrix& dataMatrix;
	int nCovariates;
	std::map<DrugIdType, int> sparseMap;	
};

#endif /* SPARSEINDEXER_H_ */
