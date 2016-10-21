/*
 * SparseIndexer.h
 *
 *  Created on: Aug 28, 2012
 *      Author: msuchard
 */

#ifndef SPARSEINDEXER_H_
#define SPARSEINDEXER_H_

#include <map>

//#include "../CompressedDataMatrix.h"
class CompressedDataMatrix; // forward reference
class CompressedDataColumn; // forward reference

namespace bsccs {

template <typename RealType>
class SparseIndexer {
public:
	SparseIndexer(CompressedDataMatrix<RealType>& matrix) : dataMatrix(matrix) {}
	virtual ~SparseIndexer() {}


	CompressedDataColumn<RealType>& getColumn(const IdType& covariate) {
		return dataMatrix.getColumn(sparseMap[covariate]);
	}

	void addColumn(const IdType& covariate, FormatType type) {
		const int index = dataMatrix.getNumberOfColumns();
		sparseMap.insert(std::make_pair(covariate, index));

		dataMatrix.push_back(type);

		// Add numerical labels
		dataMatrix.getColumn(index).add_label(covariate);
	}

	bool hasColumn(IdType covariate) const {
		return sparseMap.count(covariate) != 0;
	}

	int getIndex(IdType covariate){
		return sparseMap[covariate];
	}

private:
	CompressedDataMatrix<RealType>& dataMatrix;
//	int nCovariates;
	std::map<IdType, int> sparseMap;
};

} // namespace

#endif /* SPARSEINDEXER_H_ */
