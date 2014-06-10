/*
 * RcppModelData.h
 *
 *  Created on: Apr 24, 2014
 *      Author: msuchard
 */

#ifndef RCPPMODELDATA_H_
#define RCPPMODELDATA_H_

#include "Rcpp.h"
#include "ModelData.h"

namespace bsccs {

using Rcpp::NumericVector;
using Rcpp::IntegerVector;

class RcppModelData : public ModelData {

public:
	RcppModelData();

	RcppModelData(
			const IntegerVector& pid,
			const NumericVector& y,
			const NumericVector& z,
			const NumericVector& offs,
			const NumericVector& dxv, // dense
			const IntegerVector& siv, // sparse
			const IntegerVector& spv,
			const NumericVector& sxv,
			const IntegerVector& iiv, // indicator
			const IntegerVector& ipv
			);

	virtual ~RcppModelData();

private:
	// Disable copy-constructors and copy-assignment
	RcppModelData(const ModelData&);
	RcppModelData& operator = (const RcppModelData&);
};

} /* namespace bsccs */
#endif /* RCPPMODELDATA_H_ */
