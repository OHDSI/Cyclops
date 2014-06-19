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
#include "Iterators.h"

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
	
	double sum(const DrugIdType covariate);
	
	
protected:

    size_t getColumnIndex(const DrugIdType covariate);
    
	template <typename F>
	double reduce(const size_t index, F func) {    		    
	    double sum = 0.0;
		switch (getFormatType(index)) {
			case INDICATOR :
				sum = reduceImpl<IndicatorIterator>(index, func);
				break;				
			case SPARSE :			
				sum = reduceImpl<SparseIterator>(index, func);				
				break;
			case DENSE :
				sum = reduceImpl<DenseIterator>(index, func);				
				break;
			case INTERCEPT :
				sum = reduceImpl<InterceptIterator>(index, func);
				break;
				
		}	    
	    return sum;	
	}
	
	template <typename IteratorType, typename F>
	double reduceImpl(const size_t index, F func) {
	    double sum = 0.0;
	    IteratorType it(*this, index);
	    for (; it; ++it) {
	        sum += func(it.value());
	    }	
	    return sum;
	}
	

private:
	// Disable copy-constructors and copy-assignment
	RcppModelData(const ModelData&);
	RcppModelData& operator = (const RcppModelData&);
};

} /* namespace bsccs */
#endif /* RCPPMODELDATA_H_ */
