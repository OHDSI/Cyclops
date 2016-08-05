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
            ModelType modelType,
			const IntegerVector& pid,
			const NumericVector& y,
			const NumericVector& z,
			const NumericVector& offs,
			const NumericVector& dxv, // dense
			const IntegerVector& siv, // sparse
			const IntegerVector& spv,
			const NumericVector& sxv,
			const IntegerVector& iiv, // indicator
			const IntegerVector& ipv,
            bool useTimeAsOffset = false,
            int numTypes = 1
			);
			
	RcppModelData(
			ModelType modelType,
	        loggers::ProgressLoggerPtr log,
    	    loggers::ErrorHandlerPtr error       
        );			

	virtual ~RcppModelData();

	double sum(const IdType covariate, int power = 1);

	void standardize(const IdType covariate);

	void sumByGroup(std::vector<double>& out, const IdType covariate, const IdType groupBy, int power = 1);

	void sumByGroup(std::vector<double>& out, const IdType covariate, int power = 1);

    template <typename F>
    void transform(const size_t index, F func) {
		switch (getFormatType(index)) {
			case INDICATOR :
				transformImpl<IndicatorIterator>(index, func);
				break;
			case SPARSE :
				transformImpl<SparseIterator>(index, func);
				break;
			case DENSE :
				transformImpl<DenseIterator>(index, func);
				break;
			case INTERCEPT :
				transformImpl<InterceptIterator>(index, func);
				break;
		}
    }

	template <typename F>
	double reduce(const long index, F func) {
		if (index < 0) { // reduce outcome
			return reduceOutcomeImpl(func);
		}
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

	template <typename F>
	double innerProductWithOutcome(const size_t index, F func) {
	    double sum = 0.0;
	    switch (getFormatType(index)) {
	    case INDICATOR :
	        sum = innerProductWithOutcomeImpl<IndicatorIterator>(index, func);
	        break;
	    case SPARSE :
	        sum = innerProductWithOutcomeImpl<SparseIterator>(index, func);
	        break;
	    case DENSE :
	        sum = innerProductWithOutcomeImpl<DenseIterator>(index, func);
	        break;
	    case INTERCEPT :
	        sum = innerProductWithOutcomeImpl<InterceptIterator>(index, func);
	        break;
	    }
	    return sum;
	}

	template <typename T, typename F>
	void reduceByGroup(T& out, const size_t reductionIndex, const size_t groupByIndex, F func) {
	    if (getFormatType(groupByIndex) != INDICATOR) {
	        std::ostringstream stream;
	        stream << "Grouping by non-indicators is not yet supported.";
	        error->throwError(stream);
	    }
		switch (getFormatType(reductionIndex)) {
			case INDICATOR :
                reduceByGroupImpl<IndicatorIterator>(out, reductionIndex, groupByIndex, func);
				break;
			case SPARSE :
				reduceByGroupImpl<SparseIterator>(out, reductionIndex, groupByIndex, func);
				break;
			case DENSE :
				reduceByGroupImpl<DenseIterator>(out, reductionIndex, groupByIndex, func);
				break;
			case INTERCEPT :
			    reduceByGroupImpl<InterceptIterator>(out, reductionIndex, groupByIndex, func);
				break;
		}
	}

protected:

	template <typename T, typename F>
	void reduceByGroup(T& out, const size_t reductionIndex, const std::vector<int>& groups, F func) {
		switch (getFormatType(reductionIndex)) {
			case INDICATOR :
				reduceByGroupImpl<IndicatorIterator>(out, reductionIndex, groups, func);
				break;
			case SPARSE :
				reduceByGroupImpl<SparseIterator>(out, reductionIndex, groups, func);
				break;
			case DENSE :
				reduceByGroupImpl<DenseIterator>(out, reductionIndex, groups, func);
				break;
			case INTERCEPT :
				reduceByGroupImpl<InterceptIterator>(out, reductionIndex, groups, func);
				break;

		}
	}

	template <typename IteratorType, typename T, typename F>
	void reduceByGroupImpl(T& out, const size_t reductionIndex, const size_t groupByIndex, F func) {
	    IteratorType reduceIt(*this, reductionIndex);
	    IndicatorIterator groupByIt(*this, groupByIndex);

	    GroupByIterator<IteratorType> it(reduceIt, groupByIt);
	    for (; it; ++it) {
	        out[it.group()] += func(it.value()); // TODO compute reduction in registers
	    }
	}

	template <typename IteratorType, typename T, typename F>
	void reduceByGroupImpl(T& out, const size_t reductionIndex, const std::vector<int>& groups, F func) {
	    IteratorType it(*this, reductionIndex);

	    for (; it; ++it) {
	        out[groups[it.index()]] += func(it.value()); // TODO compute reduction in registers
	    }
	}

	template <typename IteratorType, typename F>
	void transformImpl(const size_t index, F func) {
	    IteratorType it(*this, index);
	    for (; it; ++it) {
	        it.ref() = func(it.value()); // TODO No yet implemented
	    }
	}

	template <typename F>
	double reduceOutcomeImpl(F func) {
		double sum = 0.0;
		for(auto it = std::begin(y); it != std::end(y); ++it) {
			sum += func(*it);
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

	template <typename IteratorType, typename F>
	double innerProductWithOutcomeImpl(const size_t index, F func) {
		double sum = 0.0;
		IteratorType it(*this, index);
		for (; it; ++it) {
			sum += func(y[it.index()], it.value());
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
