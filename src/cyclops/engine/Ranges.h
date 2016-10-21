/*
 * Ranges.h
 *
 *  Created on: Feb 5, 2015
 *      Author: msuchard
 */

#ifndef RANGE_H_
#define RANGE_H_

#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

namespace bsccs {

// Helper functions until we can remove raw_pointers
namespace { // anonymous

template <typename T>
inline T* begin(T* x) { return x; }

template <typename T>
inline const T* begin(const T* x) { return x; }

// inline real* begin(real* x) { return x; }
// inline int* begin(int* x) {  return x; }
//
// inline const real* begin(const real* x) { return x; }
// inline const int* begin(const int* x) {  return x; }

template <typename RealType>
inline RealType* begin(std::vector<RealType>& x) { return x.data(); }

inline int* begin(std::vector<int>& x) { return x.data(); }

template <typename RealType>
inline const RealType* begin(const std::vector<RealType>& x) { return x.data(); }

inline const int* begin(const std::vector<int>& x) { return x.data(); }

}; // namespace anonymous

namespace helper {

    namespace detail {

        template <class T>
        inline const T* begin(const T* x) { return x; }

        template <class T>
        auto begin(const std::vector<T>& x) -> decltype(std::begin(x)) { return std::begin(x); }

        template <class T>
        auto end(const std::vector<T>& x) -> decltype(std::end(x)) { return std::end(x); }

         template <typename... TContainerIt>
         auto zip_begin(TContainerIt&... containers) ->
                boost::zip_iterator<
                    decltype(boost::make_tuple(containers...))
                > {
            return boost::make_zip_iterator(boost::make_tuple(containers...));
         }

         template <typename... TContainerIt>
         auto zip_end(TContainerIt&... containers) ->
                boost::zip_iterator<
                    decltype(boost::make_tuple(containers...))
                > {
            return boost::make_zip_iterator(boost::make_tuple(containers...));
         }

         template <typename ZipIt>
         auto zip_range_type(ZipIt& begin, ZipIt& end) ->
            boost::iterator_range<
                decltype(begin)
            > {
        	return { begin, end };
        }
    } // namespace detail

    auto getRangeAll(const int length) ->
            boost::iterator_range<
                decltype(boost::make_counting_iterator(0))
            > {
        return  {
            boost::make_counting_iterator(0),
            boost::make_counting_iterator(length)
        };
    }

    template <typename RealType, typename RealVectorType>
    auto getRangeAllNumerators(const int length, const RealVectorType& y, const RealVectorType& xBeta, const RealVectorType& weight) ->
    		boost::iterator_range<
    			boost::zip_iterator<
    				boost::tuple<
    					decltype(std::begin(y)),
    					decltype(std::begin(xBeta)),
    					decltype(std::begin(weight))
						>
    			>
    		> {

    		auto x0 = std::begin(y);
    		auto x1 = std::begin(xBeta);
    		auto x2 = std::begin(weight);

    	return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x2
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0 + length, x1 + length, x2 + length
                ))
    	};
    }

    template <typename RealType, typename RealVectorType>
    auto getRangeAllDenominators(const int length, const RealVectorType& denominator, const RealVectorType& weight) ->
    		boost::iterator_range<
    			boost::zip_iterator<
    				boost::tuple<
    					decltype(std::begin(denominator)),
    					decltype(std::begin(weight))
						>
    			>
    		> {

    		auto x0 = std::begin(denominator);
    		auto x1 = std::begin(weight);

    	return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0 + length, x1 + length
                ))
    	};
    }

// // TODO Remove code duplication with immediately above
//     auto getRangeAllDenominators(const int length, const real* denominator, const RealVectorType& weight) ->
//     		boost::iterator_range<
//     			boost::zip_iterator<
//     				boost::tuple<
//     					decltype(begin(denominator)),
//     					decltype(std::begin(weight))
// 						>
//     			>
//     		> {
//
//     		auto x0 = begin(denominator);
//     		auto x1 = std::begin(weight);
//
//     	return {
//             boost::make_zip_iterator(
//                 boost::make_tuple(
//                 	x0, x1
//                 )),
//             boost::make_zip_iterator(
//                 boost::make_tuple(
//                 	x0 + length, x1 + length
//                 ))
//     	};
//     }

    template <typename RealType, typename RealVectorType>
    auto getRangeAllPredictiveLikelihood(const int length, const RealVectorType& y, const RealVectorType& xBeta,
            const RealVectorType& denominator, const double* weights, const int* pid, std::true_type) ->

        boost::iterator_range<
            boost::zip_iterator<
                boost::tuple<
                    decltype(std::begin(y)),  // 0
                    decltype(std::begin(xBeta)), // 1
                    decltype(std::begin(denominator)), // 2
                    decltype(begin(weights))
                >
            >
        > {

        auto x0 = std::begin(y);
        auto x1 = std::begin(xBeta);
        auto x2 = std::begin(denominator);
        auto x3 = begin(weights);

 		return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x2, x3
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0 + length, x1 + length, x2 + length, x3 + length
                )
            )
        };
    }

    template <typename RealType, typename RealVectorType>
    auto getRangeAllPredictiveLikelihood(const int length, const RealVectorType& y, const RealVectorType& xBeta,
            const RealVectorType& denominator, const double* weights, const int* pid, std::false_type) ->

        boost::iterator_range<
            boost::zip_iterator<
                boost::tuple<
                    decltype(std::begin(y)),  // 0
                    decltype(std::begin(xBeta)), // 1
                    decltype(boost::make_permutation_iterator( // 2
                        std::begin(denominator),
                        begin(pid)
                    )),
                    decltype(begin(weights))
                >
            >
        > {

        auto x0 = std::begin(y);
        auto x1 = std::begin(xBeta);
        auto x2 = boost::make_permutation_iterator(
                        std::begin(denominator),
                        begin(pid));
        auto x3 = begin(weights);

 		return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x2, x3
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0 + length, x1 + length, x2 + length, x3 + length
                )
            )
        };
    }
//         (BaseModel::cumulativeGradientAndHessian) ? accDenomPid : denomPid,
//         weights, hPid);

namespace independent {

//    template <class ExpXBetaType, class XBetaType, class YType, class DenominatorType, class WeightType>
    template <typename RealType, typename RealVectorType>
    auto getRangeX(const CompressedDataMatrix<RealType>& mat, const int index,
//    			ExpXBetaType& expXBeta, XBetaType& xBeta, YType& y, DenominatorType& denominator, WeightType& weight,
  					RealVectorType& expXBeta, RealVectorType& xBeta, const RealVectorType& y,
  					RealVectorType& denominator,
  					RealVectorType& weight,
  					typename IndicatorIterator<RealType>::tag) ->

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(expXBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(xBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(y),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(denominator),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(weight),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    )
 					>
 				>
 			> {

 		auto x0 = boost::make_permutation_iterator(
 					        std::begin(expXBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x1 = boost::make_permutation_iterator(
 					        std::begin(xBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x2 = boost::make_permutation_iterator(
 					        std::begin(y),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x3 = boost::make_permutation_iterator(
 					        std::begin(denominator),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x4 = boost::make_permutation_iterator(
 					        std::begin(weight),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

		auto y0 = boost::make_permutation_iterator(
					        std::begin(expXBeta),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y1 = boost::make_permutation_iterator(
					        std::begin(xBeta),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y2 = boost::make_permutation_iterator(
					        std::begin(y),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y3 = boost::make_permutation_iterator(
					        std::begin(denominator),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y4 = boost::make_permutation_iterator(
					        std::begin(weight),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

 		return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x2, x3, x4
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                	y0, y1, y2, y3, y4
                ))
        };
 	}

//    template <class ExpXBetaType, class XBetaType, class YType, class DenominatorType, class WeightType>
    template <typename RealType, typename RealVectorType>
    auto getRangeX(const CompressedDataMatrix<RealType>& mat, const int index,
//    			ExpXBetaType& expXBeta, XBetaType& xBeta, YType& y, DenominatorType& denominator, WeightType& weight,
  					RealVectorType& expXBeta, RealVectorType& xBeta, const RealVectorType& y,
  					RealVectorType& denominator,
  					RealVectorType& weight,
  					typename SparseIterator<RealType>::tag) ->

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(expXBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(xBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(y),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(denominator),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(weight),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
						decltype(std::begin(mat.getDataVectorSTL(index))) // Never deferenced
 					>
 				>
 			> {

 		auto x0 = boost::make_permutation_iterator(
 					        std::begin(expXBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x1 = boost::make_permutation_iterator(
 					        std::begin(xBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x2 = boost::make_permutation_iterator(
 					        std::begin(y),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x3 = boost::make_permutation_iterator(
 					        std::begin(denominator),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x4 = boost::make_permutation_iterator(
 					        std::begin(weight),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

 		auto x5 = std::begin(mat.getDataVectorSTL(index));

		auto y0 = boost::make_permutation_iterator(
					        std::begin(expXBeta),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y1 = boost::make_permutation_iterator(
					        std::begin(xBeta),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y2 = boost::make_permutation_iterator(
					        std::begin(y),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y3 = boost::make_permutation_iterator(
					        std::begin(denominator),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y4 = boost::make_permutation_iterator(
					        std::begin(weight),
					        std::end(mat.getCompressedColumnVectorSTL(index)));
 	  	auto y5 = std::end(mat.getDataVectorSTL(index));

 		return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x2, x3, x4, x5
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                	y0, y1, y2, y3, y4, y5
                ))
        };
 	}

//	template <class ExpXBetaType, class XBetaType, class YType, class DenominatorType, class WeightType>
    template <typename RealType, typename RealVectorType>
    auto getRangeX(const CompressedDataMatrix<RealType>& mat, const int index,
//    			ExpXBetaType& expXBeta, XBetaType& xBeta, YType& y, DenominatorType& denominator, WeightType& weight,
  					RealVectorType& expXBeta, RealVectorType& xBeta, const RealVectorType& y,
  					RealVectorType& denominator,
  					RealVectorType& weight,
  					typename DenseIterator<RealType>::tag) ->

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
		            	decltype(std::begin(expXBeta)),
		            	decltype(std::begin(xBeta)),
		            	decltype(std::begin(y)),
		            	decltype(std::begin(denominator)),
		            	decltype(std::begin(weight)),
		                decltype(std::begin(mat.getDataVectorSTL(index)))
        		    >
            	>
            > {

		const size_t K = mat.getNumberOfRows();

        auto x0 = std::begin(expXBeta);
        auto x1 = std::begin(xBeta);
        auto x2 = std::begin(y);
        auto x3 = std::begin(denominator);
        auto x4 = std::begin(weight);
        auto x5 = std::begin(mat.getDataVectorSTL(index));

        auto y0 = std::end(expXBeta);
        auto y1 = std::end(xBeta);
        auto y2 = std::end(y);
        auto y3 = x3 + K;
        auto y4 = x4 + K; // length == K + 1
        auto y5 = std::end(mat.getDataVectorSTL(index));

        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x2, x3, x4, x5
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
					y0, y1, y2, y3, y4, y5
                ))
        };
    }

//	template <class ExpXBetaType, class XBetaType, class YType, class DenominatorType, class WeightType>
    template <typename RealType, typename RealVectorType>
    auto getRangeX(const CompressedDataMatrix<RealType>& mat, const int index,
//    			ExpXBetaType& expXBeta, XBetaType& xBeta, YType& y, DenominatorType& denominator, WeightType& weight,
  					RealVectorType& expXBeta, RealVectorType& xBeta, const RealVectorType& y,
  					RealVectorType& denominator,
  					RealVectorType& weight,
  					typename InterceptIterator<RealType>::tag) ->

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
		            	decltype(std::begin(expXBeta)),
		            	decltype(std::begin(xBeta)),
		            	decltype(std::begin(y)),
		            	decltype(std::begin(denominator)),
		            	decltype(std::begin(weight))
        		    >
            	>
            > {

            // TODO NEW

		const size_t K = mat.getNumberOfRows();

        auto x0 = std::begin(expXBeta);
        auto x1 = std::begin(xBeta);
        auto x2 = std::begin(y);
        auto x3 = std::begin(denominator);
        auto x4 = std::begin(weight);

        auto y0 = std::end(expXBeta);
        auto y1 = std::end(xBeta);
        auto y2 = std::end(y);
        auto y3 = x3 + K;
        auto y4 = x4 + K; // length == K + 1

        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x2, x3, x4
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
					y0, y1, y2, y3, y4
                ))
        };
    }

    template <typename RealType, typename RealVectorType>
    auto getRangeXBeta(const CompressedDataMatrix<RealType>& mat, const int index,
  					RealVectorType& expXBeta, RealVectorType& xBeta,
  					RealVectorType& denominator,
  					const RealVectorType& offs,
  					typename IndicatorIterator<RealType>::tag) ->

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(expXBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(xBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(denominator),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(offs),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    )
 					>
 				>
 			> {

 		auto x0 = boost::make_permutation_iterator(
 					        std::begin(expXBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x1 = boost::make_permutation_iterator(
 					        std::begin(xBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x3 = boost::make_permutation_iterator(
 					        std::begin(denominator),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x4 = boost::make_permutation_iterator(
 					        std::begin(offs),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

		auto y0 = boost::make_permutation_iterator(
					        std::begin(expXBeta),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y1 = boost::make_permutation_iterator(
					        std::begin(xBeta),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y3 = boost::make_permutation_iterator(
					        std::begin(denominator),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y4 = boost::make_permutation_iterator(
					        std::begin(offs),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

 		return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x3, x4
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                	y0, y1, y3, y4
                ))
        };
 	}

    template <typename RealType, typename RealVectorType>
    auto getRangeXBeta(const CompressedDataMatrix<RealType>& mat, const int index,
  					RealVectorType& expXBeta, RealVectorType& xBeta,
  					RealVectorType& denominator,
  					const RealVectorType& offs,
  					typename SparseIterator<RealType>::tag) ->

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(expXBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(xBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(denominator),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
 					    decltype(boost::make_permutation_iterator(
 					        std::begin(offs),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)))
 					    ),
						decltype(std::begin(mat.getDataVectorSTL(index))) // Never deferenced
 					>
 				>
 			> {

 		auto x0 = boost::make_permutation_iterator(
 					        std::begin(expXBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x1 = boost::make_permutation_iterator(
 					        std::begin(xBeta),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x3 = boost::make_permutation_iterator(
 					        std::begin(denominator),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

        auto x4 = boost::make_permutation_iterator(
 					        std::begin(offs),
 					        std::begin(mat.getCompressedColumnVectorSTL(index)));

 		auto x5 = std::begin(mat.getDataVectorSTL(index));

		auto y0 = boost::make_permutation_iterator(
					        std::begin(expXBeta),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y1 = boost::make_permutation_iterator(
					        std::begin(xBeta),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y3 = boost::make_permutation_iterator(
					        std::begin(denominator),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

    	auto y4 = boost::make_permutation_iterator(
					        std::begin(offs),
					        std::end(mat.getCompressedColumnVectorSTL(index)));

 	  	auto y5 = std::end(mat.getDataVectorSTL(index));

 		return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x3, x4, x5
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                	y0, y1, y3, y4, y5
                ))
        };
 	}

    template <typename RealType, typename RealVectorType>
    auto getRangeXBeta(const CompressedDataMatrix<RealType>& mat, const int index,
  					RealVectorType& expXBeta, RealVectorType& xBeta,
  					RealVectorType& denominator,
  					const RealVectorType& offs,
  					typename DenseIterator<RealType>::tag) ->

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
		            	decltype(std::begin(expXBeta)),
		            	decltype(std::begin(xBeta)),
		            	decltype(std::begin(denominator)),
		            	decltype(std::begin(offs)),
		                decltype(std::begin(mat.getDataVectorSTL(index)))
        		    >
            	>
            > {

		const size_t K = mat.getNumberOfRows();

        auto x0 = std::begin(expXBeta);
        auto x1 = std::begin(xBeta);
        auto x3 = std::begin(denominator);
        auto x4 = std::begin(offs);
        auto x5 = std::begin(mat.getDataVectorSTL(index));

        auto y0 = std::end(expXBeta);
        auto y1 = std::end(xBeta);
        auto y3 = x3 + K;
        auto y4 = std::end(offs);
        auto y5 = std::end(mat.getDataVectorSTL(index));

        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x3, x4, x5
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
					y0, y1, y3, y4, y5
                ))
        };
    }



} // namespace independent

// START DEPENDENT
namespace dependent {

//    template <class KeyType, class IteratorType> // For sparse
    template <typename RealType>
    auto getRangeKey(const CompressedDataMatrix<RealType>& mat, const int index,
    		//KeyType& pid, IteratorType
    		int* pid,
    		typename SparseIterator<RealType>::tag
    		) ->
            boost::iterator_range<
                decltype(boost::make_permutation_iterator(
                    begin(pid),
                    std::begin(mat.getCompressedColumnVectorSTL(index))))
            > {
//        const size_t K = mat.getNumberOfRows();
        return {
    	    boost::make_permutation_iterator(
			    begin(pid),
				std::begin(mat.getCompressedColumnVectorSTL(index))),
    	    boost::make_permutation_iterator(
			    begin(pid),
				std::end(mat.getCompressedColumnVectorSTL(index)))
        };
    }

//    template <class KeyType, class IteratorType> // For indicator
    template <typename RealType>
    auto getRangeKey(const CompressedDataMatrix<RealType>& mat, const int index,
    		//KeyType& pid, IteratorType
    		int* pid,
    		typename IndicatorIterator<RealType>::tag
    		) ->
            boost::iterator_range<
                decltype(boost::make_permutation_iterator(
                    begin(pid),
                    std::begin(mat.getCompressedColumnVectorSTL(index))))
            > {
//        const size_t K = mat.getNumberOfRows();
        return {
    	    boost::make_permutation_iterator(
			    begin(pid),
				std::begin(mat.getCompressedColumnVectorSTL(index))),
    	    boost::make_permutation_iterator(
			    begin(pid),
				std::end(mat.getCompressedColumnVectorSTL(index)))
        };
    }

//    template <class KeyType> // For dense
    template <typename RealType>
    auto getRangeKey(const CompressedDataMatrix<RealType>& mat, const int index,
    		//KeyType& pid,
    		int* pid,
    		DenseTag) ->
            boost::iterator_range<
                decltype(begin(pid))
            > {
        const size_t K = mat.getNumberOfRows();
        return {
            begin(pid),
            begin(pid) + K
        };
    }

//    template <class KeyType> // For intercept
    template <typename RealType>
    auto getRangeKey(const CompressedDataMatrix<RealType>& mat, const int index,
    		//KeyType& pid,
    		int* pid,         // TODO NEW
    		InterceptTag) ->
            boost::iterator_range<
                decltype(begin(pid))
            > {
        const size_t K = mat.getNumberOfRows();
        return {
            begin(pid),
            begin(pid) + K
        };
    }

//    template <class ExpXBeta> // For dense
    template <typename RealType, typename RealVectorType>
    auto getRangeX(const CompressedDataMatrix<RealType>& mat, const int index,
                //ExpXBeta&
                RealVectorType&
                expXBeta, typename DenseIterator<RealType>::tag) ->
            boost::iterator_range<
                boost::zip_iterator<
                    boost::tuple<
                        decltype(std::begin(expXBeta)),
                        decltype(std::begin(mat.getDataVectorSTL(index)))
                    >
                >
            > {
        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                    std::begin(expXBeta),
                    std::begin(mat.getDataVectorSTL(index))
            )),
            boost::make_zip_iterator(
                boost::make_tuple(
                    std::end(expXBeta),
                    std::end(mat.getDataVectorSTL(index))
            ))
        };
    }

//    template <class ExpXBeta> // For sparse
    template <typename RealType, typename RealVectorType>
    auto getRangeX(const CompressedDataMatrix<RealType>& mat, const int index,
                //ExpXBeta&
                RealVectorType&
                expXBeta, typename SparseIterator<RealType>::tag) ->
            boost::iterator_range<
                boost::zip_iterator<
                    boost::tuple<
                	    decltype(boost::make_permutation_iterator(
			                std::begin(expXBeta),
            				std::begin(mat.getCompressedColumnVectorSTL(index)))),
                        decltype(std::begin(mat.getDataVectorSTL(index)))
                    >
                >
            > {
        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                    boost::make_permutation_iterator(
                        std::begin(expXBeta),
                        std::begin(mat.getCompressedColumnVectorSTL(index))),
                std::begin(mat.getDataVectorSTL(index))
            )),
            boost::make_zip_iterator(
                boost::make_tuple(
                    boost::make_permutation_iterator(
                        std::begin(expXBeta),
                        std::end(mat.getCompressedColumnVectorSTL(index))),
                    std::end(mat.getDataVectorSTL(index))
            ))
        };
    }

//    template <class ExpXBeta> // For indicator
    template <typename RealType, typename RealVectorType>
    auto getRangeX(const CompressedDataMatrix<RealType>& mat, const int index,
                //ExpXBeta&
                RealVectorType&
                expXBeta, typename IndicatorIterator<RealType>::tag) ->
            boost::iterator_range<
                boost::zip_iterator<
                    boost::tuple<
                	    decltype(boost::make_permutation_iterator(
			                std::begin(expXBeta),
            				std::begin(mat.getCompressedColumnVectorSTL(index))))
                    >
                >
            > {
        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                    boost::make_permutation_iterator(
                        std::begin(expXBeta),
                        std::begin(mat.getCompressedColumnVectorSTL(index)))
            )),
            boost::make_zip_iterator(
                boost::make_tuple(
                    boost::make_permutation_iterator(
                        std::begin(expXBeta),
                        std::end(mat.getCompressedColumnVectorSTL(index)))
            ))
        };
    }

//    template <class ExpXBeta> // For intercept
    template <typename RealType, typename RealVectorType>
    auto getRangeX(const CompressedDataMatrix<RealType>& mat, const int index,
                //ExpXBeta&
                RealVectorType&
                expXBeta, typename InterceptIterator<RealType>::tag) ->
            boost::iterator_range<
                boost::zip_iterator<
                    boost::tuple<
                        decltype(std::begin(expXBeta))
                    >
                >
            > { // TODO NEW
        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                    std::begin(expXBeta)
            )),
            boost::make_zip_iterator(
                boost::make_tuple(
                    std::end(expXBeta)
            ))
        };
    }

	template <class DenominatorType, class SubsetType, class IteratorType>
    auto getRangeDenominator(SubsetType& subset, const size_t N,
                DenominatorType& denominator,
                IteratorType) ->  // For indicator and sparse
 			boost::iterator_range<
                        decltype(boost::make_permutation_iterator(
                            begin(denominator),
                            std::begin(subset)))
            > {
        return {
                    boost::make_permutation_iterator(
                        begin(denominator),
                        std::begin(subset)),

                    boost::make_permutation_iterator(
                        begin(denominator),
                        std::end(subset))
        };
    }


	template <class DenominatorType, class SubsetType>
    auto getRangeDenominator(SubsetType& subset, const size_t N,
                DenominatorType& denominator,
                DenseTag) ->  // For dense
 			boost::iterator_range<
                        decltype(
                            begin(denominator))
            > {
            auto b0 = begin(denominator);

        return {  b0, b0 + N };
    }


	template <class DenominatorType, class WeightType, class SubsetType, class IteratorType>
    auto getRangeGradient(SubsetType* subset, const size_t N,
                DenominatorType& denominator, WeightType& weight,
                IteratorType) ->  // For indicator and sparse
 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
                        decltype(boost::make_permutation_iterator(
                            begin(denominator),
                            std::begin(*subset))),
 	                    decltype(boost::make_permutation_iterator(
                            std::begin(weight),
                            std::begin(*subset)))
        		    >
            	>
            > {
        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                    boost::make_permutation_iterator(
                        begin(denominator),
                        std::begin(*subset)),
                    boost::make_permutation_iterator(
                        std::begin(weight),
                        std::begin(*subset))
                )),
            boost::make_zip_iterator(
            		boost::make_tuple(
                    boost::make_permutation_iterator(
                        begin(denominator),
                        std::end(*subset)),
                    boost::make_permutation_iterator(
                        std::begin(weight),
                        std::end(*subset))
                ))
        };
    }

	template <class DenominatorType, class WeightType, class SubsetType> // For dense
    auto getRangeGradient(SubsetType* mat, const size_t N,
                DenominatorType& denominator, WeightType& weight,
                DenseTag) ->
 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<
                        decltype(begin(denominator)),
                        decltype(std::begin(weight))
        		    >
            	>
            > {

// 		const size_t K = mat.getNumberOfRows();
        auto b0 = begin(denominator);
        auto b1 = std::begin(weight);

        auto e0 = b0 + N; //K;
        auto e1 = b1 + N; //K;

        return {
            boost::make_zip_iterator(
                boost::make_tuple(
                    b0, b1
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
                    e0, e1
                ))
        };
    }

} // namespace dependent

} // namespace helper

} // namespace bsccs

#endif // RANGE_H_
