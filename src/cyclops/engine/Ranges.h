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

namespace helper {

// 
// // 	auto computeRange = helper::getRangeXCOO(modelData, index,
// // 		numerPid, numerPid2, offsExpXBeta, hXBeta, hY, hPid,
// // // 		typename IteratorType::tag()
// 
// 	template <class IteratorTag, class NumeratorType, class ExpXBetaType, class XBetaType, 
// 				class YType, class PidType, class WeightType>
//     auto getRangeXDependent(const CompressedDataMatrix& mat, const int index, IteratorTag) -> void {
//     	std::cerr << "Not yet implemented." << std::endl;
//     	std::exit(-1);
//     }
// 
// 	template <class ExpXBetaType, class XBetaType, class YType, class WeightType>        
//     auto getRangeXDependent(const CompressedDataMatrix& mat, const int index, DenseTag) -> 
// //            aux::zipper_range<
//  						boost::iterator_range<
//  						boost::zip_iterator<
//  						boost::tuple<
// 	            decltype(boost::make_counting_iterator(0)),
//             	decltype(begin(mat.getDataVector(index)))            	
//             >
//             >
//             > {            	
//         
//         auto i = boost::make_counting_iterator(0); 
//         auto x = begin(mat.getDataVector(index));               
// 		const size_t K = mat.getNumberOfRows();	
//         
//         return { 
//             boost::make_zip_iterator(
//                 boost::make_tuple(i, x)),
//             boost::make_zip_iterator(
//                 boost::make_tuple(i + K, x + K))            
//         };          
//     }
//     
// 	template <class ExpXBetaType, class XBetaType, class YType, class WeightType>    
//     auto getRangeXDependent(const CompressedDataMatrix& mat, const int index, SparseTag) -> 
// //            aux::zipper_range<
// 						boost::iterator_range<
//  						boost::zip_iterator<
//  						boost::tuple<						 
// 	            decltype(begin(mat.getCompressedColumnVector(index))),
//             	decltype(begin(mat.getDataVector(index)))            	
//             >
//             >
//             > {            	
//         
//         auto i = begin(mat.getCompressedColumnVector(index));  
//         auto x = begin(mat.getDataVector(index));             
// 		const size_t K = mat.getNumberOfEntries(index);	
//         
//         return { 
//             boost::make_zip_iterator(
//                 boost::make_tuple(i, x)),
//             boost::make_zip_iterator(
//                 boost::make_tuple(i + K, x + K))            
//         };          
//     }
//     
// 	template <class ExpXBetaType, class XBetaType, class YType, class WeightType>    
//     auto getRangeXDependent(const CompressedDataMatrix& mat, const int index, IndicatorTag) -> 
// //            aux::zipper_range<
// 						boost::iterator_range<
//  						boost::zip_iterator<
//  						boost::tuple<	            
// 	            decltype(begin(mat.getCompressedColumnVector(index)))          	
// 	          >
// 	          >
//             > {            	
//         
//         auto i = begin(mat.getCompressedColumnVector(index));             
// 		const size_t K = mat.getNumberOfEntries(index);	
//         
//         return { 
//             boost::make_zip_iterator(
//                 boost::make_tuple(i)),
//             boost::make_zip_iterator(
//                 boost::make_tuple(i + K))            
//         };          
//     }

    
    template <class ExpXBetaType, class XBetaType, class YType, class DenominatorType, class WeightType>
    auto getRangeXIndependent(const CompressedDataMatrix& mat, const int index, 
    			ExpXBetaType& expXBeta, XBetaType& xBeta, YType& y, DenominatorType& denominator, WeightType& weight,
    			IndicatorTag) ->    			    		
    		
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
 					        begin(denominator),
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
 					        begin(denominator),
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
					        begin(denominator),
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
 	
    template <class ExpXBetaType, class XBetaType, class YType, class DenominatorType, class WeightType>
    auto getRangeXIndependent(const CompressedDataMatrix& mat, const int index, 
    			ExpXBetaType& expXBeta, XBetaType& xBeta, YType& y, DenominatorType& denominator, WeightType& weight,
    			SparseTag) ->
    			
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
 					        begin(denominator),
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
 					        begin(denominator),
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
					        begin(denominator),
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
        
	template <class ExpXBetaType, class XBetaType, class YType, class DenominatorType, class WeightType>
    auto getRangeXIndependent(const CompressedDataMatrix& mat, const int index, 
    			ExpXBetaType& expXBeta, XBetaType& xBeta, YType& y, DenominatorType& denominator, WeightType& weight,
    			DenseTag) -> 

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<	            	
		            	decltype(std::begin(expXBeta)),
		            	decltype(std::begin(xBeta)),
		            	decltype(std::begin(y)),
		            	decltype(begin(denominator)),
		            	decltype(std::begin(weight)),
		                decltype(std::begin(mat.getDataVectorSTL(index)))            
        		    >
            	>
            > {            	
        
		const size_t K = mat.getNumberOfRows();	        
        
        auto x0 = std::begin(expXBeta);
        auto x1 = std::begin(xBeta);
        auto x2 = std::begin(y);
        auto x3 = begin(denominator);
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
     	
// START DEPENDENT 		
namespace dependent {

    template <class KeyType, class IteratorType> // For sparse and indicator
    auto getRangeKey(const CompressedDataMatrix& mat, const int index, KeyType& pid, IteratorType) ->
            boost::iterator_range<               
                decltype(boost::make_permutation_iterator(
                    begin(pid),
                    std::begin(mat.getCompressedColumnVectorSTL(index))))                                 
            > {            
        const size_t K = mat.getNumberOfRows();        
        return {
    	    boost::make_permutation_iterator(
			    begin(pid),
				std::begin(mat.getCompressedColumnVectorSTL(index))),
    	    boost::make_permutation_iterator(
			    begin(pid),
				std::end(mat.getCompressedColumnVectorSTL(index)))				
        };             
    }

    template <class KeyType> // For dense
    auto getRangeKey(const CompressedDataMatrix& mat, const int index, KeyType& pid, DenseTag) ->
            boost::iterator_range<
                decltype(begin(pid))
            > {            
        const size_t K = mat.getNumberOfRows();        
        return { 
            begin(pid), 
            begin(pid) + K
        };             
    }
   
    template <class ExpXBeta> // For dense
    auto getRangeX(const CompressedDataMatrix& mat, const int index,
                ExpXBeta& expXBeta, DenseTag) ->
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
    
    template <class ExpXBeta> // For sparse
    auto getRangeX(const CompressedDataMatrix& mat, const int index,
                ExpXBeta& expXBeta, SparseTag) ->
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
    
    template <class ExpXBeta> // For indicator
    auto getRangeX(const CompressedDataMatrix& mat, const int index,
                ExpXBeta& expXBeta, IndicatorTag) ->
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
    
	template <class DenominatorType, class WeightType, class SubsetType, class IteratorType>
    auto getRangeGradient(SubsetType& subset, const size_t N, 
                DenominatorType& denominator, WeightType& weight, 
                IteratorType) ->  // For indicator and sparse
 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple< 					
                        decltype(boost::make_permutation_iterator(
                            begin(denominator),
                            std::begin(subset))),					
 	                    decltype(boost::make_permutation_iterator(
                            std::begin(weight),
                            std::begin(subset)))				          
        		    >
            	>
            > {                				                		       
        return { 
            boost::make_zip_iterator(
                boost::make_tuple(
                    boost::make_permutation_iterator(
                        begin(denominator),
                        std::begin(subset)),					
                    boost::make_permutation_iterator(
                        std::begin(weight),
                        std::begin(subset))
                )),
            boost::make_zip_iterator(
            		boost::make_tuple(
                    boost::make_permutation_iterator(
                        begin(denominator),
                        std::end(subset)),					
                    boost::make_permutation_iterator(
                        std::begin(weight),
                        std::end(subset))
                ))            
        };          
    }            
    
	template <class DenominatorType, class WeightType, class SubsetType> // For dense
    auto getRangeGradient(SubsetType& mat, const size_t N, 
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
        
	template <class ExpXBetaType, class XBetaType, class YType, class DenominatorType, class PidType, class WeightType>
    auto getRangeXDependent(const CompressedDataMatrix& mat, const int index, 
    			ExpXBetaType& expXBeta, XBetaType& xBeta, YType& y, DenominatorType& denominator, PidType& pid, WeightType& weight,
    			DenseTag) -> 

 			boost::iterator_range<
 				boost::zip_iterator<
 					boost::tuple<	            	
		            	decltype(std::begin(expXBeta)),
		            	decltype(std::begin(xBeta)),
		            	decltype(std::begin(y)),
		            	
// 		            	decltype(begin(denominator)),
		            	decltype(boost::make_permutation_iterator(
			        				begin(denominator),
        							begin(pid))),
        							
		            	decltype(begin(pid)),
		            	
// 		            	decltype(std::begin(weight)),
		            	decltype(boost::make_permutation_iterator(
			        				std::begin(weight),
        							begin(pid))),
		            	
		                decltype(std::begin(mat.getDataVectorSTL(index)))            
        		    >
            	>
            > {            	
        
		const size_t K = mat.getNumberOfRows();	        
        
        auto x0 = std::begin(expXBeta);
        auto x1 = std::begin(xBeta);
        auto x2 = std::begin(y);
        
//         auto x3 = begin(denominator);
        auto x3 = boost::make_permutation_iterator(
        				begin(denominator),
        				begin(pid));                
        
        auto x4 = begin(pid);
        
//         auto x5 = std::begin(weight);              
		auto x5 = boost::make_permutation_iterator(
						std::begin(weight),
						begin(pid));
        
        auto x6 = std::begin(mat.getDataVectorSTL(index)); 
        
        auto y0 = std::end(expXBeta);
        auto y1 = std::end(xBeta);
        auto y2 = std::end(y);
        
//         auto y3 = x3 + K;
		auto y3 = boost::make_permutation_iterator(
						begin(denominator),
						x4 + K);
        
        auto y4 = x4 + K;
        
//         auto y5 = x5 + K; // length == K + 1    
		auto y5 = boost::make_permutation_iterator(
						std::begin(weight),
						x4 + K);         
        
        auto y6 = std::end(mat.getDataVectorSTL(index));         
        				                		       
        return { 
            boost::make_zip_iterator(
                boost::make_tuple(
                	x0, x1, x2, x3, x4, x5, x6
                )),
            boost::make_zip_iterator(
                boost::make_tuple(
					y0, y1, y2, y3, y4, y5, y6
                ))            
        };          
    }       

} // namespace dependent

} // namespace helper

} // namespace bsccs

#endif // RANGE_H_