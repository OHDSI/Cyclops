/*
 * RcppOutputHelper.h
 *
 *  Created on: May 30, 2014
 *      Author: msuchard
 */

#ifndef RCPPOUTPUTHELPER_H_
#define RCPPOUTPUTHELPER_H_

// #include "CyclicCoordinateDescent.h"
// #include "ModelData.h"

#include "Rcpp.h"

namespace bsccs {

namespace OutputHelper {

class RcppOutputHelper {
public:				

    RcppOutputHelper(Rcpp::List& _result) : result(_result) { }

	RcppOutputHelper& addDelimitor() { return *this; } 
	  
	RcppOutputHelper& addEndl() { return *this; }
	
	template <typename T>
	RcppOutputHelper& addText(const T& t) { return *this; }
	
	template <typename T> 
	RcppOutputHelper& addHeader(const T& t) { return addText(t); }
		
	template <typename T>
	RcppOutputHelper& addMetaKey(const T& t) { return addText(t).addDelimitor(); }
	
	template <typename T>
	RcppOutputHelper& addMetaValue(const T& t) { return addText(t).addEndl(); }
	
	template <typename T>
	RcppOutputHelper& addValue(const T& t) { return addText(t); }
	
private:
    Rcpp::List& result;
};

} // namespace OutputHelper

} // namespace bsccs

#endif /* RCPPOUTPUTHELPER_H_ */
