/*
 * RcppOutputHelper.h
 *
 *  Created on: May 30, 2014
 *      Author: msuchard
 */

#ifndef RCPPOUTPUTHELPER_H_
#define RCPPOUTPUTHELPER_H_

#include "Rcpp.h"

namespace bsccs {

namespace OutputHelper {

class RcppOutputHelper {
	
	typedef Rcpp::NumericVector Values;
	typedef bsccs::shared_ptr<Values>	ValuesPtr;
	typedef std::vector<ValuesPtr> ValuesVector;
	
public:				

    RcppOutputHelper(Rcpp::List& _result) : result(_result)
        , inMetaData(false), inTable(false)
        , columnCounter(0) { }

	RcppOutputHelper& addDelimitor() { return *this; } 
	  
	RcppOutputHelper& addEndl() { 
	    if (inMetaData) { // finished metadata line
	    
	        inMetaData = false;
	    } else if (inTable) { // finished table line
	    
	        inTable = false;
	    }	    
	    // TODO Fill NA into remaining columns, if any in table
	    columnCounter = 0;
	    return *this; 
	}
	
	template <typename T>
	RcppOutputHelper& addText(const T& t) { return *this; }
	

    RcppOutputHelper& addHeader(const char* t) {        
	    headers.push_back(std::string(t));
	    allValues.push_back(bsccs::make_shared<Values>());
	    return *this; 
	}
		
    RcppOutputHelper& addMetaKey(const char* t) {
	    currentKey = std::string(t);	    
	    return *this; 
	}
	
    RcppOutputHelper& addMetaKey(const std::string& t) {
	    currentKey = std::string(t);	    
	    return *this; 
	}	
	
	template <typename T>
	RcppOutputHelper& addMetaValue(const T& t) { 
	    result[currentKey] = t;
	    return *this; 
	}
	
// 	template <typename T>
//	RcppOutputHelper& addValue(const T& t) { 
	RcppOutputHelper& addValue(const double& t) {
		allValues[columnCounter]->push_back(t);
		columnCounter++;
		return *this; 
	}
	
	RcppOutputHelper& addValue(const string& t) {
		// Ignore
	    return *this;
	}
	
	RcppOutputHelper& endTable(const char* t) {
		Rcpp::DataFrame dataFrame;
		bool any = false;
		for (unsigned int column = 0; column < headers.size(); ++column) {
		    if (allValues[column]->size() > 0) {
    			dataFrame[headers[column]] = *allValues[column];
    			any = true;
    		}
		}
        if (any) {
    		result[t] = dataFrame;	
    	}
		return *this;
	}
	
	bool includeLabels() { return false; }
	
private:
    Rcpp::List& result;
    
    std::vector<std::string> headers;
//    std::vector<bsccs:shared_ptr<std::vector<double> > > values;
		ValuesVector	allValues;
    std::string currentKey;
    
//    bool inHeaders;
    bool inMetaData;
    bool inTable;
    
//    bool useHeaders;
    
    int columnCounter;
};

} // namespace OutputHelper

} // namespace bsccs

#endif /* RCPPOUTPUTHELPER_H_ */
