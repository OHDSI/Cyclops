/*
 * RcppProgressLogger.h
 *
 *  Created on: May 30, 2014
 *      Author: msuchard
 */

#ifndef RCPPPROGRESSLOGGER_H_
#define RCPPPROGRESSLOGGER_H_

#include <sstream>
#include <deque>

#include "Rcpp.h"
#include "Thread.h"
#include "io/ProgressLogger.h"

namespace bsccs {

namespace loggers {

class RcppProgressLogger : public ProgressLogger {
public:
    RcppProgressLogger(bool _silent = false, bool _concurrent = false) 
    	: silent(_silent), concurrent(_concurrent) { }
    
    void setSilent(bool _silent) { silent = _silent; }
    
    void setConcurrent(bool _concurrent) { concurrent = _concurrent; }

    void writeLine(const std::ostringstream& stream) {
        if (!silent) {
        	if (concurrent) {
        	    lock.lock();
        	    buffer.push_back(stream.str());
        	    lock.unlock();
        	} else {
                Rcpp::Rcout << stream.str() << std::endl;
            }            
        }
    }    
    
    void yield() { 
    	if (!concurrent) {
	        R_CheckUserInterrupt();
	    }
    }
    
    void flush() { 
        if (!concurrent) {       
            lock.lock();
            while (!buffer.empty()) {
                Rcpp::Rcout << buffer.front() << std::endl;
                buffer.pop_front();          
            }
            lock.unlock();
        }
    }
    
private:
    bool silent;
    bool concurrent;
    
    bsccs::mutex lock;    
    std::deque<std::string> buffer;        
};

class RcppErrorHandler : public ErrorHandler {
public:
	RcppErrorHandler(bool _concurrent = false)
		: concurrent(_concurrent) { }
		
	void setConcurrent(bool _concurrent) { concurrent = _concurrent; }		

    void throwError(const std::ostringstream& stream) {
		if (concurrent) {
			lock.lock();
			buffer.push_back(stream.str());
			lock.unlock();			
		} else {
			Rcpp::stop(stream.str());
		}
    }
    
    void flush() {
    	if (!concurrent && !buffer.empty()) {    		
    		std::stringstream stream;    		
    		while (!buffer.empty()) {
    			stream << buffer.front() << std::endl;
    			buffer.pop_front();
    		}
    		Rcpp::stop(stream.str());    		
    	}    
    }
    
private: 
	bool concurrent;
	bsccs::mutex lock; 
	std::deque<std::string> buffer;
};

} // namespace loggers

} // namespace bsccs

#endif /* RCPPPROGRESSLOGGER_H_ */
