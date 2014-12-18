/*
 * RcppProgressLogger.h
 *
 *  Created on: May 30, 2014
 *      Author: msuchard
 */

#ifndef RCPPPROGRESSLOGGER_H_
#define RCPPPROGRESSLOGGER_H_

#include <sstream>
#include <mutex>
#include <deque>

#include "Rcpp.h"
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
    
    std::mutex lock;    
    std::deque<std::string> buffer;        
};

class RcppErrorHandler : public ErrorHandler {
public:
    void throwError(const std::ostringstream& stream) {
       Rcpp::stop(stream.str());
    }
};

} // namespace loggers

} // namespace bsccs

#endif /* RCPPPROGRESSLOGGER_H_ */
