/*
 * RcppProgressLogger.h
 *
 *  Created on: May 30, 2014
 *      Author: msuchard
 */

#ifndef RCPPPROGRESSLOGGER_H_
#define RCPPPROGRESSLOGGER_H_

#include <sstream>

#include "Rcpp.h"
#include "io/ProgressLogger.h"

namespace bsccs {

namespace loggers {

class RcppProgressLogger : public ProgressLogger {
public:
    RcppProgressLogger(bool _silent = false) : silent(_silent) { }
    
    void setSilent(bool _silent) { silent = _silent; }

    void writeLine(const std::ostringstream& stream) {
        if (!silent) {
            Rcpp::Rcout << stream.str() << std::endl;
        }
    }    
    
    void yield() { 
        R_CheckUserInterrupt();
    }
    
private:
    bool silent;    
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
