/*
 * ProgressLogger.h
 *
 *  Created on: May 30, 2014
 *      Author: msuchard
 */

#ifndef PROGRESSLOGGER_H_
#define PROGRESSLOGGER_H_

#include <iostream>
#include <sstream>

#include "Types.h"

namespace bsccs {

namespace loggers {

class ProgressLogger {
public:	
	virtual void writeLine(const std::ostringstream& stream) = 0; // pure virtual	
	virtual void yield() = 0; // pure virtual	
};

class CoutLogger : public ProgressLogger {
public:
    void writeLine(const std::ostringstream& stream) {
        std::cout << stream.str() << std::endl;
    }    
    
    void yield() { } // Do nothing
};

typedef bsccs::shared_ptr<ProgressLogger> ProgressLoggerPtr;

class ErrorHandler {
public:
    virtual void throwError(const std::ostringstream& stream) = 0; // pure virtual
};

class CerrErrorHandler : public ErrorHandler {
public:
    void throwError(const std::ostringstream& stream) {
        std::cerr << stream.str() << std::endl;
        exit(-1);
    }
};

typedef bsccs::shared_ptr<ErrorHandler> ErrorHandlerPtr;

} // namespace loggers

} // namespace bsccs

#endif /* PROGRESSLOGGER_H_ */
