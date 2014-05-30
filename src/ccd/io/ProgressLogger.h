/*
 * ProgressLogger.h
 *
 *  Created on: May 30, 2014
 *      Author: msuchard
 */

#ifndef PROGRESSLOGGER_H_
#define PROGRESSLOGGER_H_

#include <sstream>

namespace bsccs {

namespace loggers {

// Define interface

class ProgressLogger {
public:	
	virtual void writeLine(const std::ostringstream& stream) = 0; // pure virtual		
};

class CoutLogger : public ProgressLogger {
public:
    void writeLine(const std::ostringstream& stream) {
        std::cout << stream.str() << std::endl;
    }    
};

typedef bsccs::shared_ptr<ProgressLogger> ProgressLoggerPtr;

}
}

#endif /* PROGRESSLOGGER_H_ */
