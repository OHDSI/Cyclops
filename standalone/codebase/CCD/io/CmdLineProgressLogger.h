/*
 * CmdLineProgressLogger.h
 *
 *  Created on: July 25, 2014
 *      Author: msuchard
 */

#ifndef CMDLINEPROGRESSLOGGER_H_
#define CMDLINEPROGRESSLOGGER_H_

#include <mutex>

#include "io/ProgressLogger.h"

namespace bsccs {

namespace loggers {

class CoutLogger : public ProgressLogger {
public:
    void writeLine(const std::ostringstream& stream) {
        lock.lock();
        std::cout << stream.str() << std::endl;
        lock.unlock();
    }    
    
    void yield() { } // Do nothing
    
private:
    std::mutex lock;    
};

class CerrErrorHandler : public ErrorHandler {
public:
    void throwError(const std::ostringstream& stream) {
        std::cerr << stream.str() << std::endl;
        std::exit(-1);
    }
};

} // namespace loggers

} // namespace bsccs

#endif /* CMDLINEPROGRESSLOGGER_H_ */
