/*
 * AbstractDriver.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include "AbstractDriver.h"

namespace bsccs {

AbstractDriver::AbstractDriver(
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error
		) : logger(_logger), error(_error) {
	// Do nothing
}

AbstractDriver::~AbstractDriver() {
	// Do nothing
}

} // namespace
