/*
 * AbstractDriver.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <numeric>
#include <cmath>
#include "AbstractCrossValidationDriver.h"

namespace bsccs {

AbstractCrossValidationDriver::AbstractCrossValidationDriver(
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error
	) : AbstractDriver(_logger, _error) {
	// Do nothing
}

AbstractCrossValidationDriver::~AbstractCrossValidationDriver() {
	// Do nothing
}

double AbstractCrossValidationDriver::computePointEstimate(const std::vector<double>& value) {
	// Mean of log values, ignoring nans
	double total = 0.0;
	int count = 0;
	for (auto x : value) {
		if (x == x) {		
			total += x;
			count += 1;
		}
	}
	return total / static_cast<double>(count);
}

double AbstractCrossValidationDriver::computeStDev(const std::vector<double>& value, double mean) {
	// Ignoring nans
	double inner_product = 0.0;
	int count = 0;
	for (auto x : value) {
		if (x == x) {
			inner_product += x * x;
			count += 1;
		}
	}	
	return std::sqrt(inner_product / static_cast<double>(count) - mean * mean);	
// 	return std::sqrt(std::inner_product(value.begin(), value.end(), value.begin(), 0.0)
// 	/ static_cast<double>(value.size()) - mean * mean);
}


} // namespace
