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

AbstractCrossValidationDriver::AbstractCrossValidationDriver() {
	// Do nothing
}

AbstractCrossValidationDriver::~AbstractCrossValidationDriver() {
	// Do nothing
}

double AbstractCrossValidationDriver::computePointEstimate(const std::vector<double>& value) {
	// Mean of log values
	return accumulate(value.begin(), value.end(), 0.0) / static_cast<double>(value.size());
}

double AbstractCrossValidationDriver::computeStDev(const std::vector<double>& value, double mean) {
	return std::sqrt(std::inner_product(value.begin(), value.end(), value.begin(), 0.0)
	/ static_cast<double>(value.size()) - mean * mean);
}


} // namespace
