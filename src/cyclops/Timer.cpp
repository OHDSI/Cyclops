/*
 * Timer.cpp
 *
 *  Created on: Apr 25, 2014
 *      Author: msuchard
 */


#include <cstddef>
#include "Timer.h"

namespace bsccs {

Timer::Timer() {
	gettimeofday(&time1, NULL);
}

double Timer::operator()() {
	struct timeval time2;
	gettimeofday(&time2, NULL);
	return calculateSeconds(time1, time2);
}

Timer::~Timer() { }

double Timer::calculateSeconds(const timeval &time1, const timeval &time2) {
	return time2.tv_sec - time1.tv_sec +
			(double)(time2.tv_usec - time1.tv_usec) / 1000000.0;
}

} /* namespace bsccs */
