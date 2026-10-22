/*
 * Timer.h
 *
 *  Created on: Apr 25, 2014
 *      Author: msuchard
 */

#ifndef TIMER_H_
#define TIMER_H_

// #ifndef _MSC_VER
#include <sys/time.h> // was just 'time.h'
// #endif

namespace bsccs {

class Timer {
public:
	Timer();

	double operator()();

	virtual ~Timer();

private:
	double calculateSeconds(const timeval &time1, const timeval &time2);

	struct timeval time1;
};

} /* namespace bsccs */
#endif /* TIMER_H_ */
