/*
 * LeaveOneOutDriver.h
 *
 *  Created on: Mar 15, 2013
 *      Author: msuchard
 */

#ifndef LEAVEONEOUTDRIVER_H_
#define LEAVEONEOUTDRIVER_H_

#include "AbstractDriver.h"

namespace bsccs {

	class LeaveOneOutDriver : public AbstractDriver {
	public:
		LeaveOneOutDriver(long length);
		virtual ~LeaveOneOutDriver();

		void drive(
				CyclicCoordinateDescent& ccd,
				AbstractSelector& selector,
				const CCDArguments& arguments);

		void logResults(const CCDArguments& arguments);
	private:
		const long length;
	};

} /* namespace bsccs */
#endif /* LEAVEONEOUTDRIVER_H_ */
