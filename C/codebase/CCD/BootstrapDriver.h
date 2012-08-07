/*
 * BootstrapDriver.h
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */


#ifndef BOOTSTRAPDRIVER_H_
#define BOOTSTRAPDRIVER_H_


#include <vector>

#include "AbstractDriver.h"
#include "InputReader.h"

namespace bsccs {

typedef std::vector<bsccs::real> rvector;
typedef std::vector<rvector*> rarray;
typedef	rarray::iterator rarrayIterator;

class BootstrapDriver : public AbstractDriver {
public:
	BootstrapDriver(
			int inReplicates,
			InputReader* inReader);

	virtual ~BootstrapDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

	virtual void logResults(const CCDArguments& arguments);

	void logResults(const CCDArguments& arguments, std::vector<bsccs::real>& savedBeta, std::string conditionId);

private:
	const int replicates;
	InputReader* reader;
	const int J;
	rarray estimates;
};
}
#endif /* BOOTSTRAPDRIVER_H_ */

