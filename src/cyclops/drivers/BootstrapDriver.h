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
#include "ModelData.h"

namespace bsccs {

typedef std::vector<real> rvector;
typedef std::vector<rvector*> rarray;
typedef	rarray::iterator rarrayIterator;

class BootstrapDriver : public AbstractDriver {
public:
	BootstrapDriver(
			int inReplicates,
			AbstractModelData* inModelData,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error
		);

	virtual ~BootstrapDriver();

	virtual void drive(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& arguments);

	virtual void logResults(const CCDArguments& arguments);

	void logResults(const CCDArguments& arguments, std::vector<double>& savedBeta, std::string conditionId);

	void logHR(const CCDArguments& arguments, std::vector<double>& savedBeta, std::string treatmentId);

private:
	const int replicates;
	AbstractModelData* modelData;
	const int J;
	rarray estimates;
};

} // namespace

#endif /* BOOTSTRAPDRIVER_H_ */
