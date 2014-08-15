/*
 * SqlModelData.h
 *
 *  Created on: Jun 12, 2014
 *      Author: msuchard
 */

#ifndef SQLMODELDATA_H_
#define SQLMODELDATA_H_

#include "io/ProgressLogger.h"
#include "ModelData.h"

namespace bsccs {


class SqlModelData : public ModelData {

public:

	SqlModelData(
	    ModelType modelType,
	  	loggers::ProgressLoggerPtr log,
    	loggers::ErrorHandlerPtr error
	);
	
// 	size_t append(
//         const std::vector<long>& oStratumId,
//         const std::vector<long>& oRowId,
//         const std::vector<double>& oY,
//         const std::vector<double>& oTime,
//         const std::vector<long>& cRowId,
//         const std::vector<long>& cCovariateId,
//         const std::vector<double>& cCovariateValue        
//     );	
	
	virtual ~SqlModelData();

private:
	// Disable copy-constructors and copy-assignment
	SqlModelData(const ModelData&);
	SqlModelData& operator = (const SqlModelData&);
	
	loggers::ProgressLoggerPtr log;
	loggers::ErrorHandlerPtr error;
};

} /* namespace bsccs */
#endif /* SQLMODELDATA_H_ */
