/*
 * SqlModelData.cpp
 *
 *  Created on: Jun 12, 2014
 *      Author: msuchard
 */

// #include "Rcpp.h"
#include "SqlModelData.h"
// #include "Timer.h"
// #include "RcppCcdInterface.h"
// #include "io/NewGenericInputReader.h"
// #include "RcppProgressLogger.h"




namespace bsccs {

SqlModelData::SqlModelData(
        ModelType _modelType,
	  	loggers::ProgressLoggerPtr _log,
    	loggers::ErrorHandlerPtr _error) : ModelData(_modelType, _log, _error) {
    // TODO Do something with modelTypeName
}

// size_t SqlModelData::append(
//         const std::vector<long>& oStratumId,
//         const std::vector<long>& oRowId,
//         const std::vector<double>& oY,
//         const std::vector<double>& oTime,
//         const std::vector<long>& cRowId,
//         const std::vector<long>& cCovariateId,
//         const std::vector<double>& cCovariateValue) {
//     // Check covariate dimensions
//     if ((cRowId.size() != cCovariateId.size()) ||
//         (cRowId.size() != cCovariateValue.size())) {
//         std::ostringstream stream;
//         stream << "Mismatched covariate column dimensions";
//         error->throwError(stream);
//     }
//     
//     // TODO Check model-specific outcome dimensions
//     if ((oStratumId.size() != oY.size()) ||
//         (oStratumId.size() != oRowId.size())) {
//         std::ostringstream stream;
//         stream << "Mismatched outcome column dimensions";
//         error->throwError(stream);
//     }
//     const size_t nOutcomes = oStratumId.size();
//     const size_t nCovariates = cCovariateId.size();
//     
//     size_t cOffset = 0;
//     for (size_t i = 0; i < nOutcomes; ++i) {
//         pid.push_back(oStratumId[i]);        
//         y.push_back(oY[i]);
//         
//         long currentRowId = oRowId[i];
//         // TODO Check timing on adding label as string
//         std::stringstream ss;
//         ss << currentRowId;
//         labels.push_back(ss.str());
//         while (cOffset < nCovariates && cRowId[cOffset] == currentRowId) {
//             // Process covariates
//             std::cout << "C: " << cRowId[cOffset] << ":" << cCovariateId[cOffset] << ":" << cCovariateValue[cOffset] << std::endl;
//             ++cOffset;
//         }
//     }            
//     return nOutcomes;   
// }        
 
SqlModelData::~SqlModelData() {
//	std::cout << "~SqlModelData() called." << std::endl;
}

} /* namespace bsccs */
